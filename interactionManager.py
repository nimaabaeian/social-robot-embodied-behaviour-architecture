"""executiveControl.py - social interaction state machine."""

import fcntl
import json
import re
import os
import queue
import sqlite3
from dataclasses import asdict, dataclass
import tempfile
import threading
import time
import unicodedata
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ALWAYSON_DIR = os.path.dirname(_MODULE_DIR)

load_dotenv()
load_dotenv(os.path.join(_ALWAYSON_DIR, "memory", "llm.env"), override=False)

from openai import AzureOpenAI

import yarp


class HungerModel:
    def __init__(
        self,
        drain_hours: float = 5.0,
        hungry_threshold: float = 60.0,
        starving_threshold: float = 25.0,
        persist_file: Optional[str] = None,
        log_callback=None,
    ):
        self.level: float = 100.0
        self.drain_hours = drain_hours
        self.hungry_threshold = hungry_threshold
        self.starving_threshold = starving_threshold
        self.last_update_ts: float = time.time()
        self.last_feed_ts: float = 0.0
        self.last_feed_payload: Optional[str] = None
        self._lock = threading.Lock()
        self.last_logged_level: int = 100
        self.persist_file = persist_file
        self.log_callback = log_callback
        self._load_state()

    def _save_state_locked(self) -> None:
        if not self.persist_file:
            return
        try:
            directory = os.path.dirname(self.persist_file) or "."
            os.makedirs(directory, exist_ok=True)
            file_exists_before = os.path.isfile(self.persist_file)
            fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "level": self.level,
                            "last_update_ts": self.last_update_ts,
                            "last_feed_ts": self.last_feed_ts,
                            "last_feed_payload": self.last_feed_payload,
                        },
                        fh,
                    )
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(temp_path, self.persist_file)
                if not file_exists_before and self.log_callback:
                    self.log_callback(
                        "INFO",
                        f"Hunger persistence initialized: created {self.persist_file}",
                    )
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            if self.log_callback:
                self.log_callback("WARNING", f"Failed to persist hunger state: {e}")

    def _load_state(self) -> None:
        if not self.persist_file or not os.path.isfile(self.persist_file):
            return
        try:
            with open(self.persist_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            now = time.time()
            level = float(data.get("level", 100.0))
            level = max(0.0, min(100.0, level))

            self.level = level
            self.last_logged_level = int(level)
            self.last_update_ts = float(data.get("last_update_ts", now))
            if self.last_update_ts <= 0 or self.last_update_ts > now:
                self.last_update_ts = now

            self.last_feed_ts = float(data.get("last_feed_ts", 0.0) or 0.0)
            payload = data.get("last_feed_payload")
            self.last_feed_payload = payload if isinstance(payload, str) else None
        except Exception as e:
            if self.log_callback:
                self.log_callback("WARNING", f"Failed to load hunger state: {e}")

    def set_level(self, level: float, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        with self._lock:
            self.level = max(0.0, min(100.0, float(level)))
            self.last_update_ts = now
            self.last_logged_level = int(self.level)
            self._save_state_locked()

    def update(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        with self._lock:
            old_pct = int(self.level)
            if self.drain_hours <= 0:
                drain_rate = 0.0
            else:
                drain_rate = 100.0 / (self.drain_hours * 3600.0)
            elapsed = now - self.last_update_ts
            if elapsed > 0:
                self.level -= elapsed * drain_rate
            self.last_update_ts = now
            if self.level < 0:
                self.level = 0.0
            elif self.level > 100.0:
                self.level = 100.0

            # Log when hunger drops by 1%
            current_pct = int(self.level)
            if current_pct < self.last_logged_level:
                self.last_logged_level = current_pct
                if self.log_callback:
                    self.log_callback("DEBUG", f"Hunger: {current_pct}%")

            if current_pct != old_pct:
                self._save_state_locked()

    def feed(self, delta: float, payload: str, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        self.update(now)
        with self._lock:
            self.level += delta
            if self.level > 100.0:
                self.level = 100.0
            self.last_feed_ts = now
            self.last_feed_payload = payload
            self.last_logged_level = int(self.level)
            self._save_state_locked()

    def get_state(self) -> str:
        self.update()
        with self._lock:
            if self.level >= self.hungry_threshold:
                return "HS1"
            elif self.level >= self.starving_threshold:
                return "HS2"
            else:
                return "HS3"


class ExecutiveControlModule(yarp.RFModule):
    """Runs social-state interaction trees and responsive interactions.

    Pipeline:
      Input       -> salience run RPC + landmarks stream + STT + QR feed
      Decision    -> social-state tree / hunger tree / responsive arbitration
      Output      -> TTS speech + selector/vision RPC commands + compact result
      Persistence -> SQLite logs + greeted/last_greeted JSON updates
    """

    @dataclass(frozen=True)
    class FaceSnapshot:
        face_id: str = "unknown"
        track_id: int = -1
        bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        attention: str = ""
        time_in_view: float = 0.0

    @dataclass(frozen=True)
    class InteractionAttempt:
        interaction_id: str
        track_id: int
        face_id: str
        initial_state: str
        result: Dict[str, Any]

    @dataclass(frozen=True)
    class LearningDelta:
        person_id: str
        reward_delta: float
        outcome: str
        reason: str
        success: int
        abort_reason: Optional[str] = None
        name_extracted: int = 0

    # ==================== Constants ====================

    _PROMPTS_FILE_CANDIDATES = [
        os.path.join(_MODULE_DIR, "prompts.json"),
        os.path.join(_ALWAYSON_DIR, "prompts.json"),
    ]
    _im_prompts: Dict[str, Any] = {}

    # Loaded from prompts.json
    LLM_SYSTEM_DEFAULT = ""
    LLM_SYSTEM_JSON = ""

    @classmethod
    def _load_im_prompts(cls) -> None:
        """Load (or reload) executiveControl prompts from prompts.json."""
        prompts_file = None
        for candidate in cls._PROMPTS_FILE_CANDIDATES:
            if os.path.isfile(candidate):
                prompts_file = candidate
                break

        if prompts_file is None:
            print(
                "[ERROR] prompts.json not found. Tried: "
                + ", ".join(cls._PROMPTS_FILE_CANDIDATES)
            )
            cls._im_prompts = {}
            return

        try:
            with open(prompts_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[ERROR] Failed to load prompts.json ({prompts_file}): {e}")
            cls._im_prompts = {}
            return

        cls._im_prompts = data.get("executiveControl", {})

        system_default = cls._im_prompts.get("system_default")
        system_json = cls._im_prompts.get("system_json")

        if system_default:
            cls.LLM_SYSTEM_DEFAULT = system_default
        else:
            print("[ERROR] Missing executiveControl.system_default in prompts.json")

        if system_json:
            cls.LLM_SYSTEM_JSON = system_json
        else:
            print("[ERROR] Missing executiveControl.system_json in prompts.json")

    DB_FILE = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/data_collection/executive_control.db"
    TELEGRAM_DB_FILE = os.path.join(_ALWAYSON_DIR, "memory", "chat_bot.db")
    LAST_GREETED_FILE = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/last_greeted.json"
    GREETED_TODAY_FILE = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/greeted_today.json"
    HUNGER_STATE_FILE = os.path.join(_ALWAYSON_DIR, "memory", "hunger_state.json")

    # Timeouts (seconds)
    SS1_STT_TIMEOUT = 10.0
    SS2_STT_TIMEOUT = 10.0
    SS2_GREET_TIMEOUT = 10.0
    SS3_STT_TIMEOUT = 12.0
    LLM_TIMEOUT = 60.0

    # SS3 conversation limits
    SS3_MAX_TURNS = 3
    SS3_MAX_TIME = 120.0

    VALID_STATES = {"ss1", "ss2", "ss3", "ss4"}

    # Target monitor
    MONITOR_HZ = 15.0
    TARGET_LOST_TIMEOUT = 12.0  # Continuous absence needed before abort

    # Responsive interactions
    RESPONSIVE_GREET_REGEX = re.compile(
        r"\b(hello|hi|hey|ciao|buongiorno|good\s+morning)\b"
    )
    RESPONSIVE_GREET_COOLDOWN_SEC = 10.0
    RESPONSIVE_ALLOWED_ATTENTION = {"MUTUAL_GAZE", "NEAR_GAZE"}
    LLM_ASYNC_DEADLINE_SEC = 2.5
    LLM_ASYNC_LOCAL_WAIT_SEC = 1.0
    LLM_ASYNC_NAME_WAIT_SEC = 1.0
    LLM_ASYNC_POLL_SEC = 0.05
    DB_QUEUE_MAXSIZE = 512

    # ==================== Lifecycle ====================

    def __init__(self):
        super().__init__()
        self.module_name = "executiveControl"
        self.period = 1.0
        self._running = True
        self.run_lock = threading.Lock()
        self.log_buffer: List[Dict] = []

        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # YARP ports
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.stt_port: Optional[yarp.BufferedPortBottle] = None

        # RPC clients (lazy)
        self._selector_rpc: Optional[yarp.RpcClient] = None
        self._vision_rpc: Optional[yarp.RpcClient] = None

        # speech_port is BufferedPortBottle so write() is non-blocking
        self.speech_port: Optional[yarp.BufferedPortBottle] = None

        self._cached_starter: Optional[str] = None

        # Landmarks reader
        self._faces_lock = threading.Lock()
        self._latest_faces: List[Dict] = []
        self._latest_faces_ts: float = 0.0
        self._landmarks_reader_stop = threading.Event()
        self._landmarks_reader_thread: Optional[threading.Thread] = None

        # LLM async worker
        self._llm_req_queue: queue.Queue = queue.Queue(maxsize=64)
        self._llm_req_seq = 0
        self._llm_req_seq_lock = threading.Lock()
        self._llm_res_lock = threading.Lock()
        self._llm_results: Dict[int, Any] = {}
        self._llm_cancelled: set[int] = set()
        self._llm_worker_stop = threading.Event()
        self._llm_worker_thread: Optional[threading.Thread] = None

        # Abort mechanism
        self.abort_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._current_track_id: Optional[int] = None

        # Azure LLM clients (initialised in configure via setup_azure_llms)
        self.llm_extract: Optional[AzureOpenAI] = None
        self.llm_chat: Optional[AzureOpenAI] = None
        self._llm_deployment: str = ""
        self._llm_max_tokens: int = 2000

        # Error recovery
        self.llm_retry_attempts = 3
        self.llm_retry_delay = 1.0

        # DB queue
        self._db_queue: queue.Queue = queue.Queue(maxsize=self.DB_QUEUE_MAXSIZE)
        self._db_thread: Optional[threading.Thread] = None

        # Hunger and QR
        self.hunger = HungerModel(persist_file=self.HUNGER_STATE_FILE)
        self.hunger_port: Optional[yarp.BufferedPortBottle] = (
            None  # output to chatBot
        )
        self.qr_port: Optional[yarp.BufferedPortBottle] = None
        self._qr_thread: Optional[threading.Thread] = None
        self._qr_stop_event = threading.Event()
        self._qr_cooldown_sec = 3.0
        self._last_scan_ts = 0.0
        self._last_scan_payload: Optional[str] = None
        self._feed_condition = threading.Condition()
        self._feed_wait_timeout_sec = 8.0
        self._meal_mapping = {
            "SMALL_MEAL": 10.0,
            "MEDIUM_MEAL": 25.0,
            "LARGE_MEAL": 45.0,
        }

        # Responsive interaction path
        self._responsive_stop_event = threading.Event()
        self._responsive_thread: Optional[threading.Thread] = None
        self._responsive_queue: queue.Queue = queue.Queue(maxsize=32)
        self._responsive_active = threading.Event()
        self._responsive_greet_cooldown: Dict[str, float] = {}

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            ExecutiveControlModule._load_im_prompts()

            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            self.setName(self.module_name)

            drain_hours = (
                rf.find("drain_hours").asFloat64() if rf.check("drain_hours") else 5.0
            )
            hungry_th = (
                rf.find("hungry_threshold").asFloat64()
                if rf.check("hungry_threshold")
                else 60.0
            )
            starving_th = (
                rf.find("starving_threshold").asFloat64()
                if rf.check("starving_threshold")
                else 25.0
            )
            self.hunger = HungerModel(
                drain_hours=drain_hours,
                hungry_threshold=hungry_th,
                starving_threshold=starving_th,
                persist_file=self.HUNGER_STATE_FILE,
                log_callback=self._log,
            )
            hs_boot = self.hunger.get_state()
            with self.hunger._lock:
                lvl_boot = self.hunger.level
            self._log(
                "INFO",
                f"Hunger restored: {lvl_boot:.1f}% ({hs_boot}) from {self.HUNGER_STATE_FILE}",
            )

            if rf.check("qr_cooldown_sec"):
                self._qr_cooldown_sec = rf.find("qr_cooldown_sec").asFloat64()
            if rf.check("feed_wait_timeout_sec"):
                self._feed_wait_timeout_sec = rf.find(
                    "feed_wait_timeout_sec"
                ).asFloat64()
            if rf.check("llm_async_deadline_sec"):
                self.LLM_ASYNC_DEADLINE_SEC = rf.find("llm_async_deadline_sec").asFloat64()
            if rf.check("llm_async_local_wait_sec"):
                self.LLM_ASYNC_LOCAL_WAIT_SEC = rf.find("llm_async_local_wait_sec").asFloat64()

            self.handle_port.open("/" + self.module_name)

            self.landmarks_port = yarp.BufferedPortBottle()
            self.stt_port = yarp.BufferedPortBottle()
            self.speech_port = yarp.BufferedPortBottle()  # non-blocking write
            self.qr_port = yarp.BufferedPortBottle()

            def _open_and_log(port: yarp.Port, port_name: str) -> bool:
                if not port.open(port_name):
                    self._log("ERROR", f"Failed to open port: {port_name}")
                    return False
                self._log("INFO", f"Port open: {port_name}")
                return True

            ports = [
                (self.landmarks_port, "landmarks:i"),
                (self.stt_port, "stt:i"),
                (self.speech_port, "speech:o"),
            ]
            for port, suffix in ports:
                pn = f"/alwayson/{self.module_name}/{suffix}"
                if not _open_and_log(port, pn):
                    return False

            qr_pn = f"/alwayson/{self.module_name}/qr:i"
            if not _open_and_log(self.qr_port, qr_pn):
                self._log(
                    "WARNING",
                    f"QR port unavailable ({qr_pn}); continuing without QR input",
                )
                self.qr_port = None

            self.hunger_port = yarp.BufferedPortBottle()
            hunger_pn = f"/alwayson/{self.module_name}/hunger:o"
            if not _open_and_log(self.hunger_port, hunger_pn):
                self._log(
                    "WARNING",
                    f"Hunger port unavailable ({hunger_pn}); continuing without hunger publish",
                )
                self.hunger_port = None

            self._ensure_json_file(self.LAST_GREETED_FILE, {})
            self._ensure_json_file(self.GREETED_TODAY_FILE, {})
            self._init_db()

            self._landmarks_reader_stop.clear()
            self._landmarks_reader_thread = threading.Thread(
                target=self._landmarks_reader_loop, daemon=True
            )
            self._landmarks_reader_thread.start()

            self._db_thread = threading.Thread(target=self._db_worker, daemon=True)
            self._db_thread.start()

            self._log(
                "INFO",
                f"QR: connect /alwayson/vision/qr:o → /alwayson/{self.module_name}/qr:i manually",
            )

            self._qr_stop_event.clear()
            self._qr_thread = threading.Thread(target=self._qr_reader_loop, daemon=True)
            self._qr_thread.start()

            self._responsive_stop_event.clear()
            self._responsive_thread = threading.Thread(
                target=self._responsive_loop, daemon=True
            )
            self._responsive_thread.start()

            self.setup_azure_llms()
            self._start_llm_worker()
            threading.Thread(
                target=self._generate_starter_background, daemon=True
            ).start()

            self._log("INFO", "ExecutiveControlModule ready")
            return True
        except Exception as e:
            self._log("ERROR", f"configure() failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def interruptModule(self) -> bool:
        self._log("INFO", "Interrupting...")
        self._running = False
        self.abort_event.set()
        self._landmarks_reader_stop.set()
        self._qr_stop_event.set()
        self._responsive_stop_event.set()
        self._stop_llm_worker()
        self.handle_port.interrupt()
        for port in [
            self.landmarks_port,
            self.stt_port,
            self.speech_port,
            self.hunger_port,
        ]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        self._log("INFO", "Closing...")
        self._landmarks_reader_stop.set()
        self._qr_stop_event.set()
        self._responsive_stop_event.set()
        self._stop_llm_worker()
        if self._landmarks_reader_thread:
            self._landmarks_reader_thread.join(timeout=2.0)
        if self._qr_thread:
            self._qr_thread.join(timeout=2.0)
        if self._responsive_thread:
            self._responsive_thread.join(timeout=2.0)
        self._queue_put_drop_oldest(self._db_queue, None, "DB queue close")
        if self._db_thread:
            self._db_thread.join(timeout=3.0)
        self.handle_port.close()
        for port in [
            self.landmarks_port,
            self.stt_port,
            self.speech_port,
            self.hunger_port,
        ]:
            if port:
                port.close()
        if self._selector_rpc:
            self._selector_rpc.close()
        if self._vision_rpc:
            self._vision_rpc.close()
        return True

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        self.hunger.update()
        # Publish current hunger state so chatBot (and any other subscriber) can read it.
        if self.hunger_port:
            hs = self.hunger.get_state()
            b = self.hunger_port.prepare()
            b.clear()
            b.addString(hs)
            self.hunger_port.write()
        return self._running

    # ==================== RPC Handler ====================

    def respond(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        reply.clear()
        try:
            if cmd.size() < 1:
                return self._reply_error(reply, "Empty command")

            command = cmd.get(0).asString()

            if command in ["status", "ping"]:
                is_locked = self._is_run_lock_busy()
                busy = is_locked or self._responsive_active.is_set()
                return self._reply_ok(
                    reply,
                    {
                        "success": True,
                        "status": "ready",
                        "module": self.module_name,
                        "busy": busy,
                    },
                )

            if command == "help":
                reply.clear()
                reply.addString(
                    "run <track_id> <face_id> <ss1|ss2|ss3|ss4>  -- start interaction\n"
                    "hunger <hs1|hs2|hs3>                         -- set hunger (hs1=full, hs2=hungry, hs3=starving)\n"
                    "status                                        -- check if busy\n"
                    "quit                                          -- shut down"
                )
                return True

            if command == "quit":
                self._running = False
                self.stopModule()
                return self._reply_ok(
                    reply, {"success": True, "message": "Shutting down"}
                )

            if command == "hunger":
                if cmd.size() < 2:
                    return self._reply_error(reply, "Usage: hunger <hs1|hs2|hs3>")
                level_arg = cmd.get(1).asString().lower()
                level_map = {"hs1": 100.0, "hs2": 59.0, "hs3": 24.0}
                if level_arg not in level_map:
                    return self._reply_error(
                        reply,
                        "Invalid hunger state. Use: hs1 (100%), hs2 (59%), hs3 (24%)",
                    )
                new_level = level_map[level_arg]
                self.hunger.set_level(new_level, now=time.time())
                hs = self.hunger.get_state()
                self._log("INFO", f"Hunger manually set to {new_level}% ({hs})")
                return self._reply_ok(
                    reply,
                    {"success": True, "hunger_level": new_level, "hunger_state": hs},
                )

            if command != "run":
                return self._reply_error(reply, f"Unknown command: {command}")

            if self._responsive_active.is_set():
                return self._reply_ok(
                    reply,
                    {
                        "success": False,
                        "error": "responsive_interaction_running",
                    },
                )

            if cmd.size() < 4:
                return self._reply_error(
                    reply, "Usage: run <track_id> <face_id> <ss1|ss2|ss3|ss4>"
                )

            track_id = cmd.get(1).asInt32()
            face_id = cmd.get(2).asString()
            social_state = cmd.get(3).asString().lower()

            if social_state not in self.VALID_STATES:
                return self._reply_error(reply, f"Invalid state: {social_state}")

            if not self.run_lock.acquire(blocking=False):
                return self._reply_error(reply, "Another action is running")

            try:
                self.log_buffer = []
                interaction_id = uuid.uuid4().hex
                self._log(
                    "INFO",
                    f"--- Interaction start: id={interaction_id} track={track_id} face={face_id} state={social_state} ---",
                )
                self.ensure_stt_ready("english")

                result = self._execute_interaction(track_id, face_id, social_state)
                result["interaction_id"] = interaction_id

                # DB save async
                interaction_attempt = self.InteractionAttempt(
                    interaction_id=interaction_id,
                    track_id=track_id,
                    face_id=face_id,
                    initial_state=social_state,
                    result=dict(result),
                )
                self._enqueue_db_event(
                    (
                        "interaction",
                        asdict(interaction_attempt),
                    )
                )

                ar = result.get("abort_reason")
                compact_abort_reason = None
                if ar:
                    if ar in (
                        "target_lost",
                        "target_not_biggest",
                        "target_monitor_abort",
                    ):
                        # If the user responded at least once in SS3, target loss is not
                        # a failure — negative reward is reserved for no response at all.
                        if not result.get("talked"):
                            compact_abort_reason = "face_disappeared"
                    else:
                        # Only apply not_responded penalty when user never spoke at all
                        if not result.get("talked"):
                            compact_abort_reason = "not_responded"

                # Success if explicitly succeeded, or if user talked (even if aborted after)
                compact_success = bool(result.get("success", False)) or (
                    result.get("talked", False) and compact_abort_reason is None
                )

                compact = {
                    "interaction_id": interaction_id,
                    "success": compact_success,
                    "track_id": track_id,
                    "name": None,
                    "name_extracted": False,
                    "abort_reason": compact_abort_reason,
                    "initial_state": social_state,
                    "final_state": result.get("final_state", social_state),
                }

                if result.get("extracted_name"):
                    compact["name"] = result.get("extracted_name")
                    compact["name_extracted"] = True
                elif social_state in ("ss2", "ss3", "ss4"):
                    compact["name"] = face_id

                for extra in [
                    "interaction_tag",
                    "hunger_state_start",
                    "hunger_state_end",
                    "stomach_level_start",
                    "stomach_level_end",
                    "meals_eaten_count",
                    "last_meal_payload",
                ]:
                    if extra in result:
                        compact[extra] = result[extra]

                reply.addString("ok")
                reply.addString(json.dumps(compact, ensure_ascii=False))
            finally:
                self.run_lock.release()

            return True
        except Exception as e:
            self._log("ERROR", f"Exception in respond: {e}")
            import traceback

            traceback.print_exc()
            try:
                self.run_lock.release()
            except RuntimeError:
                pass
            return self._reply_error(reply, str(e))

    def _reply_ok(self, reply: yarp.Bottle, data: Dict) -> bool:
        reply.addString("ok")
        reply.addString(json.dumps(data, ensure_ascii=False))
        return True

    def _reply_error(self, reply: yarp.Bottle, error: str) -> bool:
        reply.addString("ok")
        reply.addString(
            json.dumps(
                {"success": False, "error": error, "logs": self.log_buffer},
                ensure_ascii=False,
            )
        )
        return True

    def _is_run_lock_busy(self) -> bool:
        is_locked = not self.run_lock.acquire(blocking=False)
        if not is_locked:
            self.run_lock.release()
        return is_locked

    # ==================== Interaction Execution ====================

    def _execute_interaction(
        self, track_id: int, face_id: str, social_state: str
    ) -> Dict[str, Any]:
        result = {
            "success": False,
            "initial_state": social_state,
            "final_state": social_state,
            "greeted": False,
            "talked": False,
            "extracted_name": None,
            "abort_reason": None,
            "target_stayed_biggest": True,
            "logs": [],
        }

        # ss4 is no-op
        if social_state == "ss4":
            result["success"] = True
            result["final_state"] = "ss4"
            result["logs"] = self.log_buffer.copy()
            self._log("INFO", "ss4: no-op")
            return result

        # Resolve face_id
        if not self._is_face_id_resolved(face_id):
            self._log("INFO", "face_id unresolved – waiting...")
            face_id = self._wait_face_resolve(track_id, face_id, 5.0)
            if not self._is_face_id_resolved(face_id):
                result["abort_reason"] = "face_id_unresolved"
                result["logs"] = self.log_buffer.copy()
                return result

        # Start target monitor
        self._start_monitor(track_id, result)

        try:
            hs = self.hunger.get_state()
            result["hunger_state_start"] = hs
            with self.hunger._lock:
                result["stomach_level_start"] = self.hunger.level
            result["interaction_tag"] = f"{social_state.upper()}{hs}"

            if hs == "HS3" or (hs == "HS2" and social_state == "ss3"):
                self._run_hunger_feed_tree(track_id, face_id, social_state, hs, result)
            else:
                if social_state == "ss1":
                    self._run_ss1_tree(track_id, face_id, result)
                elif social_state == "ss2":
                    self._run_ss2_tree(track_id, face_id, result)
                elif social_state == "ss3":
                    self._run_ss3_tree(track_id, face_id, result)

            hs_end = self.hunger.get_state()
            result["hunger_state_end"] = hs_end
            with self.hunger._lock:
                result["stomach_level_end"] = self.hunger.level
        except Exception as e:
            self._log("ERROR", f"Tree execution error: {e}")
            result["abort_reason"] = f"exception: {e}"

        # Stop monitor
        self._stop_monitor()

        result["logs"] = self.log_buffer.copy()
        return result

    # ==================== Target Monitor ====================

    def _target_monitor_loop(self, track_id: int, result: Dict):
        """Abort when track_id stays absent for TARGET_LOST_TIMEOUT."""
        last_seen = time.time()
        last_iter_ts = time.time()

        while not self.abort_event.is_set():
            now = time.time()
            if now - last_iter_ts > 1.5:
                # Thread starvation guard.
                last_seen = now
            last_iter_ts = now

            try:
                # Ignore short landmark hiccups.
                faces = self.parse_landmarks_latest(staleness_sec=5.0)
                found = any(f.get("track_id") == track_id for f in faces)

                if found:
                    last_seen = time.time()
                else:
                    elapsed = time.time() - last_seen
                    if elapsed > self.TARGET_LOST_TIMEOUT:
                        self._log(
                            "WARNING",
                            f"Monitor: track_id {track_id} absent for {elapsed:.1f}s",
                        )
                        result["target_stayed_biggest"] = False
                        result["abort_reason"] = "target_lost"
                        self.abort_event.set()
                        return
            except Exception as e:
                self._log("WARNING", f"Monitor error: {e}")

            time.sleep(1.0 / self.MONITOR_HZ)

    def _start_monitor(self, track_id: int, result: Dict):
        self.abort_event.clear()
        self._current_track_id = track_id
        self._monitor_thread = threading.Thread(
            target=self._target_monitor_loop,
            args=(track_id, result),
            daemon=True,
        )
        self._monitor_thread.start()

    def _stop_monitor(self):
        self.abort_event.set()
        self._current_track_id = None
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

    def _reset_selector_cooldown(self, face_id: str, track_id: int):
        """Ask salienceNetwork to reset proactive cooldown for this person."""
        try:
            rpc = self._get_selector_rpc()
            cmd = yarp.Bottle()
            cmd.addString("reset_cooldown")
            cmd.addString(face_id)
            cmd.addInt32(track_id)
            reply = yarp.Bottle()
            rpc.write(cmd, reply)
        except Exception as e:
            self._log("WARNING", f"reset_selector_cooldown failed: {e}")

    @staticmethod
    def _face_area(face: Dict) -> float:
        bbox = face.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return bbox[2] * bbox[3]
        return 0.0

    # ==================== Tree SS1: Unknown ====================

    def _run_ss1_tree(self, track_id: int, face_id: str, result: Dict):
        """
        1) Greet via TTS
        2) Wait response  → if none → abort
        3) Ask name       → if none → abort
        4) Extract name   → retry once → if fail → abort
        5) Say 'Nice to meet you', register name, update last_greeted
        """
        self._log("INFO", "SS1: start")

        # 1) Greet via TTS
        self._clear_stt_buffer()
        self._speak_and_wait(self._im_prompts.get("ss1_greeting", "Hi there!"))
        result["greeted"] = True

        # 2) Wait for response
        if self._check_abort(result):
            return
        utterance = self._wait_user_utterance_abortable(self.SS1_STT_TIMEOUT)
        if not utterance:
            self._log("WARNING", "SS1: no response to greeting")
            result["abort_reason"] = (
                result.get("abort_reason") or "no_response_greeting"
            )
            return

        self._log("INFO", f"SS1: response: '{utterance}'")
        if self._check_abort(result):
            return

        # 3) Ask name
        self._clear_stt_buffer()
        self._speak(
            self._im_prompts.get("ss1_ask_name", "We have not met, what's your name?")
        )
        if self._check_abort(result):
            return

        name_utterance = self._wait_user_utterance_abortable(self.SS2_STT_TIMEOUT)
        if not name_utterance:
            self._log("WARNING", "SS1: no response to name question")
            result["abort_reason"] = result.get("abort_reason") or "no_response_name"
            return

        # 4) Extract name (retry once)
        name = self._try_extract_name(name_utterance)
        if not name:
            self._log("INFO", "SS1: retrying name extraction")
            if self._check_abort(result):
                return
            self._clear_stt_buffer()
            self._speak(
                self._im_prompts.get(
                    "ss1_ask_name_retry",
                    "Sorry, I didn't catch that. What's your name?",
                )
            )
            name_utterance2 = self._wait_user_utterance_abortable(self.SS2_STT_TIMEOUT)
            if name_utterance2:
                name = self._try_extract_name(name_utterance2)

        if not name:
            self._log("WARNING", "SS1: name extraction failed")
            result["abort_reason"] = "name_extraction_failed"
            return

        if self._check_abort(result):
            return

        # 5) Register and finish
        self._log("INFO", f"SS1: name extracted: '{name}'")
        result["extracted_name"] = name

        threading.Thread(
            target=self._submit_face_name, args=(track_id, name), daemon=True
        ).start()
        threading.Thread(
            target=self._write_last_greeted,
            args=(track_id, face_id, name, name),
            daemon=True,
        ).start()

        self._speak_and_wait(
            self._im_prompts.get("ss1_nice_to_meet", "Nice to meet you")
        )

        result["success"] = True
        result["greeted"] = True
        result["final_state"] = "ss3"
        self._log("INFO", "SS1: complete")

    # ==================== Tree SS2: Known, not greeted ====================

    def _run_ss2_tree(self, track_id: int, face_id: str, result: Dict):
        """
        1) Say "Hi <name>"
        2) If responded → go to ss3 tree
        3) If not → say "Hi <name>" again
        4) If responded → go to ss3 tree
        5) Else abort
        """
        self._log("INFO", f"SS2: start for '{face_id}'")

        if face_id.lower() in ("unknown", "unmatched") or face_id.isdigit():
            self._log("WARNING", f"SS2: invalid name '{face_id}' – aborting")
            result["abort_reason"] = "invalid_name"
            return

        for attempt in range(2):
            self._log("INFO", f"SS2: greeting attempt {attempt+1}/2")
            if self._check_abort(result):
                return

            self._clear_stt_buffer()
            _greeting_tpl = self._im_prompts.get("ss2_greeting", "Hello {name}")
            self._speak_and_wait(_greeting_tpl.format(name=face_id))
            result["greeted"] = True

            if self._check_abort(result):
                return

            utterance = self._wait_user_utterance_abortable(self.SS2_GREET_TIMEOUT)
            if utterance:
                self._log("INFO", f"SS2: response: '{utterance}'")
                threading.Thread(
                    target=self._write_last_greeted,
                    args=(track_id, face_id, face_id, face_id),
                    daemon=True,
                ).start()
                result["success"] = True
                result["final_state"] = "ss3"
                self._log("INFO", "SS2: → ss3 tree")
                self._run_ss3_tree(track_id, face_id, result)
                return
            else:
                self._log("WARNING", f"SS2: no response attempt {attempt+1}")

        result["abort_reason"] = "no_response_greeting"
        self._log("INFO", "SS2: failed after 2 attempts")

    # ==================== Telegram User Lookup ====================

    def _lookup_telegram_user(self, face_name: str) -> Optional[Dict[str, Any]]:
        """Look up a face name in chat_bot.db user_memory by name or nickname.

        Matching is case-insensitive. Returns the parsed user record dict on
        match, or None if the DB is missing / no match is found.
        """
        if not face_name or face_name.lower() in (
            "unknown",
            "unmatched",
            "recognizing",
        ):
            return None

        db_path = self.TELEGRAM_DB_FILE
        if not os.path.isfile(db_path):
            self._log("DEBUG", f"Telegram DB not found: {db_path}")
            return None

        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            rows = conn.execute("SELECT chat_id, data_json FROM user_memory").fetchall()
            conn.close()
        except Exception as e:
            self._log("WARNING", f"Telegram DB read failed: {e}")
            return None

        face_norm = self._normalize_name(face_name)
        best_record: Optional[Dict[str, Any]] = None

        for _chat_id, data_json in rows:
            try:
                record = json.loads(data_json or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict):
                continue

            db_name = self._normalize_name(record.get("name") or "")
            db_nick = self._normalize_name(record.get("nickname") or "")

            if db_name and db_name == face_norm:
                best_record = record
                break  # exact name match — done
            if db_nick and db_nick == face_norm:
                best_record = record
                break  # exact nickname match — done
            # Partial: face_name is a first-name substring of the DB name
            if db_name and face_norm in db_name.split():
                best_record = record
                # keep looking for an exact match
            elif db_nick and face_norm in db_nick.split():
                if best_record is None:
                    best_record = record

        if best_record:
            self._log(
                "INFO",
                f"Telegram user match for '{face_name}': "
                f"name={best_record.get('name')}, nick={best_record.get('nickname')}",
            )
        else:
            self._log("DEBUG", f"No telegram user match for '{face_name}'")

        return best_record

    @staticmethod
    def _build_face_user_context(record: Dict[str, Any]) -> str:
        """Build a concise spoken-context string from a telegram user record.

        Adapted from ChatBotModule._build_user_context but tuned for
        face-to-face interaction (no texting style, no emoji references).
        Returns empty string if record has no useful data.
        """
        if not record or not isinstance(record, dict):
            return ""

        def _s(v: Any, mx: int = 60) -> str:
            if not v or not isinstance(v, str):
                return ""
            return " ".join(v.split())[:mx]

        def _sl(v: Any, mx: int = 5) -> List[str]:
            if not isinstance(v, list):
                return []
            return [s for s in (_s(item) for item in v) if s][:mx]

        parts: List[str] = []

        name = _s(record.get("name"))
        nickname = _s(record.get("nickname"))
        age = record.get("age")
        if name:
            frag = f"Their name is {name}."
            if nickname:
                frag += f" They go by {nickname}."
            if age and isinstance(age, int) and 5 <= age <= 120:
                frag += f" They are {age} years old."
            parts.append(frag)
        elif age and isinstance(age, int) and 5 <= age <= 120:
            parts.append(f"They are {age} years old.")

        likes = _sl(record.get("likes"))
        topics = _sl(record.get("favorite_topics"))
        interests = list(dict.fromkeys(topics + likes))[:5]
        if interests:
            parts.append(f"They like: {', '.join(interests)}.")

        dislikes = _sl(record.get("dislikes"), mx=3)
        if dislikes:
            parts.append(f"They dislike: {', '.join(dislikes)}.")

        update = _s(record.get("last_personal_update"), mx=80)
        if update:
            parts.append(f"Recent life update: {update}.")

        jokes_raw = record.get("inside_jokes")
        jokes_confirmed: List[str] = []
        if isinstance(jokes_raw, dict):
            ranked = sorted(
                (
                    (phrase, meta)
                    for phrase, meta in jokes_raw.items()
                    if isinstance(phrase, str) and isinstance(meta, dict) and int(meta.get("count", 0) or 0) >= 2
                ),
                key=lambda item: int(item[1].get("last_seen", 0) or 0),
            )
            jokes_confirmed = [
                " ".join(p.split())[:80]
                for p, _ in ranked
                if isinstance(p, str) and len(p.strip()) >= 3
            ]
        elif isinstance(jokes_raw, list):  # backward-compatible read of older records
            jokes_confirmed = _sl(jokes_raw, mx=3)

        if jokes_confirmed:
            parts.append(f"Inside joke: {jokes_confirmed[-1]}.")

        ctx = " ".join(parts)
        if len(ctx) > 500:
            ctx = ctx[:497].rsplit(" ", 1)[0] + "..."
        return ctx

    # ==================== Tree SS3: Known, greeted, not talked ====================

    def _run_ss3_tree(self, track_id: int, face_id: str, result: Dict):
        """
        Short conversation with proactive starter.
        Up to 3 turns; turn 3 is acknowledgment only.
        If at least one response → talked=True → ss4.
        If the person is known in the Telegram DB, their profile is used
        to personalise the opening and follow-up LLM responses.
        """
        self._log("INFO", "SS3: start")

        if self._check_abort(result):
            return

        # --- Telegram user lookup (best-effort, never blocks interaction) ---
        user_context = ""
        try:
            tg_record = self._lookup_telegram_user(face_id)
            if tg_record:
                user_context = self._build_face_user_context(tg_record)
                if user_context:
                    self._log(
                        "INFO", f"SS3: personalised context ({len(user_context)} chars)"
                    )
                    result["telegram_user_matched"] = True
        except Exception as e:
            self._log(
                "WARNING", f"SS3: telegram lookup failed (continuing without): {e}"
            )

        # --- Opening line ---
        # Never block on remote LLM service: submit async request and use a bounded
        # local wait before falling back.
        fallback_starter = self._cached_starter or self._im_prompts.get(
            "convo_starter_fallback", "How are you doing these days?"
        )
        starter_req = self._submit_llm_request(
            kind="starter",
            utterance="",
            user_context=user_context,
            timeout_sec=self.LLM_ASYNC_DEADLINE_SEC,
        )
        starter = self._consume_llm_result_or_default(
            req_id=starter_req,
            default=fallback_starter,
            local_wait_sec=self.LLM_ASYNC_LOCAL_WAIT_SEC,
        )

        self._cached_starter = None
        threading.Thread(target=self._generate_starter_background, daemon=True).start()

        self._clear_stt_buffer()
        self._speak_and_wait(starter)

        if self._check_abort(result):
            return

        user_responded = False
        turns = 0

        while turns < self.SS3_MAX_TURNS:
            if self._check_abort(result):
                break

            utterance = self._wait_user_utterance_abortable(self.SS3_STT_TIMEOUT)
            if not utterance:
                self._log("INFO", "SS3: no response – ending conversation")
                break

            turns += 1
            user_responded = True
            self._log("INFO", f"SS3 turn {turns}: '{utterance}'")

            if self._check_abort(result):
                break

            is_last = turns >= self.SS3_MAX_TURNS

            default = (
                self._im_prompts.get("closing_ack_fallback", "That's nice!")
                if is_last
                else self._im_prompts.get("ss3_mid_turn_fallback", "I see.")
            )
            req_id = self._submit_llm_request(
                kind="closing" if is_last else "followup",
                utterance=utterance,
                user_context=user_context,
                timeout_sec=self.LLM_ASYNC_DEADLINE_SEC,
            )
            reply_text = self._consume_llm_result_or_default(
                req_id=req_id,
                default=default,
                local_wait_sec=self.LLM_ASYNC_LOCAL_WAIT_SEC,
            )

            if reply_text is None:
                break

            if self._check_abort(result):
                break

            self._speak_and_wait(reply_text)

        if user_responded:
            result["success"] = True
            result["talked"] = True
            result["final_state"] = "ss4"
            self._log("INFO", f"SS3: complete ({turns} turns)")
        else:
            result["abort_reason"] = (
                result.get("abort_reason") or "no_response_conversation"
            )
            self._log("WARNING", "SS3: no user response")

    # ==================== Hunger / QR Feeding ====================

    def _run_hunger_feed_tree(
        self, track_id: int, face_id: str, social_state: str, hs: str, result: Dict
    ):
        self._log("INFO", f"Hunger tree: {hs}")
        feed_wait_timeout_sec = float(self._feed_wait_timeout_sec)

        self._clear_stt_buffer()
        self._speak_and_wait(
            self._im_prompts.get(
                "hunger_ask_feed", "I'm so hungry, would you feed me please?"
            )
        )

        meals_eaten = 0
        start_wait_ts = time.time()
        timeouts = 0
        max_timeouts = 2  # 1st: prompt, 2nd: abort

        while not self._check_abort(result):
            fed, payload, new_ts = self._wait_for_feed_since(
                start_wait_ts, feed_wait_timeout_sec
            )

            if fed:
                meals_eaten += 1
                result["last_meal_payload"] = payload
                with self.hunger._lock:
                    lvl = self.hunger.level
                self._log(
                    "INFO",
                    f"Proactive feed: {payload} → meal #{meals_eaten}, stomach {lvl:.1f}",
                )
                self._speak_and_wait(
                    self._im_prompts.get(
                        "hunger_thank_feed", "Yummy, thank you so much."
                    )
                )

                new_hs = self.hunger.get_state()
                if new_hs == "HS1":
                    self._log("INFO", "Hunger satisfied, ending feeding interaction.")
                    break
                else:
                    self._speak_and_wait(
                        self._im_prompts.get(
                            "hunger_still_hungry",
                            "I'm still hungry. Give me more please.",
                        )
                    )
                    start_wait_ts = new_ts
                    timeouts = 0  # Reset timeout counter after successful feed
            else:
                if self._check_abort(result):
                    break

                timeouts += 1
                if timeouts >= max_timeouts:
                    self._log("INFO", "Hunger tree: max timeouts reached, aborting")
                    if not result.get("abort_reason"):
                        result["abort_reason"] = "no_food_qr"
                    break

                # First timeout: prompt user to look for food
                self._speak_and_wait(
                    self._im_prompts.get(
                        "hunger_look_around",
                        "Take a look around, you will find some food for me.",
                    )
                )
                start_wait_ts = time.time()

        result["meals_eaten_count"] = meals_eaten
        if meals_eaten > 0:
            result["success"] = True
        else:
            if not result.get("abort_reason"):
                result["abort_reason"] = "no_food_qr"

        # Keep social state progression consistent:
        # - do NOT promote to ss4 here (feeding is not the short convo)
        # - but if input is ss1/ss2, keep it as-is (caller already passes it)
        result["final_state"] = social_state

    def _wait_for_feed_since(
        self, ts: float, timeout: float
    ) -> Tuple[bool, Optional[str], float]:
        with self._feed_condition:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.abort_event.is_set():
                    return False, None, time.time()

                with self.hunger._lock:
                    lfts = self.hunger.last_feed_ts
                    payload = self.hunger.last_feed_payload

                if lfts > ts:
                    return True, payload, lfts

                wait_time = deadline - time.time()
                if wait_time > 0:
                    self._feed_condition.wait(min(wait_time, 0.5))

            return False, None, time.time()

    @staticmethod
    def _normalize_qr_payload(payload: Optional[str]) -> str:
        if not payload:
            return ""
        return payload.strip().upper()

    def _qr_reader_loop(self):
        while self._running and not self._qr_stop_event.is_set():
            if not self.qr_port:
                time.sleep(0.1)
                continue

            try:
                btl = self.qr_port.read(False)
                if not btl or btl.size() == 0:
                    time.sleep(0.05)
                    continue

                raw_val = btl.get(0).asString()
                val = self._normalize_qr_payload(raw_val)
                if val and val in self._meal_mapping:
                    now = time.time()
                    if now - self._last_scan_ts < self._qr_cooldown_sec:
                        time.sleep(0.02)
                        continue

                    self._last_scan_ts = now
                    self._last_scan_payload = val
                    delta = self._meal_mapping[val]
                    self.hunger.feed(delta, val, now)
                    self._enqueue_responsive_event(
                        "qr_feed",
                        {
                            "payload": val,
                            "delta": delta,
                            "timestamp": now,
                        },
                    )

                    hs_now = self.hunger.get_state()
                    with self.hunger._lock:
                        lvl = self.hunger.level
                    self._log(
                        "INFO", f"QR: {val} (+{delta}) → stomach {lvl:.1f} ({hs_now})"
                    )
                    with self._feed_condition:
                        self._feed_condition.notify_all()
                elif raw_val:
                    self._log("DEBUG", f"Ignoring unknown QR payload: '{raw_val}'")
                time.sleep(0.02)

            except Exception as e:
                self._log("WARNING", f"QR read iteration failed: {e}")
                time.sleep(0.1)

    def _enqueue_responsive_event(self, event_type: str, payload: Dict[str, Any]):
        # Do not queue responsive events while a proactive interaction is running.
        # This enforces "no deferred responsive execution after proactive ends".
        if self._is_run_lock_busy():
            self._log(
                "DEBUG", f"Dropping responsive event '{event_type}' (proactive running)"
            )
            return
        self._queue_put_drop_oldest(
            self._responsive_queue,
            (event_type, payload),
            f"Responsive queue ({event_type})",
            level="DEBUG",
        )

    def _responsive_loop(self):
        self._log("INFO", "Responsive loop started")
        while self._running and not self._responsive_stop_event.is_set():
            try:
                event_type, payload = self._responsive_queue.get_nowait()
            except queue.Empty:
                event_type, payload = None, None

            # Never execute responsive interactions while another interaction is active.
            # If an event arrives during proactive execution, drop it rather than defer it.
            if self._responsive_active.is_set() or self._is_run_lock_busy():
                if event_type is not None:
                    self._log(
                        "DEBUG",
                        f"Dropping responsive event '{event_type}' (interaction busy)",
                    )
                time.sleep(0.05)
                continue

            if event_type == "qr_feed":
                self._run_responsive_qr_ack(
                    payload if isinstance(payload, dict) else {}
                )
                continue

            try:
                bottle = self.stt_port.read(False) if self.stt_port else None
                if not bottle or bottle.size() <= 0:
                    time.sleep(0.05)
                    continue

                utterance = self._extract_stt_text(bottle)
                if not utterance:
                    continue

                if not self._is_responsive_greeting(utterance):
                    self._log(
                        "DEBUG", f"Responsive: STT '{utterance[:40]}' - not a greeting"
                    )
                    continue

                self._log("INFO", f"Responsive: greeting detected '{utterance}'")

                candidate = self._responsive_single_candidate()
                if not candidate:
                    self._log("DEBUG", "Responsive: no candidate found, skipping")
                    continue

                track_id, face_id, is_known = candidate
                cooldown_key = self._responsive_cooldown_key(
                    face_id=face_id,
                    track_id=track_id,
                    is_known=is_known,
                )
                if self._is_greet_in_responsive_cooldown(cooldown_key):
                    self._log(
                        "DEBUG", f"Responsive: '{cooldown_key}' in cooldown, skipping"
                    )
                    continue

                if is_known:
                    self._run_responsive_greeting(track_id, face_id)
                else:
                    self._run_responsive_unknown_intro(track_id, face_id)
            except Exception as e:
                self._log("WARNING", f"Responsive loop iteration failed: {e}")
                time.sleep(0.05)

        self._log("INFO", "Responsive loop stopped")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Lowercase + strip + remove accents for robust name matching.
        'André' == 'andre', 'Marco' == 'marco', 'Ñoño' == 'nono'.
        """
        s = (name or "").strip().lower()
        # Decompose to NFD then drop combining diacritics (category Mn)
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    @staticmethod
    def _normalize_text_for_match(text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def _is_responsive_greeting(self, utterance: str) -> bool:
        normalized = self._normalize_text_for_match(utterance)
        if not normalized:
            return False
        return self.RESPONSIVE_GREET_REGEX.search(normalized) is not None

    @staticmethod
    def _is_responsive_known_name(face_id: str) -> bool:
        if not face_id:
            return False
        lowered = face_id.strip().lower()
        if lowered in ("unknown", "unmatched", "recognizing"):
            return False
        if lowered.isdigit():
            return False
        return True

    @staticmethod
    def _responsive_cooldown_key(face_id: str, track_id: int, is_known: bool) -> str:
        return face_id if is_known else f"unknown:{track_id}"

    def _responsive_single_candidate(self) -> Optional[Tuple[int, str, bool]]:
        """Find the best candidate for responsive greeting.

        Returns the biggest-bbox face as (track_id, face_id, is_known).
        Gaze/attention is NOT required — the utterance itself is sufficient
        signal that the person is addressing the robot.
        """
        with self._faces_lock:
            age = time.time() - self._latest_faces_ts
            cache_size = len(self._latest_faces)

        faces = self.parse_landmarks_latest(staleness_sec=30.0)
        if not faces:
            self._log(
                "DEBUG",
                f"Responsive: no faces in landmarks (cache age={age:.1f}s, cached={cache_size})",
            )
            return None

        candidates: List[Tuple[int, str, float, bool]] = (
            []
        )  # (track_id, face_id, area, is_known)

        for face in faces:
            face_id = str(face.get("face_id", "")).strip()
            track_id = face.get("track_id")
            bbox = face.get("bbox", [0, 0, 0, 0])
            area = (
                bbox[2] * bbox[3]
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4
                else 0
            )

            if not isinstance(track_id, int):
                continue

            is_known = self._is_responsive_known_name(face_id)
            candidates.append((track_id, face_id, area, is_known))

        if not candidates:
            self._log("DEBUG", f"Responsive: no usable faces from {len(faces)} total")
            return None

        # Pick biggest bbox among visible faces
        best = max(candidates, key=lambda c: c[2])
        known_label = "known" if best[3] else "unknown"
        self._log(
            "DEBUG",
            f"Responsive: selected {best[1]} ({known_label}, track={best[0]}, area={best[2]:.0f})",
        )
        return (best[0], best[1], best[3])

    def _is_greet_in_responsive_cooldown(self, name: str) -> bool:
        key = name.lower()
        now = time.time()
        last = self._responsive_greet_cooldown.get(key, 0.0)
        if now - last < self.RESPONSIVE_GREET_COOLDOWN_SEC:
            return True
        self._responsive_greet_cooldown[key] = now
        return False

    def _run_responsive_qr_ack(self, payload: Optional[Dict[str, Any]] = None):
        payload = payload or {}
        if not self.run_lock.acquire(blocking=False):
            self._log("DEBUG", "Responsive QR ack skipped (busy)")
            return
        self._responsive_active.set()
        try:
            meal_payload = payload.get("payload")
            if meal_payload:
                self._log("INFO", f"Responsive QR ack: {meal_payload}")
            self._responsive_speak_and_wait(
                self._im_prompts.get("responsive_qr_ack_text", "yummy, thank you")
            )
            self._enqueue_db_event(
                (
                    "responsive",
                    {
                        "type": "qr_feed",
                        "track_id": None,
                        "name": None,
                        "payload": meal_payload,
                    },
                )
            )
        except Exception as e:
            self._log("WARNING", f"Responsive QR ack failed: {e}")
        finally:
            self.run_lock.release()
            self._responsive_active.clear()

    def _run_responsive_greeting(self, track_id: int, name: str):
        if not self.run_lock.acquire(blocking=False):
            self._log("DEBUG", f"Responsive greeting skipped for '{name}' (busy)")
            return
        self._responsive_active.set()

        # Tell salienceNetwork to look at the person who spoke
        self._set_selector_track_override(track_id)

        # CRITICAL FIX: Start the monitor so the robot doesn't ghost-stare if they leave
        dummy_result = {}
        self._start_monitor(track_id, dummy_result)

        try:
            self._log("INFO", f"Responsive greeting: '{name}' (track={track_id})")
            _greet_tpl = self._im_prompts.get("responsive_greeting", "Hi {name}")
            self._responsive_speak_and_wait(_greet_tpl.format(name=name))

            # --- Wait to see if the person continues talking ---
            self._clear_stt_buffer()
            utterance = self._wait_user_utterance_abortable(self.SS3_STT_TIMEOUT)

            if utterance:
                self._log(
                    "INFO",
                    f"Responsive greeting: follow-up utterance received, entering SS3 conversation",
                )

                # Best-effort Telegram user lookup for personalisation
                user_context = ""
                try:
                    tg_record = self._lookup_telegram_user(name)
                    if tg_record:
                        user_context = self._build_face_user_context(tg_record)
                        if user_context:
                            self._log(
                                "INFO",
                                f"Responsive SS3: personalised context ({len(user_context)} chars)",
                            )
                except Exception as e:
                    self._log("DEBUG", f"Responsive SS3: telegram lookup skipped: {e}")

                turns = 0
                while utterance:
                    turns += 1
                    self._log("INFO", f"Responsive SS3 turn {turns}: '{utterance}'")
                    is_last = turns >= self.SS3_MAX_TURNS

                    default = (
                        self._im_prompts.get("closing_ack_fallback", "That's nice!")
                        if is_last
                        else self._im_prompts.get("ss3_mid_turn_fallback", "I see.")
                    )
                    req_id = self._submit_llm_request(
                        kind="closing" if is_last else "followup",
                        utterance=utterance,
                        user_context=user_context,
                        timeout_sec=self.LLM_ASYNC_DEADLINE_SEC,
                    )
                    reply_text = self._consume_llm_result_or_default(
                        req_id=req_id,
                        default=default,
                        local_wait_sec=self.LLM_ASYNC_LOCAL_WAIT_SEC,
                    )
                    self._responsive_speak_and_wait(reply_text)

                    if is_last:
                        break

                    utterance = self._wait_user_utterance_abortable(
                        self.SS3_STT_TIMEOUT
                    )
                    if not utterance:
                        self._log(
                            "INFO",
                            "Responsive SS3: no further response – ending conversation",
                        )

                self._log("INFO", f"Responsive SS3: complete ({turns} turn(s))")
            else:
                self._log("INFO", "Responsive greeting: no follow-up – stopping")

            self._write_last_greeted(track_id, face_id=name, code=name, person_key=name)
            self._mark_greeted_today(name)
            self._enqueue_db_event(
                (
                    "responsive",
                    {
                        "type": "greeting",
                        "track_id": track_id,
                        "name": name,
                        "payload": None,
                    },
                )
            )
        except Exception as e:
            self._log("WARNING", f"Responsive greeting failed: {e}")
        finally:
            self._stop_monitor()
            self._reset_selector_cooldown(name, track_id)
            self._set_selector_track_override(-1)  # Return to IPS
            self.run_lock.release()
            self._responsive_active.clear()

    def _run_responsive_unknown_intro(self, track_id: int, face_id: str):
        if not self.run_lock.acquire(blocking=False):
            self._log(
                "DEBUG", f"Responsive unknown intro skipped for track={track_id} (busy)"
            )
            return
        self._responsive_active.set()

        # Tell salienceNetwork to look at the person who spoke
        self._set_selector_track_override(track_id)

        # CRITICAL FIX: Start the monitor so the robot doesn't ghost-stare if they leave
        dummy_result = {}
        self._start_monitor(track_id, dummy_result)

        try:
            self._log(
                "INFO",
                f"Responsive unknown greeting: track={track_id} face_id='{face_id}'",
            )

            self._responsive_speak_and_wait(
                self._im_prompts.get("ss1_greeting", "Hi there!")
            )

            self._clear_stt_buffer()
            self._speak(
                self._im_prompts.get(
                    "ss1_ask_name", "We have not met, what's your name?"
                )
            )

            name_utterance = self._wait_user_utterance_abortable(self.SS2_STT_TIMEOUT)
            if not name_utterance:
                self._log(
                    "INFO", "Responsive unknown intro: no response to name question"
                )
                return

            name = self._try_extract_name(name_utterance)
            if not name:
                self._clear_stt_buffer()
                self._speak(
                    self._im_prompts.get(
                        "ss1_ask_name_retry",
                        "Sorry, I didn't catch that. What's your name?",
                    )
                )
                name_utterance2 = self._wait_user_utterance_abortable(
                    self.SS2_STT_TIMEOUT
                )
                if name_utterance2:
                    name = self._try_extract_name(name_utterance2)

            if not name:
                self._log("INFO", "Responsive unknown intro: name extraction failed")
                return

            self._log(
                "INFO",
                f"Responsive unknown intro: extracted name '{name}' for track={track_id}",
            )
            threading.Thread(
                target=self._submit_face_name, args=(track_id, name), daemon=True
            ).start()
            self._write_last_greeted(track_id, face_id=name, code=name, person_key=name)
            self._mark_greeted_today(name)
            self._responsive_speak_and_wait(
                self._im_prompts.get("ss1_nice_to_meet", "Nice to meet you")
            )

            self._enqueue_db_event(
                (
                    "responsive",
                    {
                        "type": "unknown_intro",
                        "track_id": track_id,
                        "name": name,
                        "payload": face_id,
                    },
                )
            )
        except Exception as e:
            self._log("WARNING", f"Responsive unknown intro failed: {e}")
        finally:
            self._stop_monitor()
            self._reset_selector_cooldown(face_id, track_id)
            self._set_selector_track_override(-1)  # Return to IPS
            self.run_lock.release()
            self._responsive_active.clear()

    def _responsive_speak_and_wait(self, text: str) -> bool:
        ok = self._speak(text)
        wc = len(text.split())
        wait = wc / self.TTS_WORDS_PER_SECOND + self.TTS_END_MARGIN
        wait = max(self.TTS_MIN_WAIT, min(self.TTS_MAX_WAIT, wait))
        time.sleep(wait)
        return ok

    def _start_llm_worker(self):
        if self._llm_worker_thread and self._llm_worker_thread.is_alive():
            return
        self._llm_worker_stop.clear()
        self._llm_worker_thread = threading.Thread(
            target=self._llm_worker_loop, daemon=True
        )
        self._llm_worker_thread.start()

    def _stop_llm_worker(self):
        self._llm_worker_stop.set()
        try:
            self._llm_req_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._llm_worker_thread and self._llm_worker_thread.is_alive():
            self._llm_worker_thread.join(timeout=2.0)

    def _next_llm_req_id(self) -> int:
        with self._llm_req_seq_lock:
            self._llm_req_seq += 1
            return self._llm_req_seq

    def _submit_llm_request(
        self,
        kind: str,
        utterance: str,
        user_context: str,
        timeout_sec: float,
    ) -> Optional[int]:
        req_id = self._next_llm_req_id()
        deadline_ts = time.time() + max(0.1, timeout_sec)
        request = {
            "req_id": req_id,
            "kind": kind,
            "utterance": utterance,
            "user_context": user_context,
            "deadline_ts": deadline_ts,
        }
        try:
            self._llm_req_queue.put_nowait(request)
            self._log("DEBUG", f"LLM req {req_id} {kind}")
            return req_id
        except queue.Full:
            self._log("WARNING", f"LLM drop {kind} queue_full")
            return None

    def _poll_llm_result(self, req_id: int) -> Any:
        with self._llm_res_lock:
            return self._llm_results.pop(req_id, None)

    def _cancel_llm_request(self, req_id: Optional[int]):
        if req_id is None:
            return
        with self._llm_res_lock:
            self._llm_cancelled.add(req_id)
            self._llm_results.pop(req_id, None)

    def _consume_llm_result_or_default(
        self,
        req_id: Optional[int],
        default: Any,
        local_wait_sec: float,
    ) -> Any:
        if req_id is None:
            return default

        deadline = time.time() + max(0.0, local_wait_sec)
        while time.time() < deadline:
            if self.abort_event.is_set():
                self._cancel_llm_request(req_id)
                self._log("DEBUG", f"LLM fallback {req_id} abort")
                return default
            text = self._poll_llm_result(req_id)
            if text is not None:
                self._log("DEBUG", f"LLM ok {req_id}")
                return text
            time.sleep(self.LLM_ASYNC_POLL_SEC)

        text = self._poll_llm_result(req_id)
        if text is not None:
            self._log("DEBUG", f"LLM ok {req_id}")
            return text

        self._cancel_llm_request(req_id)
        self._log("DEBUG", f"LLM fallback {req_id} timeout")
        return default

    def _llm_worker_loop(self):
        self._log("INFO", "LLM async worker started")
        while self._running and not self._llm_worker_stop.is_set():
            try:
                request = self._llm_req_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if request is None:
                break

            req_id = request.get("req_id")
            kind = request.get("kind")
            utterance = request.get("utterance", "")
            user_context = request.get("user_context", "")
            deadline_ts = float(request.get("deadline_ts", 0.0))

            if time.time() > deadline_ts:
                self._log("DEBUG", f"LLM drop {req_id} {kind} late")
                continue

            with self._llm_res_lock:
                if req_id in self._llm_cancelled:
                    self._llm_cancelled.remove(req_id)
                    self._log("DEBUG", f"LLM drop {req_id} {kind} cancelled")
                    continue

            try:
                if kind == "starter":
                    text = self._llm_generate_convo_starter(user_context=user_context)
                elif kind == "closing":
                    text = self._llm_generate_closing_acknowledgment(
                        utterance, user_context=user_context
                    )
                elif kind == "extract_name":
                    text = self._llm_extract_name(utterance)
                else:
                    text = self._llm_generate_followup(
                        utterance, [], user_context=user_context
                    )
            except Exception as e:
                self._log("WARNING", f"LLM worker request failed (kind={kind}): {e}")
                continue

            if text is None or time.time() > deadline_ts:
                self._log("DEBUG", f"LLM drop {req_id} {kind} expired")
                continue

            with self._llm_res_lock:
                if req_id in self._llm_cancelled:
                    self._llm_cancelled.remove(req_id)
                    self._log("DEBUG", f"LLM drop {req_id} {kind} cancelled")
                    continue
                self._llm_results[req_id] = text

        self._log("INFO", "LLM async worker stopped")

    # ==================== Abort helpers ====================

    def _check_abort(self, result: Dict) -> bool:
        if self.abort_event.is_set():
            if not result.get("abort_reason"):
                result["abort_reason"] = "target_monitor_abort"
            self._log("WARNING", f"Abort detected: {result['abort_reason']}")
            return True
        return False

    def _wait_user_utterance_abortable(self, timeout: float) -> Optional[str]:
        """Wait for STT with abort checking."""
        start = time.time()
        next_log_time = start + 1.0
        while time.time() - start < timeout:
            if time.time() >= next_log_time:
                self._log("INFO", "waiting for the response")
                next_log_time += 1.0
            loop_start = time.time()
            if self.abort_event.is_set():
                return None
            bottle = self.stt_port.read(False)
            if bottle and bottle.size() > 0:
                self._log("DEBUG", f"STT Raw: {bottle.toString()}")
                text = self._extract_stt_text(bottle)
                if text and text.strip():
                    return text.strip()

            time.sleep(0.1)

            # Compensate for blocking (GIL starvation from other threads)
            scan_duration = time.time() - loop_start
            if scan_duration > 0.5:
                # If we were blocked for more than 0.5s, don't count it towards timeout
                start += scan_duration - 0.1
                self._log(
                    "DEBUG",
                    f"Main loop blocked for {scan_duration:.2f}s, extending timeout",
                )
        return None

    def _try_extract_name(self, utterance: str) -> Optional[str]:
        # Fast regex check for common patterns
        match = re.search(
            r"(?i)(?:my name is|my name's|i am|i'm|im|call me)\s+([a-z][a-z'\-]+)",
            utterance,
        )
        if match:
            return match.group(1).title()

        req_id = self._submit_llm_request(
            kind="extract_name",
            utterance=utterance,
            user_context="",
            timeout_sec=self.LLM_ASYNC_DEADLINE_SEC,
        )
        extraction = self._consume_llm_result_or_default(
            req_id=req_id,
            default={},
            local_wait_sec=self.LLM_ASYNC_NAME_WAIT_SEC,
        )

        if isinstance(extraction, dict) and extraction.get("answered") and extraction.get("name"):
            return extraction["name"]
        return None

    # ==================== YARP Port Helpers ====================

    @staticmethod
    def _is_face_id_resolved(face_id: str) -> bool:
        return face_id.lower() not in ("recognizing", "unmatched")

    def _wait_face_resolve(self, track_id: int, face_id: str, timeout: float) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout:
            faces = self.parse_landmarks_latest()
            for f in faces:
                if f.get("track_id") == track_id:
                    fid = f.get("face_id", "recognizing")
                    if self._is_face_id_resolved(fid):
                        return fid
            time.sleep(0.2)
        return face_id

    def parse_landmarks_latest(self, staleness_sec: float = 2.0) -> List[Dict]:
        with self._faces_lock:
            if time.time() - self._latest_faces_ts > staleness_sec:
                return []
            return list(self._latest_faces)

    def _landmarks_reader_loop(self):
        # _latest_faces_ts is ONLY updated when real face data is stored.
        # Empty bottles are silently ignored; the staleness check in
        # parse_landmarks_latest() naturally expires stale data.
        while self._running and not self._landmarks_reader_stop.is_set():
            if not self.landmarks_port:
                time.sleep(0.01)
                continue
            try:
                bottle = self.landmarks_port.read(False)
                if bottle:
                    landmarks = []
                    for i in range(bottle.size()):
                        face = bottle.get(i).asList()
                        if face:
                            data = self._parse_face_bottle(face)
                            if data:
                                landmarks.append(data)
                    if landmarks:
                        # Only timestamp the cache when we have real face data
                        with self._faces_lock:
                            self._latest_faces = landmarks
                            self._latest_faces_ts = time.time()
                    elif bottle.size() > 0:
                        # Non-empty bottle but all faces failed to parse — log once to aid debugging
                        self._log(
                            "DEBUG",
                            f"Landmarks: bottle size={bottle.size()} but 0 faces parsed",
                        )
                    # Empty bottle → leave cache untouched
                else:
                    time.sleep(0.01)
            except Exception as e:
                self._log("WARNING", f"Landmarks read/parse failed: {e}")
                time.sleep(0.05)

    def _parse_face_bottle(self, bottle: yarp.Bottle) -> Optional[Dict]:
        """Parse a single face sub-bottle from the landmarks stream.

        vision.py encodes most fields as flat key/value pairs:
            ... "face_id" "Neem" "track_id" 86 "attention" "MUTUAL_GAZE" ...
        but bbox and gaze_direction are encoded as nested lists:
            (bbox x y w h)   (gaze_direction fx fy fz)
        where the list's first element is the key name.
        """
        data = {}
        try:
            i = 0
            while i < bottle.size():
                item = bottle.get(i)
                if item.isList():
                    # Nested-list field: first element is the key name
                    lst = item.asList()
                    if lst and lst.size() >= 2:
                        key = lst.get(0).asString()
                        if key in ("bbox", "gaze_direction"):
                            data[key] = [
                                lst.get(j).asFloat64() for j in range(1, lst.size())
                            ]
                    i += 1
                else:
                    # Flat key / value pair
                    if i + 1 >= bottle.size():
                        break
                    key = item.asString()
                    val = bottle.get(i + 1)
                    if key in ("face_id", "distance", "attention", "zone"):
                        data[key] = val.asString()
                    elif key in ("track_id", "is_talking"):
                        data[key] = val.asInt32()
                    elif key in ("time_in_view", "pitch", "yaw", "roll", "cos_angle"):
                        data[key] = val.asFloat64()
                    i += 2
            return data if data else None
        except Exception as e:
            self._log("WARNING", f"Face bottle parse failed: {e}")
            return None

    def _extract_stt_text(self, bottle: yarp.Bottle) -> Optional[str]:
        try:
            if bottle.size() >= 1:
                first = bottle.get(0)
                raw = first.toString()
                if raw.startswith('"'):
                    end_idx = raw.find('"', 1)
                    if end_idx > 1:
                        text = raw[1:end_idx]
                        if text.strip():
                            return text.strip()
                elif ' ""' in raw:
                    text = raw.split(' ""')[0].strip()
                    if text:
                        return text
                else:
                    text = first.asString()
                    if text and text.strip():
                        return text.strip()
        except Exception as e:
            self._log("WARNING", f"STT parse failed: {e}")
        return None

    def _clear_stt_buffer(self):
        cleared = 0
        while self.stt_port.read(False):
            cleared += 1
        if cleared > 0:
            self._log("DEBUG", f"Cleared {cleared} STT messages")

    def ensure_stt_ready(self, language: str = "english") -> bool:
        self._log("INFO", "STT ready")
        return True

    # ==================== Speech Output ====================

    TTS_WORDS_PER_SECOND = 3.0
    TTS_END_MARGIN = 0.5
    TTS_MIN_WAIT = 1.0
    TTS_MAX_WAIT = 8.0

    def _speak(self, text: str) -> bool:
        try:
            if not self.speech_port:
                return False
            # BufferedPortBottle.prepare() + write() is non-blocking:
            # returns immediately after enqueuing under the YARP output thread.
            b = self.speech_port.prepare()
            b.clear()
            b.addString(text)
            self.speech_port.write()
            return True
        except Exception as e:
            self._log("ERROR", f"Speak failed: {e}")
            return False

    def _speak_and_wait(self, text: str) -> bool:
        ok = self._speak(text)
        wc = len(text.split())
        wait = wc / self.TTS_WORDS_PER_SECOND + self.TTS_END_MARGIN
        wait = max(self.TTS_MIN_WAIT, min(self.TTS_MAX_WAIT, wait))

        # Wait with abort checking
        t0 = time.time()
        while time.time() - t0 < wait:
            if self.abort_event.is_set():
                break
            time.sleep(0.1)

        return ok

    # ==================== YARP RPC Commands ====================

    def _get_selector_rpc(self) -> yarp.RpcClient:
        """Lazily create an RPC client connected to /salienceNetwork."""
        if self._selector_rpc is None:
            client = yarp.RpcClient()
            lp = f"/{self.module_name}/salienceNetwork/rpc"
            if not client.open(lp):
                raise RuntimeError(f"Failed to open {lp}")
            if not client.addOutput("/salienceNetwork"):
                client.close()
                raise RuntimeError("Failed to connect to /salienceNetwork")
            self._selector_rpc = client
        return self._selector_rpc

    def _set_selector_track_override(self, track_id: int):
        """Send set_track_id to salienceNetwork to override (or release) gaze."""
        try:
            rpc = self._get_selector_rpc()
            cmd = yarp.Bottle()
            cmd.addString("set_track_id")
            cmd.addInt32(track_id)
            reply = yarp.Bottle()
            rpc.write(cmd, reply)
            self._log("INFO", f"Selector override → track_id={track_id}")
        except Exception as e:
            self._log("WARNING", f"set_selector_track_override failed: {e}")
            if self._selector_rpc:
                self._selector_rpc.close()
                self._selector_rpc = None

    def _get_vision_rpc(self) -> yarp.RpcClient:
        if self._vision_rpc is None:
            client = yarp.RpcClient()
            lp = f"/{self.module_name}/vision/rpc"
            if not client.open(lp):
                raise RuntimeError(f"Failed to open {lp}")
            if not client.addOutput("/alwayson/vision/rpc"):
                client.close()
                raise RuntimeError("Failed to connect to /alwayson/vision/rpc")
            self._vision_rpc = client
        return self._vision_rpc

    def _submit_face_name(self, track_id: int, name: str) -> bool:
        try:
            rpc = self._get_vision_rpc()
            cmd = yarp.Bottle()
            cmd.addString("name")
            cmd.addString(name)
            cmd.addString("id")
            cmd.addInt32(track_id)
            reply = yarp.Bottle()
            rpc.write(cmd, reply)
            resp = reply.toString()
            self._log("INFO", f"Submitted name '{name}' for track {track_id}: {resp}")
            return "ok" in resp.lower()
        except Exception as e:
            self._log("ERROR", f"Name submission failed: {e}")
            if self._vision_rpc:
                self._vision_rpc.close()
                self._vision_rpc = None
            return False

    # ==================== Database ====================

    def _enqueue_db_event(self, item: Tuple[str, Dict[str, Any]]):
        self._queue_put_drop_oldest(self._db_queue, item, "DB queue")

    def _queue_put_drop_oldest(
        self,
        q: queue.Queue,
        item: Any,
        label: str,
        level: str = "WARNING",
    ):
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            self._log(level, f"{label} full, dropping oldest")
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            self._log(level, f"{label} still full, dropping new item")

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self.DB_FILE), exist_ok=True)
            conn = sqlite3.connect(self.DB_FILE)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT,
                timestamp TEXT, track_id INTEGER, face_id TEXT,
                initial_state TEXT, final_state TEXT,
                success INTEGER, abort_reason TEXT,
                greeted INTEGER, talked INTEGER,
                extracted_name TEXT,
                target_stayed_biggest INTEGER,
                transcript TEXT,
                full_result TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS responsive_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                track_id INTEGER,
                name TEXT,
                payload TEXT
            )""")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_time ON interactions(timestamp)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_track ON interactions(track_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_interaction_id ON interactions(interaction_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_responsive_time ON responsive_interactions(timestamp)"
            )
            conn.commit()
            conn.close()
            self._log("INFO", f"DB ready: {self.DB_FILE}")
        except Exception as e:
            self._log("ERROR", f"DB init failed: {e}")

    def _open_db_connection(self, timeout: float) -> sqlite3.Connection:
        conn = sqlite3.connect(self.DB_FILE, timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def _db_worker(self):
        conn = None
        while True:
            try:
                item = self._db_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue
            if item is None:
                break
            table, data = item
            try:
                if conn is None:
                    conn = self._open_db_connection(timeout=10.0)
                if table == "interaction":
                    self._save_interaction_to_db(conn, data)
                elif table == "responsive":
                    self._save_responsive_interaction_to_db(conn, data)
            except Exception as e:
                self._log("ERROR", f"DB write failed: {e}")
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _save_interaction_to_db(self, conn: sqlite3.Connection, data: Dict):
        try:
            r = data["result"]
            c = conn.cursor()
            # Build transcript from logs
            transcript_lines = [
                log["message"]
                for log in r.get("logs", [])
                if "User:" in log.get("message", "")
                or "Robot:" in log.get("message", "")
                or "Asking" in log.get("message", "")
                or "Response" in log.get("message", "")
            ]
            c.execute(
                """INSERT INTO interactions
                (interaction_id,timestamp,track_id,face_id,initial_state,final_state,success,abort_reason,
                 greeted,talked,extracted_name,target_stayed_biggest,transcript,full_result)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data.get("interaction_id"),
                    datetime.now().isoformat(),
                    data["track_id"],
                    data["face_id"],
                    data["initial_state"],
                    r.get("final_state", ""),
                    int(r.get("success", False)),
                    r.get("abort_reason"),
                    int(r.get("greeted", False)),
                    int(r.get("talked", False)),
                    r.get("extracted_name"),
                    int(r.get("target_stayed_biggest", True)),
                    json.dumps(transcript_lines, ensure_ascii=False),
                    json.dumps(r, ensure_ascii=False),
                ),
            )
            conn.commit()
        except Exception as e:
            self._log("ERROR", f"DB save failed: {e}")

    def _save_responsive_interaction_to_db(
        self, conn: sqlite3.Connection, data: Dict[str, Any]
    ):
        try:
            c = conn.cursor()
            c.execute(
                """INSERT INTO responsive_interactions
                (interaction_id,timestamp,type,track_id,name,payload)
                VALUES (?,?,?,?,?,?)""",
                (
                    data.get("interaction_id"),
                    datetime.now().astimezone().isoformat(),
                    data.get("type"),
                    data.get("track_id"),
                    data.get("name"),
                    data.get("payload"),
                ),
            )
            conn.commit()
        except Exception as e:
            self._log("ERROR", f"Responsive DB save failed: {e}")

    # ==================== JSON Persistence ====================

    def _ensure_json_file(self, filename: str, default: Any):
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump(default, f)

    def _load_json(self, filename: str, default: Any) -> Any:
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, filename: str, data: Any):
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def _save_json_atomic(self, filename: str, data: Any):
        directory = os.path.dirname(filename) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, filename)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _write_last_greeted(
        self, track_id: int, face_id: str, code: str, person_key: Optional[str] = None
    ):
        try:
            path = self.LAST_GREETED_FILE
            raw_entries = self._load_json(path, {})
            entries: Dict[str, Dict[str, Any]] = {}

            if isinstance(raw_entries, dict):
                entries = {
                    str(k): v for k, v in raw_entries.items() if isinstance(v, dict)
                }
            elif isinstance(raw_entries, list):
                for entry in raw_entries:
                    if not isinstance(entry, dict):
                        continue
                    legacy_key = entry.get("assigned_code_or_name") or entry.get(
                        "face_id"
                    )
                    if legacy_key:
                        entries[str(legacy_key)] = entry

            key = (person_key or "").strip()
            if not key:
                if face_id and face_id.lower() not in (
                    "unknown",
                    "unmatched",
                    "recognizing",
                ):
                    key = face_id
                else:
                    key = f"unknown:{track_id}"

            entries[key] = {
                "timestamp": datetime.now().isoformat(),
                "track_id": track_id,
                "face_id": face_id,
                "assigned_code_or_name": code,
            }
            self._save_json_atomic(path, entries)
        except Exception as e:
            self._log("ERROR", f"Write last_greeted failed: {e}")

    def _mark_greeted_today(self, name: str) -> None:
        try:
            key = (name or "").strip()
            if not key:
                return
            path = self.GREETED_TODAY_FILE
            # Cross-process exclusive lock so this read-modify-write is atomic
            # relative to salienceNetwork's concurrent snapshot writes.
            lock_path = path + ".lock"
            with open(lock_path, "w") as _lf:
                fcntl.flock(_lf, fcntl.LOCK_EX)
                try:
                    raw = self._load_json(path, {})
                    entries = raw if isinstance(raw, dict) else {}
                    entries[key] = datetime.now().astimezone().isoformat()
                    self._save_json_atomic(path, entries)
                finally:
                    fcntl.flock(_lf, fcntl.LOCK_UN)
        except Exception as e:
            self._log("WARNING", f"Write greeted_today failed: {e}")

    # ==================== LLM Integration ====================

    def setup_llm(
        self, deployment_name: str, max_completion_tokens: int
    ) -> AzureOpenAI:
        """Instantiate an AzureOpenAI client for a given deployment.

        Stores deployment_name and max_completion_tokens on self for use in
        _llm_request().  Raises RuntimeError if required env vars are missing.
        Note: GPT-5 models only support temperature=1 (the default) — it is not passed.
        """
        # --- Read and validate env ---
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().strip('"').strip("'")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set or empty")

        # Normalize: strip full chat-completions path down to base resource URL
        if "/openai/" in endpoint:
            endpoint = endpoint.split("/openai/")[0]
        endpoint = endpoint.rstrip("/")

        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set or empty")

        api_version = os.getenv("OPENAI_API_VERSION", "").strip().strip('"').strip(
            "'"
        ) or os.getenv("AZURE_OPENAI_API_VERSION", "").strip().strip('"').strip("'")
        if not api_version:
            raise RuntimeError(
                "OPENAI_API_VERSION (or AZURE_OPENAI_API_VERSION) is not set or empty"
            )

        if not deployment_name:
            raise RuntimeError("deployment_name must not be empty")

        self._llm_deployment = deployment_name
        self._llm_max_tokens = max_completion_tokens

        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=self.LLM_TIMEOUT,
        )

    def setup_azure_llms(self) -> None:
        """Create Azure OpenAI client – all tasks use gpt5-nano."""
        dep_nano = os.getenv(
            "AZURE_DEPLOYMENT_GPT5_NANO", "contact-Yogaexperiment_gpt5nano"
        )

        # Both clients share the same AzureOpenAI instance; routing preserved for
        # forward-compatibility (llm_extract for JSON tasks, llm_chat for conversation).
        client: AzureOpenAI = self.setup_llm(dep_nano, max_completion_tokens=2000)
        self.llm_extract = client
        self.llm_chat = client

        self._log("INFO", f"Azure LLMs ready – all tasks using nano={dep_nano}")

    def _llm_request(
        self,
        prompt: str,
        json_format: bool = False,
        system: Optional[str] = None,
        options: Optional[dict] = None,
        format_obj: Optional[object] = None,
    ) -> str:
        """Send a prompt to the Azure gpt5-nano deployment via the direct OpenAI SDK.

        Both llm_extract and llm_chat point to the same AzureOpenAI client.
        Routing logic is preserved for forward-compatibility:
          - json_format / format_obj → llm_extract (nano)
          - conversational responses  → llm_chat   (nano)
        """
        # --- Route ---
        is_json_task = json_format or (format_obj is not None)
        client = self.llm_extract if is_json_task else self.llm_chat

        if client is None:
            self._log(
                "ERROR", "LLM client not initialised – call setup_azure_llms() first"
            )
            return ""

        # --- Per-call max_completion_tokens override ---
        max_tokens = self._llm_max_tokens
        if options and "num_predict" in options:
            max_tokens = options["num_predict"]

        # --- Build messages ---
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.llm_retry_attempts):
            try:
                resp = client.chat.completions.create(
                    model=self._llm_deployment,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                last_error = "Empty response"
            except Exception as e:
                last_error = str(e)
                self._log(
                    "WARNING",
                    f"LLM attempt {attempt + 1}/{self.llm_retry_attempts} failed: {e}",
                )
                if attempt < self.llm_retry_attempts - 1:
                    time.sleep(self.llm_retry_delay)
        self._log(
            "ERROR",
            f"LLM failed after {self.llm_retry_attempts} attempts: {last_error}",
        )
        return ""

    def _llm_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[dict] = None,
        format_obj: Optional[object] = None,
    ) -> Dict:
        text = self._llm_request(
            prompt,
            json_format=True,
            system=system,
            options=options,
            format_obj=format_obj,
        )
        if not text:
            return {}
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                r = json.loads(text[start : end + 1])
                if isinstance(r, dict):
                    return r
            except json.JSONDecodeError:
                pass
        try:
            r = json.loads(text)
            return r if isinstance(r, dict) else {}
        except json.JSONDecodeError:
            return {}

    GREETING_KEYWORDS = {
        "hello",
        "hi",
        "hey",
        "ciao",
        "hola",
        "salut",
        "hallo",
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "okay",
        "ok",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "greetings",
        "sup",
        "what's up",
    }

    def _llm_detect_greeting_response(self, utterance: str) -> Dict:
        lower = utterance.lower()
        for kw in self.GREETING_KEYWORDS:
            if kw in lower:
                return {"responded": True, "confidence": 1.0}
        if lower.strip():
            return {"responded": True, "confidence": 0.5}
        return {"responded": False, "confidence": 0.0}

    def _llm_extract_name(self, utterance: str) -> Dict:
        _tmpl = self._im_prompts.get(
            "extract_name_prompt",
            'Utterance: "{utterance}"\nExtract the speaker\'s own name.\n'
            '- Look for patterns: "my name is", "I\'m", "call me", "I am".\n'
            "- Name must be Title Case, single token preferred (hyphens/apostrophes OK).\n"
            "- If multiple names, pick the first self-referential one.\n"
            "- Never invent a name from greetings or filler.\n"
            "- If no name is stated: answered=false, name=null, confidence=0.0.",
        )
        prompt = _tmpl.format(utterance=utterance)
        # format_obj is passed for routing only; JSON is parsed by _llm_json
        schema = {
            "type": "object",
            "properties": {
                "answered": {"type": "boolean"},
                "name": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["answered", "name", "confidence"],
            "additionalProperties": False,
        }
        r = self._llm_json(
            prompt,
            system=self.LLM_SYSTEM_JSON,
            options={"num_predict": 2000},
            format_obj=schema,
        )
        conf = float(r.get("confidence", 0) or 0)
        conf = max(0.0, min(1.0, conf))  # clamp: model may exceed [0, 1]
        return {
            "answered": r.get("answered", False) is True,
            "name": r.get("name") or None,
            "confidence": conf,
        }

    def _llm_generate_convo_starter(self, user_context: str = "") -> str:
        if user_context:
            prompt = self._im_prompts.get(
                "convo_starter_personalized_prompt",
                "You know this about the person you're talking to:\n{user_context}\n\n"
                "Ask ONE short, friendly question that shows you remember them. "
                "6 to 14 words. Reference something you know about them if relevant. "
                "No greeting, no sensitive topics. Output only the sentence.",
            ).format(user_context=user_context)
        else:
            prompt = self._im_prompts.get(
                "convo_starter_prompt",
                "Ask ONE short, friendly question about the person's day or wellbeing. "
                "6 to 12 words. No greeting, no name, no sensitive topics. "
                "Output only the sentence.",
            )
        text = self._llm_request(
            prompt, system=self.LLM_SYSTEM_DEFAULT, options={"num_predict": 2000}
        )
        return (
            text.strip("\"'").strip()
            if text and len(text) < 150
            else self._im_prompts.get(
                "convo_starter_fallback", "How are you doing these days?"
            )
        )

    def _generate_starter_background(self, user_context: str = ""):
        try:
            starter = self._llm_generate_convo_starter(user_context=user_context)
            if starter:
                self._cached_starter = starter
        except Exception as e:
            self._log("WARNING", f"Starter prefetch failed: {e}")

    def _llm_generate_followup(
        self, last_utterance: str, history: List[str], user_context: str = ""
    ) -> str:
        if user_context:
            _tmpl = self._im_prompts.get(
                "followup_personalized_prompt",
                "You know this about the person:\n{user_context}\n\n"
                "User said: '{last_utterance}'\nRespond in 1 sentence (2 max). 22 words or fewer. "
                "Reflect the user's sentiment. Use what you know about them to make the response personal. "
                "You may ask at most one short follow-up question. Output only the spoken text.",
            )
        else:
            _tmpl = self._im_prompts.get(
                "followup_prompt",
                "User said: '{last_utterance}'\nRespond in 1 sentence (2 max). 22 words or fewer. "
                "Reflect the user's sentiment. You may ask at most one short follow-up question. "
                "Output only the spoken text.",
            )
        prompt = _tmpl.format(last_utterance=last_utterance, user_context=user_context)
        text = self._llm_request(
            prompt, system=self.LLM_SYSTEM_DEFAULT, options={"num_predict": 2000}
        )
        return (
            text.strip("\"'").strip()
            if text and len(text) < 200
            else self._im_prompts.get("followup_fallback", "That's interesting!")
        )

    def _llm_generate_closing_acknowledgment(
        self, last_utterance: str, user_context: str = ""
    ) -> str:
        if user_context:
            _tmpl = self._im_prompts.get(
                "closing_ack_personalized_prompt",
                "You know this about the person:\n{user_context}\n\n"
                "Person said: '{last_utterance}'\nWarm, personal acknowledgment. 4 to 10 words. "
                "No question mark. Output only the spoken text.",
            )
        else:
            _tmpl = self._im_prompts.get(
                "closing_ack_prompt",
                "Person said: '{last_utterance}'\nWarm acknowledgment. 4 to 8 words. "
                "No question mark. Output only the spoken text.",
            )
        prompt = _tmpl.format(last_utterance=last_utterance, user_context=user_context)
        text = self._llm_request(
            prompt, system=self.LLM_SYSTEM_DEFAULT, options={"num_predict": 2000}
        )
        return (
            text.strip("\"'").strip()
            if text and len(text) < 100
            else self._im_prompts.get("closing_ack_fallback", "That's nice!")
        )

    # ==================== Helpers ====================

    def _log(self, level: str, message: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{level}] {message}")
        self.log_buffer.append({"timestamp": ts, "level": level, "message": message})


# ==================== Main ====================

if __name__ == "__main__":
    import sys

    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("[ERROR] YARP network not available – start yarpserver first.")
        sys.exit(1)

    module = ExecutiveControlModule()
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.configure(sys.argv)

    print("=" * 60)
    print(" ExecutiveControlModule")
    print(" ss1=unknown  ss2=known/not-greeted  ss3=greeted  ss4=no-op")
    print()
    print(" yarp connect /alwayson/vision/landmarks:o /alwayson/executiveControl/landmarks:i")
    print(" yarp connect /speech2text/text:o          /alwayson/executiveControl/stt:i")
    print(" yarp connect /icub/cam/left               /alwayson/vision/img:i")
    print(" yarp connect /alwayson/executiveControl/speech:o /acapelaSpeak/speech:i")
    print()
    print(
        " RPC: echo 'run <track_id> <face_id> <ss1|ss2|ss3|ss4>' | yarp rpc /executiveControl"
    )
    print("=" * 60)

    try:
        module.runModule(rf)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        module.interruptModule()
        module.close()
        yarp.Network.fini()
