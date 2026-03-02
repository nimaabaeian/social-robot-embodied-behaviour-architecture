"""
interactionManager.py – YARP RFModule for Social Interaction State Trees

Uses Ollama LLM (Llama 3.2 3B Instruct) for natural language understanding
and generation.

Social States:
  ss1: unknown                 → greet, ask name, register
  ss2: known, not greeted      → greet by name → ss3
  ss3: known, greeted          → short conversation (up to 3 turns)
  ss4: known, talked           → no-op (ultimate state)

Target Monitor: continuously checks that the interaction target is still the
biggest-bbox face; sets abort_event if target is lost or displaced.

YARP Connections:
    yarp connect /alwayson/vision/landmarks:o /interactionManager/landmarks:i
    yarp connect /speech2text/text:o          /interactionManager/stt:i
    yarp connect /icub/cam/left               /interactionManager/camLeft:i
    yarp connect /interactionManager/speech:o /acapelaSpeak/speech:i

RPC:
    echo "run <track_id> <face_id> <ss1|ss2|ss3|ss4>" | yarp rpc /interactionManager
"""

import concurrent.futures
import ctypes
import json
import re
import os
import queue
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yarp

try:
    import cv2
    import numpy as np
except ImportError:
    pass


class HungerModel:
    def __init__(self, drain_hours: float = 6.0, hungry_threshold: float = 60.0, starving_threshold: float = 25.0):
        self.level: float = 100.0
        self.drain_hours = drain_hours
        self.hungry_threshold = hungry_threshold
        self.starving_threshold = starving_threshold
        self.last_update_ts: float = time.time()
        self.last_feed_ts: float = 0.0
        self.last_feed_payload: Optional[str] = None
        self._lock = threading.Lock()

    def update(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        with self._lock:
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

    def get_state(self) -> str:
        self.update()
        with self._lock:
            if self.level >= self.hungry_threshold:
                return "HS1"
            elif self.level >= self.starving_threshold:
                return "HS2"
            else:
                return "HS3"

class InteractionManagerModule(yarp.RFModule):

    # ==================== Constants ====================

    OLLAMA_URL = "http://localhost:11434"
    LLM_MODEL = "llama3.2:3b"
    LLM_SYSTEM_DEFAULT = "You are a friendly social robot assistant. Be concise, natural, and polite. Follow the user/task instructions exactly."
    LLM_SYSTEM_JSON = "You are a strict JSON generator for information extraction. Output ONLY a single JSON object. No extra text, no markdown, no code fences."
    DB_FILE = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/data_collection/interaction_manager.db"
    LAST_GREETED_FILE = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/last_greeted.json"
    GREETED_TODAY_FILE = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/greeted_today.json"

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
    TARGET_LOST_TIMEOUT = 8.0  # seconds before declaring target lost

    # Responsive interactions
    RESPONSIVE_GREET_REGEX = re.compile(r"\b(hello|hi|hey|ciao|buongiorno|good\s+morning)\b")
    RESPONSIVE_GREET_COOLDOWN_SEC = 10.0
    RESPONSIVE_ALLOWED_ATTENTION = {"MUTUAL_GAZE", "NEAR_GAZE"}

    # ==================== Lifecycle ====================

    def __init__(self):
        super().__init__()
        self.module_name = "interactionManager"
        self.period = 1.0
        self._running = True
        self.run_lock = threading.Lock()
        self.log_buffer: List[Dict] = []

        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # YARP ports (no context port)
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.stt_port: Optional[yarp.BufferedPortBottle] = None
        self.speech_port: Optional[yarp.Port] = None

        # RPC clients (lazy)
        self._interaction_rpc: Optional[yarp.RpcClient] = None
        self._vision_rpc: Optional[yarp.RpcClient] = None

        # YARP ports — speech_port is BufferedPortBottle so write() is non-blocking
        self.speech_port: Optional[yarp.BufferedPortBottle] = None

        self._cached_starter: Optional[str] = None

        # Landmarks reader
        self._faces_lock = threading.Lock()
        self._latest_faces: List[Dict] = []
        self._latest_faces_ts: float = 0.0
        self._landmarks_reader_stop = threading.Event()
        self._landmarks_reader_thread: Optional[threading.Thread] = None

        # LLM Global Pool
        self._llm_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Abort mechanism
        self.abort_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._current_track_id: Optional[int] = None

        # Error recovery
        self.llm_retry_attempts = 3
        self.llm_retry_delay = 1.0
        self.ollama_last_check = 0.0
        self.ollama_check_interval = 60.0

        # DB queue
        self._db_queue: queue.Queue = queue.Queue()
        self._db_thread: Optional[threading.Thread] = None

        # Hunger and QR
        self.hunger = HungerModel()
        self.cam_left_port: Optional[yarp.BufferedPortImageRgb] = None
        self._qr_thread: Optional[threading.Thread] = None
        self._qr_stop_event = threading.Event()
        self._qr_cooldown_sec = 3.0
        self._last_scan_ts = 0.0
        self._last_scan_payload: Optional[str] = None
        self._feed_condition = threading.Condition()
        self._feed_wait_timeout_sec = 8.0
        self._feed_timeout_behaviour = "right_there"
        self._meal_mapping = {
            "SMALL_MEAL": 10.0,
            "MEDIUM_MEAL": 25.0,
            "LARGE_MEAL": 45.0
        }

        # Responsive interaction path
        self._responsive_stop_event = threading.Event()
        self._responsive_thread: Optional[threading.Thread] = None
        self._responsive_queue: queue.Queue = queue.Queue(maxsize=32)
        self._responsive_active = threading.Event()
        self._responsive_greet_cooldown: Dict[str, float] = {}

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            self.setName(self.module_name)

            drain_hours  = rf.find("drain_hours").asFloat64()        if rf.check("drain_hours")        else 6.0
            hungry_th    = rf.find("hungry_threshold").asFloat64()   if rf.check("hungry_threshold")   else 60.0
            starving_th  = rf.find("starving_threshold").asFloat64() if rf.check("starving_threshold") else 25.0
            self.hunger  = HungerModel(drain_hours=drain_hours,
                                       hungry_threshold=hungry_th,
                                       starving_threshold=starving_th)

            if rf.check("qr_cooldown_sec"):       self._qr_cooldown_sec       = rf.find("qr_cooldown_sec").asFloat64()
            if rf.check("feed_wait_timeout_sec"): self._feed_wait_timeout_sec = rf.find("feed_wait_timeout_sec").asFloat64()
            if rf.check("feed_timeout_behaviour"): self._feed_timeout_behaviour = rf.find("feed_timeout_behaviour").asString()

            self.handle_port.open("/" + self.module_name)

            self.landmarks_port = yarp.BufferedPortBottle()
            self.stt_port       = yarp.BufferedPortBottle()
            self.speech_port    = yarp.BufferedPortBottle()  # non-blocking write
            self.cam_left_port  = yarp.BufferedPortImageRgb()

            ports = [
                (self.landmarks_port, "landmarks:i"),
                (self.stt_port,       "stt:i"),
                (self.speech_port,    "speech:o"),
                (self.cam_left_port,  "camLeft:i"),
            ]
            for port, suffix in ports:
                pn = f"/{self.module_name}/{suffix}"
                if not port.open(pn):
                    self._log("ERROR", f"Failed to open port: {pn}")
                    return False
                self._log("INFO", f"Port open: {pn}")

            self._ensure_json_file(self.LAST_GREETED_FILE, {})
            self._ensure_json_file(self.GREETED_TODAY_FILE, {})
            self._init_db()

            self._landmarks_reader_stop.clear()
            self._landmarks_reader_thread = threading.Thread(
                target=self._landmarks_reader_loop, daemon=True)
            self._landmarks_reader_thread.start()

            self._db_thread = threading.Thread(target=self._db_worker, daemon=True)
            self._db_thread.start()

            self._log("INFO", f"Camera: connect /icub/cam/left → /{self.module_name}/camLeft:i manually")

            self._qr_stop_event.clear()
            self._qr_thread = threading.Thread(target=self._qr_reader_loop, daemon=True)
            self._qr_thread.start()

            self._responsive_stop_event.clear()
            self._responsive_thread = threading.Thread(target=self._responsive_loop, daemon=True)
            self._responsive_thread.start()

            self.ensure_ollama_and_model()
            threading.Thread(target=self._generate_starter_background, daemon=True).start()
            threading.Thread(target=self._prewarm_rpc_connections, daemon=True).start()

            self._log("INFO", "InteractionManagerModule ready")
            return True
        except Exception as e:
            self._log("ERROR", f"configure() failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def interruptModule(self) -> bool:
        self._log("INFO", "Interrupting...")
        self._running = False
        self.abort_event.set()
        self._landmarks_reader_stop.set()
        self._qr_stop_event.set()
        self._responsive_stop_event.set()
        self.handle_port.interrupt()
        for port in [self.landmarks_port, self.stt_port, self.speech_port, self.cam_left_port]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        self._log("INFO", "Closing...")
        self._landmarks_reader_stop.set()
        self._qr_stop_event.set()
        self._responsive_stop_event.set()
        if self._landmarks_reader_thread:
            self._landmarks_reader_thread.join(timeout=2.0)
        if self._qr_thread:
            self._qr_thread.join(timeout=2.0)
        if self._responsive_thread:
            self._responsive_thread.join(timeout=2.0)
        self._db_queue.put(None)
        if self._db_thread:
            self._db_thread.join(timeout=3.0)
        self.handle_port.close()
        for port in [self.landmarks_port, self.stt_port, self.speech_port, self.cam_left_port]:
            if port:
                port.close()
        if self._interaction_rpc:
            self._interaction_rpc.close()
        if self._vision_rpc:
            self._vision_rpc.close()
        self._llm_pool.shutdown(wait=False, cancel_futures=True)
        return True

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        self.hunger.update()
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
                return self._reply_ok(reply, {
                    "success": True, "status": "ready",
                    "module": self.module_name, "busy": busy,
                })

            if command == "help":
                return self._reply_ok(reply, {
                    "success": True,
                    "commands": [
                        "run <track_id> <face_id> <ss1|ss2|ss3|ss4>",
                        "status", "help", "quit",
                    ],
                })

            if command == "quit":
                self._running = False
                self.stopModule()
                return self._reply_ok(reply, {"success": True, "message": "Shutting down"})

            if command != "run":
                return self._reply_error(reply, f"Unknown command: {command}")

            if self._responsive_active.is_set():
                return self._reply_ok(reply, {
                    "success": False,
                    "error": "responsive_interaction_running",
                })

            if cmd.size() < 4:
                return self._reply_error(reply, "Usage: run <track_id> <face_id> <ss1|ss2|ss3|ss4>")

            track_id = cmd.get(1).asInt32()
            face_id = cmd.get(2).asString()
            social_state = cmd.get(3).asString().lower()

            if social_state not in self.VALID_STATES:
                return self._reply_error(reply, f"Invalid state: {social_state}")

            if not self.run_lock.acquire(blocking=False):
                return self._reply_error(reply, "Another action is running")

            try:
                self.log_buffer = []
                self._log("INFO", f"--- Interaction start: track={track_id} face={face_id} state={social_state} ---")
                self.ensure_stt_ready("english")

                result = self._execute_interaction(track_id, face_id, social_state)

                # DB save async
                self._db_queue.put(("interaction", {
                    "track_id": track_id,
                    "face_id": face_id,
                    "initial_state": social_state,
                    "result": dict(result),
                }))

                ar = result.get("abort_reason")
                compact_abort_reason = None
                if ar:
                    if ar in ("target_lost", "target_not_biggest", "target_monitor_abort"):
                        compact_abort_reason = "face_disappeared"
                    else:
                        compact_abort_reason = "not_responded"

                compact_success = bool(result.get("success", False)) and compact_abort_reason is None

                compact = {
                    "success": compact_success,
                    "track_id": track_id,
                    "name": None,
                    "name_extracted": False,
                    "abort_reason": compact_abort_reason,
                    "initial_state": social_state,
                    "final_state": result.get("final_state", social_state)
                }
                
                if result.get("extracted_name"):
                    compact["name"] = result.get("extracted_name")
                    compact["name_extracted"] = True
                elif social_state in ("ss2", "ss3", "ss4"):
                    compact["name"] = face_id
                    
                for extra in ["interaction_tag", "hunger_state_start", "hunger_state_end",
                              "stomach_level_start", "stomach_level_end", "meals_eaten_count", "last_meal_payload"]:
                    if extra in result:
                        compact[extra] = result[extra]

                reply.addString("ok")
                reply.addString(json.dumps(compact, ensure_ascii=False))
            finally:
                self.run_lock.release()

            return True
        except Exception as e:
            self._log("ERROR", f"Exception in respond: {e}")
            import traceback; traceback.print_exc()
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
        reply.addString(json.dumps({"success": False, "error": error, "logs": self.log_buffer},
                                   ensure_ascii=False))
        return True

    def _is_run_lock_busy(self) -> bool:
        is_locked = not self.run_lock.acquire(blocking=False)
        if not is_locked:
            self.run_lock.release()
        return is_locked

    # ==================== Interaction Execution ====================

    def _execute_interaction(self, track_id: int, face_id: str, social_state: str) -> Dict[str, Any]:
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
        self.abort_event.clear()
        self._current_track_id = track_id
        self._monitor_thread = threading.Thread(
            target=self._target_monitor_loop,
            args=(track_id, result),
            daemon=True,
        )
        self._monitor_thread.start()

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
        self.abort_event.set()
        self._current_track_id = None
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        result["logs"] = self.log_buffer.copy()
        return result

    # ==================== Target Monitor ====================

    def _target_monitor_loop(self, track_id: int, result: Dict):
        """Abort if target is no longer biggest bbox or disappears."""
        last_seen     = time.time()
        last_iter_ts  = time.time()

        while not self.abort_event.is_set():
            now = time.time()
            if now - last_iter_ts > 1.5:
                # Thread was starved for >1.5 s – reset watchdog silently
                last_seen = now
            last_iter_ts = now

            try:
                faces = self.parse_landmarks_latest()
                found = False
                biggest_tid = None

                if faces:
                    biggest     = max(faces, key=lambda f: self._face_area(f))
                    biggest_tid = biggest.get("track_id")
                    for f in faces:
                        if f.get("track_id") == track_id:
                            found = True
                            break

                if found:
                    last_seen = time.time()
                    if biggest_tid is not None and biggest_tid != track_id:
                        self._log("WARNING", f"Monitor: target {track_id} displaced by {biggest_tid}")
                        result["target_stayed_biggest"] = False
                        result["abort_reason"] = "target_not_biggest"
                        self.abort_event.set()
                        return
                else:
                    elapsed = time.time() - last_seen
                    if elapsed > self.TARGET_LOST_TIMEOUT:
                        self._log("WARNING", f"Monitor: target {track_id} lost for {elapsed:.1f}s")
                        result["target_stayed_biggest"] = False
                        result["abort_reason"] = "target_lost"
                        self.abort_event.set()
                        return
            except Exception as e:
                self._log("WARNING", f"Monitor error: {e}")

            time.sleep(1.0 / self.MONITOR_HZ)

    @staticmethod
    def _face_area(face: Dict) -> float:
        bbox = face.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return bbox[2] * bbox[3]
        return 0.0

    # ==================== Tree SS1: Unknown ====================

    def _run_ss1_tree(self, track_id: int, face_id: str, result: Dict):
        """
        1) Execute ao_hi
        2) Wait response  → if none → abort
        3) Ask name       → if none → abort
        4) Extract name   → retry once → if fail → abort
        5) Say 'Nice to meet you', register name, update last_greeted
        """
        self._log("INFO", "SS1: start")

        # 1) ao_hi
        self._clear_stt_buffer()
        threading.Thread(target=self._execute_behaviour, args=("ao_hi",), daemon=True).start()
        result["greeted"] = True

        # 2) Wait for response
        if self._check_abort(result):
            return
        utterance = self._wait_user_utterance_abortable(self.SS1_STT_TIMEOUT)
        if not utterance:
            self._log("WARNING", "SS1: no response to greeting")
            result["abort_reason"] = result.get("abort_reason") or "no_response_greeting"
            return

        self._log("INFO", f"SS1: response: '{utterance}'")
        if self._check_abort(result):
            return

        # 3) Ask name
        self._clear_stt_buffer()
        self._speak("We have not met, what's your name?")
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
            self._speak("Sorry, I didn't catch that. What's your name?")
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

        threading.Thread(target=self._submit_face_name, args=(track_id, name), daemon=True).start()
        threading.Thread(
            target=self._write_last_greeted,
            args=(track_id, face_id, name, name),
            daemon=True
        ).start()

        self._speak_and_wait("Nice to meet you")

        result["success"]     = True
        result["greeted"]     = True
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
            self._speak_and_wait(f"Hello {face_id}")
            result["greeted"] = True

            if self._check_abort(result):
                return

            utterance = self._wait_user_utterance_abortable(self.SS2_GREET_TIMEOUT)
            if utterance:
                self._log("INFO", f"SS2: response: '{utterance}'")
                threading.Thread(target=self._write_last_greeted,
                                 args=(track_id, face_id, face_id, face_id), daemon=True).start()
                result["success"]     = True
                result["final_state"] = "ss3"
                self._log("INFO", "SS2: → ss3 tree")
                self._run_ss3_tree(track_id, face_id, result)
                return
            else:
                self._log("WARNING", f"SS2: no response attempt {attempt+1}")

        result["abort_reason"] = "no_response_greeting"
        self._log("INFO", "SS2: failed after 2 attempts")

    # ==================== Tree SS3: Known, greeted, not talked ====================

    def _run_ss3_tree(self, track_id: int, face_id: str, result: Dict):
        """
        Short conversation with proactive starter.
        Up to 3 turns; turn 3 is acknowledgment only.
        If at least one response → talked=True → ss4.
        """
        self._log("INFO", "SS3: start")

        if self._check_abort(result):
            return

        starter = self._cached_starter or "How are you doing these days?"
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

            if is_last:
                future = self._llm_pool.submit(self._llm_generate_closing_acknowledgment, utterance)
            else:
                future = self._llm_pool.submit(self._llm_generate_followup, utterance, [])

            default    = "That's nice!" if is_last else "I see."
            reply_text = self._await_future_abortable(future, default, self.LLM_TIMEOUT)

            if reply_text is None:
                break

            if self._check_abort(result):
                break

            self._speak_and_wait(reply_text)

        if user_responded:
            result["success"]     = True
            result["talked"]      = True
            result["final_state"] = "ss4"
            self._log("INFO", f"SS3: complete ({turns} turns)")
        else:
            result["abort_reason"] = result.get("abort_reason") or "no_response_conversation"
            self._log("WARNING", "SS3: no user response")

    # ==================== Hunger / QR Feeding ====================

    def _run_hunger_feed_tree(self, track_id: int, face_id: str, social_state: str, hs: str, result: Dict):
        self._log("INFO", f"Hunger tree: {hs}")
        feed_wait_timeout_sec = float(self._feed_wait_timeout_sec)
        
        self._clear_stt_buffer()
        self._speak_and_wait("I'm so hungry, would you feed me please?")
        
        meals_eaten = 0
        start_wait_ts = time.time()
        
        while not self._check_abort(result):
            fed, payload, new_ts = self._wait_for_feed_since(start_wait_ts, feed_wait_timeout_sec)
            
            if fed:
                meals_eaten += 1
                result["last_meal_payload"] = payload
                with self.hunger._lock:
                    lvl = self.hunger.level
                self._log("INFO", f"Proactive feed: {payload} → meal #{meals_eaten}, stomach {lvl:.1f}")
                self._speak_and_wait("Yummy, thank you so much.")
                
                new_hs = self.hunger.get_state()
                if new_hs == "HS1":
                    self._log("INFO", "Hunger satisfied, ending feeding interaction.")
                    break
                else:
                    self._speak_and_wait("I'm still hungry. Give me more please.")
                    start_wait_ts = new_ts
            else:
                if self._check_abort(result):
                    break
                start_wait_ts = time.time()
                self._execute_behaviour(self._feed_timeout_behaviour)
        
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

    def _wait_for_feed_since(self, ts: float, timeout: float) -> Tuple[bool, Optional[str], float]:
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
        try:
            import cv2
            import numpy as np
            qr_detector = cv2.QRCodeDetector()
        except ImportError:
            self._log("WARNING", "cv2/numpy not found – QR feeding disabled")
            return

        while self._running and not self._qr_stop_event.is_set():
            if not self.cam_left_port:
                time.sleep(0.1)
                continue
                
            try:
                yimg = self.cam_left_port.read(False)
                if not yimg:
                    time.sleep(0.05)
                    continue
                    
                w, h = yimg.width(), yimg.height()
                if w <= 0 or h <= 0:
                    continue

                frame = None
                try:
                    # Use getRawImage() + ctypes for a fast zero-copy numpy view.
                    # toBytes() does not exist in YARP Python bindings.
                    raw = yimg.getRawImage()
                    sz = yimg.getRawImageSize()
                    row_size = yimg.getRowSize()
                    buf = (ctypes.c_uint8 * sz).from_address(int(raw))
                    arr = np.frombuffer(buf, dtype=np.uint8).copy()
                    if row_size == w * 3:
                        frame = arr[:h * w * 3].reshape((h, w, 3))
                    else:
                        # Row padding present – extract each row individually
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                        for row_idx in range(h):
                            rs = row_idx * row_size
                            frame[row_idx] = arr[rs:rs + w * 3].reshape((w, 3))
                except Exception:
                    pass

                if frame is None:
                    self._log("WARNING", "QR: could not decode YARP image to numpy array")
                    time.sleep(0.1)
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                raw_val, pts, _ = qr_detector.detectAndDecode(gray)
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
                    self._enqueue_responsive_event("qr_feed", {
                        "payload": val,
                        "delta": delta,
                        "timestamp": now,
                    })
                    
                    hs_now = self.hunger.get_state()
                    with self.hunger._lock:
                        lvl = self.hunger.level
                    self._log("INFO", f"QR: {val} (+{delta}) → stomach {lvl:.1f} ({hs_now})")
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
            self._log("DEBUG", f"Dropping responsive event '{event_type}' (proactive running)")
            return
        try:
            self._responsive_queue.put_nowait((event_type, payload))
        except queue.Full:
            self._log("DEBUG", f"Dropping responsive event '{event_type}' (queue full)")

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
                    self._log("DEBUG", f"Dropping responsive event '{event_type}' (interaction busy)")
                time.sleep(0.05)
                continue

            if event_type == "qr_feed":
                self._run_responsive_qr_ack(payload if isinstance(payload, dict) else {})
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
                    self._log("DEBUG", f"Responsive: STT '{utterance[:40]}' - not a greeting")
                    continue

                self._log("INFO", f"Responsive: greeting detected '{utterance}'")

                candidate = self._responsive_single_candidate()
                if not candidate:
                    self._log("DEBUG", "Responsive: no candidate found, skipping")
                    continue

                track_id, name = candidate
                if self._is_greet_in_responsive_cooldown(name):
                    self._log("DEBUG", f"Responsive: '{name}' in cooldown, skipping")
                    continue

                self._run_responsive_greeting(track_id, name)
            except Exception as e:
                self._log("WARNING", f"Responsive loop iteration failed: {e}")
                time.sleep(0.05)

        self._log("INFO", "Responsive loop stopped")

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

    def _responsive_single_candidate(self) -> Optional[Tuple[int, str]]:
        """Find the best candidate for responsive greeting.

        Returns the biggest-bbox known face. Gaze/attention is NOT required —
        the utterance itself is sufficient signal that the person is addressing
        the robot.
        """
        with self._faces_lock:
            age = time.time() - self._latest_faces_ts
            cache_size = len(self._latest_faces)

        faces = self.parse_landmarks_latest(staleness_sec=30.0)
        if not faces:
            self._log("DEBUG", f"Responsive: no faces in landmarks (cache age={age:.1f}s, cached={cache_size})")
            return None

        candidates: List[Tuple[int, str, float]] = []  # (track_id, name, area)

        for face in faces:
            face_id = str(face.get("face_id", "")).strip()
            track_id = face.get("track_id")
            bbox = face.get("bbox", [0, 0, 0, 0])
            area = bbox[2] * bbox[3] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 0

            if not self._is_responsive_known_name(face_id):
                self._log("DEBUG", f"Responsive: track={track_id} face_id='{face_id}' (not known)")
                continue

            if not isinstance(track_id, int):
                continue

            candidates.append((track_id, face_id, area))

        if not candidates:
            self._log("DEBUG", f"Responsive: no known faces from {len(faces)} total")
            return None

        # Pick biggest bbox among known faces
        best = max(candidates, key=lambda c: c[2])
        self._log("DEBUG", f"Responsive: selected {best[1]} (track={best[0]}, area={best[2]:.0f})")
        return (best[0], best[1])

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
            t = threading.Thread(target=self._execute_behaviour, args=("ao_start",), daemon=True)
            t.start()
            self._responsive_speak_and_wait("yummy, thank you")
            t.join(timeout=5.0)
            self._execute_behaviour("ao_stop")
            self._db_queue.put(("responsive", {
                "type": "qr_feed",
                "track_id": None,
                "name": None,
                "payload": meal_payload,
            }))
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
        try:
            self._log("INFO", f"Responsive greeting: '{name}' (track={track_id})")
            t = threading.Thread(target=self._execute_behaviour, args=("ao_start",), daemon=True)
            t.start()
            self._responsive_speak_and_wait(f"Hi {name}")
            t.join(timeout=5.0)
            self._execute_behaviour("ao_stop")
            self._write_last_greeted(track_id, face_id=name, code=name, person_key=name)
            self._mark_greeted_today(name)
            self._db_queue.put(("responsive", {
                "type": "greeting",
                "track_id": track_id,
                "name": name,
                "payload": None,
            }))
        except Exception as e:
            self._log("WARNING", f"Responsive greeting failed: {e}")
        finally:
            self.run_lock.release()
            self._responsive_active.clear()

    def _responsive_speak_and_wait(self, text: str) -> bool:
        ok = self._speak(text)
        wc = len(text.split())
        wait = wc / self.TTS_WORDS_PER_SECOND + self.TTS_END_MARGIN
        wait = max(self.TTS_MIN_WAIT, min(self.TTS_MAX_WAIT, wait))
        time.sleep(wait)
        return ok

    # ==================== Abort helpers ====================

    def _await_future_abortable(self, future, default, total_timeout):
        deadline = time.time() + total_timeout
        while time.time() < deadline:
            if self.abort_event.is_set():
                future.cancel()
                return None
            try:
                res = future.result(timeout=0.1)
                return res if res else default
            except concurrent.futures.TimeoutError:
                continue
        return default

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
                start += (scan_duration - 0.1)
                self._log("DEBUG", f"Main loop blocked for {scan_duration:.2f}s, extending timeout")
        return None

    def _try_extract_name(self, utterance: str) -> Optional[str]:
        # Fast regex check for common patterns
        match = re.search(r"(?i)(?:my name is|my name's|i am|i'm|im|call me)\s+([a-z-]+)", utterance)
        if match:
             return match.group(1).title()

        extraction = self._llm_extract_name(utterance)
        if extraction.get("answered") and extraction.get("name"):
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
                        self._log("DEBUG", f"Landmarks: bottle size={bottle.size()} but 0 faces parsed")
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
                            data[key] = [lst.get(j).asFloat64() for j in range(1, lst.size())]
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

    def _prewarm_rpc_connections(self):
        """Eagerly open the interaction RPC connection so the first behaviour
        command is sent without the TCP-setup delay."""
        time.sleep(1.0)  # brief wait for interactionInterface to register
        for attempt in range(5):
            try:
                self._get_interaction_rpc()
                self._log("INFO", "Interaction RPC pre-warmed")
                return
            except Exception as e:
                self._log("DEBUG", f"RPC pre-warm attempt {attempt + 1} failed: {e}")
                time.sleep(2.0)

    def _get_interaction_rpc(self) -> yarp.RpcClient:
        if self._interaction_rpc is None:
            client = yarp.RpcClient()
            lp = f"/{self.module_name}/interactionInterface/rpc"
            if not client.open(lp):
                raise RuntimeError(f"Failed to open {lp}")
            if not client.addOutput("/interactionInterface"):
                client.close()
                raise RuntimeError("Failed to connect to /interactionInterface")
            self._interaction_rpc = client
        return self._interaction_rpc

    def _get_vision_rpc(self) -> yarp.RpcClient:
        if self._vision_rpc is None:
            client = yarp.RpcClient()
            lp = f"/{self.module_name}/objectRecognition/rpc"
            if not client.open(lp):
                raise RuntimeError(f"Failed to open {lp}")
            if not client.addOutput("/objectRecognition"):
                client.close()
                raise RuntimeError("Failed to connect to /objectRecognition")
            self._vision_rpc = client
        return self._vision_rpc

    def _execute_behaviour(self, behaviour: str) -> bool:
        try:
            rpc = self._get_interaction_rpc()
            cmd = yarp.Bottle()
            cmd.addString("exe")
            cmd.addString(behaviour)
            reply = yarp.Bottle()
            self._log("INFO", f"Behaviour '{behaviour}' sending...")
            rpc.write(cmd, reply)
            self._log("INFO", f"Behaviour '{behaviour}' ack'd")
            return True
        except Exception as e:
            self._log("ERROR", f"Behaviour failed: {e}")
            if self._interaction_rpc:
                self._interaction_rpc.close()
                self._interaction_rpc = None
            return False

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

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.DB_FILE)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                track_id INTEGER,
                name TEXT,
                payload TEXT
            )""")
            conn.commit()
            conn.close()
            self._log("INFO", f"DB ready: {self.DB_FILE}")
        except Exception as e:
            self._log("ERROR", f"DB init failed: {e}")

    def _db_worker(self):
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
                if table == "interaction":
                    self._save_interaction_to_db(data)
                elif table == "responsive":
                    self._save_responsive_interaction_to_db(data)
            except Exception as e:
                self._log("ERROR", f"DB write failed: {e}")

    def _save_interaction_to_db(self, data: Dict):
        try:
            r = data["result"]
            conn = sqlite3.connect(self.DB_FILE, timeout=10.0)
            c = conn.cursor()
            # Build transcript from logs
            transcript_lines = [
                log["message"] for log in r.get("logs", [])
                if "User:" in log.get("message", "") or "Robot:" in log.get("message", "")
                   or "Asking" in log.get("message", "") or "Response" in log.get("message", "")
            ]
            c.execute("""INSERT INTO interactions
                (timestamp,track_id,face_id,initial_state,final_state,success,abort_reason,
                 greeted,talked,extracted_name,target_stayed_biggest,transcript,full_result)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now().isoformat(), data["track_id"], data["face_id"],
                 data["initial_state"], r.get("final_state", ""),
                 int(r.get("success", False)), r.get("abort_reason"),
                 int(r.get("greeted", False)), int(r.get("talked", False)),
                 r.get("extracted_name"),
                 int(r.get("target_stayed_biggest", True)),
                 json.dumps(transcript_lines, ensure_ascii=False),
                 json.dumps(r, ensure_ascii=False)))
            conn.commit()
            conn.close()
        except Exception as e:
            self._log("ERROR", f"DB save failed: {e}")

    def _save_responsive_interaction_to_db(self, data: Dict[str, Any]):
        try:
            conn = sqlite3.connect(self.DB_FILE, timeout=10.0)
            c = conn.cursor()
            c.execute("""INSERT INTO responsive_interactions
                (timestamp,type,track_id,name,payload)
                VALUES (?,?,?,?,?)""",
                (datetime.now().astimezone().isoformat(),
                 data.get("type"),
                 data.get("track_id"),
                 data.get("name"),
                 data.get("payload")))
            conn.commit()
            conn.close()
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

    def _write_last_greeted(self, track_id: int, face_id: str, code: str, person_key: Optional[str] = None):
        try:
            path = self.LAST_GREETED_FILE
            raw_entries = self._load_json(path, {})
            entries: Dict[str, Dict[str, Any]] = {}

            if isinstance(raw_entries, dict):
                entries = {str(k): v for k, v in raw_entries.items() if isinstance(v, dict)}
            elif isinstance(raw_entries, list):
                for entry in raw_entries:
                    if not isinstance(entry, dict):
                        continue
                    legacy_key = entry.get("assigned_code_or_name") or entry.get("face_id")
                    if legacy_key:
                        entries[str(legacy_key)] = entry

            key = (person_key or "").strip()
            if not key:
                if face_id and face_id.lower() not in ("unknown", "unmatched", "recognizing"):
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
            raw = self._load_json(path, {})
            entries = raw if isinstance(raw, dict) else {}
            entries[key] = datetime.now().astimezone().isoformat()
            self._save_json_atomic(path, entries)
        except Exception as e:
            self._log("WARNING", f"Write greeted_today failed: {e}")

    # ==================== LLM Integration ====================

    def _check_ollama_binary_exists(self) -> bool:
        for path in ["/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/ollama/bin/ollama"]:
            if os.path.exists(path):
                return True
        return False

    def _install_ollama_server(self) -> bool:
        try:
            self._log("INFO", "Installing Ollama...")
            r = subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True, capture_output=True, text=True, timeout=300
            )
            return r.returncode == 0
        except Exception as e:
            self._log("ERROR", f"Ollama install failed: {e}")
            return False

    def _start_ollama_server(self) -> bool:
        try:
            try:
                req = urllib.request.Request(f"{self.OLLAMA_URL}/api/tags")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            subprocess.Popen(["ollama", "serve"],
                             stdout=open("/tmp/ollama_server.log", "w"),
                             stderr=subprocess.STDOUT, start_new_session=True)
            for i in range(30):
                time.sleep(1)
                try:
                    req = urllib.request.Request(f"{self.OLLAMA_URL}/api/tags")
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    continue
            return False
        except Exception as e:
            self._log("ERROR", f"Ollama start failed: {e}")
            return False

    def ensure_ollama_and_model(self) -> bool:
        try:
            if not self._check_ollama_binary_exists():
                if not self._install_ollama_server():
                    return False
            if not self._start_ollama_server():
                return False
            req = urllib.request.Request(f"{self.OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                if not any(self.LLM_MODEL in m for m in models):
                    subprocess.run(["ollama", "pull", self.LLM_MODEL],
                                   capture_output=True, text=True, timeout=600)
            self.ollama_last_check = time.time()
            return True
        except Exception as e:
            self._log("ERROR", f"Ollama setup failed: {e}")
            return False

    def _llm_request(self, prompt: str, json_format: bool = False, system: Optional[str] = None, options: Optional[dict] = None, format_obj: Optional[object] = None) -> str:
        last_error = None
        for attempt in range(self.llm_retry_attempts):
            try:
                t = time.time()
                if t - self.ollama_last_check > self.ollama_check_interval:
                    self.ollama_last_check = t
                payload: Dict[str, Any] = {"model": self.LLM_MODEL, "prompt": prompt, "stream": False}
                if system is not None:
                    payload["system"] = system
                if options is not None:
                    payload["options"] = options
                if format_obj is not None:
                    payload["format"] = format_obj
                elif json_format:
                    payload["format"] = "json"
                data = json.dumps(payload).encode()
                req = urllib.request.Request(f"{self.OLLAMA_URL}/api/generate",
                                             data=data,
                                             headers={"Content-Type": "application/json"})
                http_timeout = min(self.LLM_TIMEOUT, 10.0)
                with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                    result = json.loads(resp.read().decode()).get("response", "").strip()
                    if result:
                        return result
                    last_error = "Empty response"
            except Exception as e:
                last_error = str(e)
                if attempt < self.llm_retry_attempts - 1:
                    time.sleep(self.llm_retry_delay)
        self._log("ERROR", f"LLM failed after {self.llm_retry_attempts} attempts: {last_error}")
        return ""

    def _llm_json(self, prompt: str, system: Optional[str] = None, options: Optional[dict] = None, format_obj: Optional[object] = None) -> Dict:
        text = self._llm_request(prompt, json_format=True, system=system, options=options, format_obj=format_obj)
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
                r = json.loads(text[start:end+1])
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
        "hello", "hi", "hey", "ciao", "hola", "salut", "hallo",
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok",
        "good morning", "good afternoon", "good evening",
        "howdy", "greetings", "sup", "what's up",
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
        prompt = (f"Extract the explicitly stated name of the person speaking from the utterance.\n"
                  f"Utterance: '{utterance}'\n"
                  f"Rules:\n"
                  f"1. If no name is explicitly stated, you MUST return answered=false and name=null.\n"
                  f"2. Do NOT guess names from greetings or conversational filler.\n"
                  f"3. If multiple names appear, pick the first explicitly self-referential name.\n"
                  f"4. Preserve capitalization as a normal name (Title Case) in output.\n"
                  f"Examples:\n"
                  f"- \"Hi, I'm Robert\" -> {{\"answered\": true, \"name\": \"Robert\", \"confidence\": 1.0}}\n"
                  f"- \"Hello there\" -> {{\"answered\": false, \"name\": null, \"confidence\": 0.0}}\n"
                  f"- \"Call me Alice\" -> {{\"answered\": true, \"name\": \"Alice\", \"confidence\": 1.0}}\n"
                  f"- \"Hey, how are you?\" -> {{\"answered\": false, \"name\": null, \"confidence\": 0.0}}\n"
                  f"- \"Yes, I am Sarah and this is Bob.\" -> {{\"answered\": true, \"name\": \"Sarah\", \"confidence\": 1.0}}\n")
        schema = {
            "type": "object",
            "properties": {
                "answered": {"type": "boolean"},
                "name": {"type": ["string", "null"]},
                "confidence": {"type": "number"}
            },
            "required": ["answered", "name", "confidence"],
            "additionalProperties": False
        }
        r = self._llm_json(prompt, system=self.LLM_SYSTEM_JSON, options={"temperature": 0, "num_predict": 80}, format_obj=schema)
        return {"answered": r.get("answered", False) is True,
                "name": r.get("name") or None,
                "confidence": float(r.get("confidence", 0) or 0)}

    def _llm_generate_convo_starter(self) -> str:
        text = self._llm_request(
            "Generate a natural conversation starter. NO greetings. Ask ONE short question about their day or wellbeing. "
            "Under 15 words. Output ONLY the sentence, NO quotes.",
            system=self.LLM_SYSTEM_DEFAULT,
            options={"temperature": 0.4, "num_predict": 40}
        )
        return text.strip("\"'").strip() if text and len(text) < 150 else "How are you doing these days?"

    def _generate_starter_background(self):
        try:
            starter = self._llm_generate_convo_starter()
            if starter:
                self._cached_starter = starter
        except Exception as e:
            self._log("WARNING", f"Starter prefetch failed: {e}")

    def _llm_generate_followup(self, last_utterance: str, history: List[str]) -> str:
        text = self._llm_request(
            f"User said: '{last_utterance}'\n"
            f"Generate a friendly, natural response. 1-2 sentences. Under 25 words. "
            f"NO meta-commentary. Output ONLY the response, NO quotes.",
            system=self.LLM_SYSTEM_DEFAULT,
            options={"temperature": 0.5, "num_predict": 80}
        )
        return text.strip("\"'").strip() if text and len(text) < 200 else "That's interesting!"

    def _llm_generate_closing_acknowledgment(self, last_utterance: str) -> str:
        text = self._llm_request(
            f"Person said: '{last_utterance}'\n"
            f"Generate a warm acknowledgment. NO questions. Under 10 words. "
            f"Output ONLY the acknowledgment, NO quotes.",
            system=self.LLM_SYSTEM_DEFAULT,
            options={"temperature": 0.4, "num_predict": 30}
        )
        return text.strip("\"'").strip() if text and len(text) < 100 else "That's nice!"

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

    module = InteractionManagerModule()
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.configure(sys.argv)

    print("=" * 60)
    print(" InteractionManagerModule")
    print(" ss1=unknown  ss2=known/not-greeted  ss3=greeted  ss4=no-op")
    print()
    print(" yarp connect /alwayson/vision/landmarks:o /interactionManager/landmarks:i")
    print(" yarp connect /speech2text/text:o          /interactionManager/stt:i")
    print(" yarp connect /icub/cam/left               /interactionManager/camLeft:i")
    print(" yarp connect /interactionManager/speech:o /acapelaSpeak/speech:i")
    print()
    print(" RPC: echo 'run <track_id> <face_id> <ss1|ss2|ss3|ss4>' | yarp rpc /interactionManager")
    print("=" * 60)

    try:
        module.runModule(rf)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        module.interruptModule()
        module.close()
        yarp.Network.fini()
