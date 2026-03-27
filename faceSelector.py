"""salienceNetwork.py - real-time face selection and interaction gating."""

import fcntl
import json
import os
import queue
import sqlite3
from dataclasses import asdict, dataclass
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    import yarp
except ImportError:
    print("[ERROR] YARP Python bindings required.")
    sys.exit(1)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


class SalienceNetworkModule(yarp.RFModule):
    """Selects faces by IPS and triggers executive interactions.

    Pipeline:
      Input       -> vision landmarks + STM context + memory files
      Decision    -> social state + IPS + arbitration (override/interaction/cooldown)
      Output      -> targetCmd to vision + run/status RPC to executive
      Persistence -> JSON memories + SQLite event logs
    """

    @dataclass(frozen=True)
    class FaceSnapshot:
        face_id: str = "unknown"
        track_id: int = -1
        bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        distance: str = "UNKNOWN"
        gaze_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        pitch: float = 0.0
        yaw: float = 0.0
        roll: float = 0.0
        cos_angle: float = 0.0
        attention: str = "AWAY"
        is_talking: int = 0
        time_in_view: float = 0.0

    @dataclass(frozen=True)
    class InteractionAttempt:
        attempt_id: str
        track_id: int
        face_id: str
        person_id: str
        start_ss: str
        success: int
        final_state: Optional[str]
        abort_reason: Optional[str]
        exec_interaction_id: Optional[str]
        duration_sec: float

    @dataclass(frozen=True)
    class LearningDelta:
        person_id: str
        reward_delta: float
        outcome: str
        reason: str
        success: int
        abort_reason: Optional[str]
        name_extracted: int
        exec_interaction_id: Optional[str]
        old_prox: Optional[float]
        old_cent: Optional[float]
        old_vel: Optional[float]
        old_gaze: Optional[float]
        new_prox: Optional[float]
        new_cent: Optional[float]
        new_vel: Optional[float]
        new_gaze: Optional[float]

    # ==================== Adaptive IPS Constants ====================
    # Baseline IPS weights
    BASELINE_WEIGHTS = {"prox": 0.5, "cent": 0.15, "vel": 0.3, "gaze": 0.5}

    # Minimum IPS by social state
    SS_THRESHOLDS = {
        "ss1": 1.0,  # Stranger: standard hurdle
        "ss2": 0.8,  # Friend (ungreeted): eager to initiate
        "ss3": 1.2,  # Friend (greeted): needs strong intent to bother again
        "ss4": 99.0,  # Ultimate: never proactive
    }

    IPS_HYSTERESIS_BONUS = 0.3  # Stickiness for the current target
    HABITUATION_LAMBDA = 0.05  # Habituation decay
    WEIGHT_SHIFT_RATE = 0.05  # How much weights drift per interaction (+/-)
    TARGET_LOG_MIN_PERIOD_SEC = 1.0
    TARGET_LOG_IPS_DELTA = 0.15
    DB_QUEUE_MAXSIZE = 1024
    IO_QUEUE_MAXSIZE = 256

    # Social-state labels
    SS_DESCRIPTIONS = {
        "ss1": "Unknown",
        "ss2": "Known, Not Greeted",
        "ss3": "Known, Greeted, No Talk",
        "ss4": "Known, Greeted, Talked",
    }
    TIMEZONE = ZoneInfo("Europe/Rome")

    # ==================== Lifecycle ====================

    def __init__(self):
        super().__init__()

        self.module_name = "salienceNetwork"
        self.period = 0.05  # 20 Hz
        self._running = True

        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

        self.executive_control_rpc_name = "/executiveControl"

        self.learning_path = Path(
            "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/learning.json"
        )
        self.greeted_path = Path(
            "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/greeted_today.json"
        )
        self.talked_path = Path(
            "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/talked_today.json"
        )
        self.last_greeted_path = Path(
            "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/last_greeted.json"
        )

        # YARP ports
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.debug_port: Optional[yarp.Port] = None
        self.vision_cmd_port: Optional[yarp.BufferedPortBottle] = None
        self.executive_control_rpc: Optional[yarp.RpcClient] = None
        self.facetracker_rpc: Optional[yarp.RpcClient] = None
        self.stm_context_port: Optional[yarp.BufferedPortBottle] = None

        # RPC input (set_track_id from executiveControl)
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)
        self.rpc_override_track_id: int = -1  # -1 = no override (normal IPS)
        # State tracking (thread-safe)
        self.state_lock = threading.Lock()
        self.current_faces: List[Dict[str, Any]] = []
        self.selected_target: Optional[Dict[str, Any]] = None
        self.selected_bbox_last: Optional[Tuple[float, float, float, float]] = None
        self.interaction_busy = False
        self.interaction_thread: Optional[threading.Thread] = None

        self.area_history: Dict[int, float] = {}  # Maps track_id to previous bbox area
        self.current_target_track_id: int = -1  # For applying Hysteresis

        # Guards memory dicts and file I/O
        self._memory_lock = threading.Lock()
        # Guards interaction_busy transitions and spawn decisions
        self._interaction_lock = threading.Lock()

        # Cooldown config
        self.last_interaction_time: Dict[str, float] = {}
        self.cooldown_lively: float = 3.0
        self.cooldown_calm: float = 15.0
        self.cooldown_default: float = 5.0
        self.min_track_ips: float = 0.6
        self.exec_rpc_retry_sec: float = 1.0

        self.current_context_label: int = -1
        self.frame_skip_counter = 0
        self.frame_skip_rate = 0

        # Cache recent landmarks for sparse upstream publishing.
        self._latest_landmarks: List[Dict[str, Any]] = []
        self._latest_landmarks_ts: float = 0.0
        self.landmarks_stale_sec: float = 0.30

        self.greeted_today: Dict[str, str] = {}
        self.talked_today: Dict[str, str] = {}
        self.learning_data: Dict[str, Dict] = {}

        self._last_greeted_snapshot: Dict[str, Dict[str, Any]] = {}
        self._last_greeted_lock = threading.Lock()

        self.track_to_person: Dict[int, str] = {}
        self._current_day: Optional[date] = None

        self.verbose_debug = False
        self.ports_connected_logged = False
        self.status_log_period_sec: float = 1.0
        self._last_status_log_ts: float = 0.0
        self._last_status_line: str = ""
        self._last_target_key: Tuple[int, str, str] = (-2, "", "")
        self._last_sent_track_id: int = -99999
        self._last_sent_ips: float = -1.0
        self._next_exec_rpc_try_ts: float = 0.0
        self._exec_rpc_offline_logged: bool = False
        self._last_target_log_key: Tuple[int, str, str, int] = (-2, "", "", -1)
        self._last_target_log_ips: float = -1.0
        self._last_target_log_ts: float = 0.0

        self.DISAPPEAR_WINDOW_SEC = 30.0
        self.DISAPPEAR_THRESHOLD = 2
        self._disappear_events: Dict[str, List[float]] = {}

        self._io_queue: queue.Queue = queue.Queue(maxsize=self.IO_QUEUE_MAXSIZE)
        self._io_thread: Optional[threading.Thread] = None

        self.db_path = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/data_collection/salience_network.db"
        self._db_queue: queue.Queue = queue.Queue(maxsize=self.DB_QUEUE_MAXSIZE)
        self._db_thread: Optional[threading.Thread] = None
        self._context_connected_logged = False

    # ------------------------------------------------------------------ configure
    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            try:
                self.setName(self.module_name)
            except Exception:
                pass

            if rf.check("executive_control_rpc"):
                self.executive_control_rpc_name = rf.find(
                    "executive_control_rpc"
                ).asString()
            if rf.check("learning_path"):
                self.learning_path = Path(rf.find("learning_path").asString())
            if rf.check("greeted_path"):
                self.greeted_path = Path(rf.find("greeted_path").asString())
            if rf.check("talked_path"):
                self.talked_path = Path(rf.find("talked_path").asString())
            if rf.check("rate"):
                self.period = rf.find("rate").asFloat64()
            if rf.check("verbose"):
                self.verbose_debug = rf.find("verbose").asBool()
            if rf.check("landmarks_stale_sec"):
                self.landmarks_stale_sec = rf.find("landmarks_stale_sec").asFloat64()
            if rf.check("status_log_period_sec"):
                self.status_log_period_sec = max(
                    0.1, rf.find("status_log_period_sec").asFloat64()
                )
            if rf.check("min_track_ips"):
                self.min_track_ips = max(0.0, rf.find("min_track_ips").asFloat64())
            if rf.check("exec_rpc_retry_sec"):
                self.exec_rpc_retry_sec = max(
                    0.1, rf.find("exec_rpc_retry_sec").asFloat64()
                )

            # --- Open ports ---
            def _open_port(attr: str, cls, name: str) -> bool:
                port = cls()
                next_log_ts = 0.0
                while not port.open(name):
                    now = time.time()
                    if now >= next_log_ts:
                        self._log("INFO", f"Waiting local port availability: {name}")
                        next_log_ts = now + 2.0
                    try:
                        port.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                setattr(self, attr, port)
                return True

            if not _open_port(
                "landmarks_port",
                yarp.BufferedPortBottle,
                f"/alwayson/{self.module_name}/landmarks:i",
            ):
                return False

            if not _open_port(
                "vision_cmd_port",
                yarp.BufferedPortBottle,
                f"/alwayson/{self.module_name}/targetCmd:o",
            ):
                return False

            if not _open_port(
                "debug_port", yarp.Port, f"/alwayson/{self.module_name}/debug:o"
            ):
                return False

            handle_name = f"/{self.module_name}"
            handle_next_log_ts = 0.0
            while not self.handle_port.open(handle_name):
                now = time.time()
                if now >= handle_next_log_ts:
                    self._log("INFO", f"Waiting local port availability: {handle_name}")
                    handle_next_log_ts = now + 2.0
                time.sleep(0.5)

            self.stm_context_port = yarp.BufferedPortBottle()
            _ctx_local = f"/alwayson/{self.module_name}/context:i"
            ctx_next_log_ts = 0.0
            while not self.stm_context_port.open(_ctx_local):
                now = time.time()
                if now >= ctx_next_log_ts:
                    self._log("INFO", f"Waiting local port availability: {_ctx_local}")
                    ctx_next_log_ts = now + 2.0
                time.sleep(0.5)

            self.executive_control_rpc = yarp.RpcClient()
            exec_rpc_local = f"/{self.module_name}/executiveControl:rpc"
            if not self.executive_control_rpc.open(exec_rpc_local):
                self._log("ERROR", f"Failed to open port: {exec_rpc_local}")
                return False
            if yarp.Network.connect(exec_rpc_local, self.executive_control_rpc_name):
                self._log("INFO", f"RPC connected → {self.executive_control_rpc_name}")

            # --- Connect to FaceTracker RPC and send 'run' ---
            self.facetracker_rpc = yarp.RpcClient()
            face_rpc_local = f"/{self.module_name}/faceTracker:rpc"
            if not self.facetracker_rpc.open(face_rpc_local):
                self._log("ERROR", f"Failed to open port: {face_rpc_local}")
                return False
            yarp.Network.connect(face_rpc_local, "/faceTracker/rpc")

            manual_connections = [
                f"yarp connect /alwayson/vision/landmarks:o /alwayson/{self.module_name}/landmarks:i",
                f"yarp connect /alwayson/{self.module_name}/targetCmd:o /alwayson/vision/targetCmd:i",
            ]
            if self.stm_context_port is not None:
                manual_connections.append(
                    f"yarp connect /alwayson/stm/context:o /alwayson/{self.module_name}/context:i"
                )

            self._log("INFO", "Connect these ports:")
            for cmd in manual_connections:
                self._log("INFO", f"  {cmd}")

            self._wait_for_manual_connections()
            self._send_facetracker_cmd("run", retries=1)

            self._load_all_json_files()
            self._refresh_last_greeted_snapshot()
            self._current_day = self._get_today_date()

            self._init_db()
            for target, name in [
                (self._io_worker, "_io_thread"),
                (self._db_worker, "_db_thread"),
                (self._last_greeted_refresh_loop, "_lg_refresh_thread"),
            ]:
                t = threading.Thread(target=target, daemon=True)
                t.start()
                setattr(self, name, t)

            self._log("INFO", f"SalienceNetworkModule ready @ {1.0/self.period:.0f} Hz")
            threading.Thread(target=self._prewarm_rpc_connections, daemon=True).start()
            return True

        except Exception as e:
            self._log("ERROR", f"configure() failed: {e}")
            return False

    def _wait_for_manual_connections(self):
        """Log current connection status once without blocking startup.

        Required streams are handled in updateModule() as data arrives.
        STM context is optional and can be connected at any time.
        """
        checks = [
            (
                f"/alwayson/vision/landmarks:o -> /alwayson/{self.module_name}/landmarks:i",
                lambda: self.landmarks_port is not None
                and self.landmarks_port.getInputCount() > 0,
            ),
            (
                f"/alwayson/{self.module_name}/targetCmd:o -> /alwayson/vision/targetCmd:i",
                lambda: self.vision_cmd_port is not None
                and self.vision_cmd_port.getOutputCount() > 0,
            ),
        ]

        pending = [name for name, ok in checks if not ok()]
        if pending:
            pending_labels = ", ".join(
                item.split(" -> ")[0].split("/")[-1] for item in pending
            )
            self._log(
                "INFO",
                f"Startup without blocking; pending required ports: {pending_labels}",
            )
        else:
            self._log("INFO", "Required ports already connected.")

        if self.stm_context_port is not None and self.stm_context_port.getInputCount() == 0:
            self._log("INFO", "STM context port is optional; can connect later.")

    def interruptModule(self) -> bool:
        self._log("INFO", "Interrupting...")
        self._running = False
        for port in [
            self.landmarks_port,
            self.debug_port,
            self.executive_control_rpc,
            self.vision_cmd_port,
            self.stm_context_port,
            self.handle_port,
            self.facetracker_rpc,
        ]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        self._log("INFO", "Closing...")
        if self.interaction_thread and self.interaction_thread.is_alive():
            self.interaction_thread.join(timeout=5.0)
        self._enqueue_save("greeted")
        self._enqueue_save("talked")
        self._enqueue_save("learning")
        self._queue_put_drop_oldest(self._io_queue, None, "IO queue close")
        if self._io_thread:
            self._io_thread.join(timeout=5.0)
        self._queue_put_drop_oldest(self._db_queue, None, "DB queue close")
        if self._db_thread:
            self._db_thread.join(timeout=3.0)
        if self.facetracker_rpc:
            # Send 'sus' before shutting down, only try once so we don't delay close()
            self._send_facetracker_cmd("sus", retries=1)
            self.facetracker_rpc.close()

        for port in [
            self.landmarks_port,
            self.debug_port,
            self.executive_control_rpc,
            self.vision_cmd_port,
            self.stm_context_port,
            self.handle_port,
        ]:
            if port:
                port.close()
        return True

    def respond(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        """Handle RPC commands (e.g. set_track_id from executiveControl)."""
        reply.clear()
        if cmd.size() < 1:
            reply.addString("error")
            reply.addString("empty command")
            return True
        command = cmd.get(0).asString()
        if command == "set_track_id":
            if cmd.size() < 2:
                reply.addString("error")
                reply.addString("usage: set_track_id <int>")
                return True
            tid = cmd.get(1).asInt32()
            self.rpc_override_track_id = tid
            self._log("INFO", f"RPC override track_id set to {tid}")
            reply.addString("ok")
            return True

        if command == "reset_cooldown":
            if cmd.size() < 3:
                reply.addString("error")
                return True
            fid = cmd.get(1).asString()
            tid = cmd.get(2).asInt32()
            cd_key = self._cooldown_key(fid, tid)
            self.last_interaction_time[cd_key] = time.time()
            reply.addString("ok")
            return True

        reply.addString("error")
        reply.addString(f"unknown command: {command}")
        return True

    def _send_facetracker_cmd(self, command: str, retries: int = 1):
        """Send a command (run/sus) to faceTracker, retrying if it's not up yet."""
        if not self.facetracker_rpc:
            return

        for attempt in range(retries):
            # Check if connected. If not, try to connect.
            if self.facetracker_rpc.getOutputCount() == 0:
                yarp.Network.connect(
                    f"/{self.module_name}/faceTracker:rpc", "/faceTracker"
                )

            if self.facetracker_rpc.getOutputCount() > 0:
                cmd = yarp.Bottle()
                cmd.addString(command)
                reply = yarp.Bottle()
                self.facetracker_rpc.write(cmd, reply)

                # FaceTracker.cpp replies with VOCAB_OK or VOCAB_FAILED
                if reply.size() > 0 and reply.get(0).asVocab32() == yarp.Vocab32_encode(
                    "ok"
                ):
                    self._log("INFO", f"Sent '{command}' to /faceTracker successfully")
                    return
                else:
                    self._log(
                        "WARNING",
                        f"'/faceTracker' replied unexpectedly to '{command}': {reply.toString()}",
                    )
                    return
            else:
                if attempt < retries - 1:
                    time.sleep(1.0)
                else:
                    self._log(
                        "WARNING",
                        f"Could not connect to /faceTracker to send '{command}'",
                    )

    def getPeriod(self) -> float:
        return self.period

    # ------------------------------------------------------------------ updateModule
    def updateModule(self) -> bool:
        if not self._running:
            return False

        try:
            landmarks_connected = self.landmarks_port.getInputCount() > 0

            if not landmarks_connected:
                if not self.ports_connected_logged:
                    self._log("INFO", "wait: landmarks")
                    self.ports_connected_logged = True
                return True

            if self.ports_connected_logged:
                self._log("INFO", "stream: on")
                self.ports_connected_logged = False

            today = self._get_today_date()
            if self._current_day != today:
                self._reload_memory_from_disk_and_prune_today()
                self._enqueue_save("greeted")
                self._enqueue_save("talked")
                self._current_day = today

            if self.stm_context_port is not None:
                if (
                    not self._context_connected_logged
                    and self.stm_context_port.getInputCount() > 0
                ):
                    self._log("INFO", "context: connected")
                    self._context_connected_logged = True
                ctx_btl = self.stm_context_port.read(False)
                if ctx_btl is not None:
                    self.current_context_label = ctx_btl.get(2).asInt8()

            faces = self._read_landmarks()

            with self.state_lock:
                self.current_faces = self._compute_face_states(faces)

            current_time = time.time()
            pending_exec_check = None
            command_target_for_io = None
            with self._interaction_lock:
                with self.state_lock:
                    # State invariants:
                    # - interaction_busy transitions are serialized by interaction_lock.
                    # - current_faces/selected_target snapshots are protected by state_lock.
                    # - rpc_override_track_id has priority over IPS arbitration.
                    # - while interaction_busy, target lock follows selected track when visible.

                    # --- A. ATTENTION LAYER (Who to look at) ---
                    best_face = self._select_best_face(self.current_faces)

                    # RPC override: executiveControl can force gaze via set_track_id
                    override_tid = self.rpc_override_track_id
                    if override_tid >= 0:
                        override_face = next(
                            (
                                f
                                for f in self.current_faces
                                if f.get("track_id") == override_tid
                            ),
                            None,
                        )
                        if override_face is not None:
                            target_to_look_at = override_face
                        else:
                            target_to_look_at = {
                                "track_id": override_tid,
                                "ips": 0.0,
                            }
                    elif self.interaction_busy and self.selected_target:
                        active_track_id = self.selected_target.get("track_id")
                        active_face = next(
                            (
                                f
                                for f in self.current_faces
                                if f.get("track_id") == active_track_id
                            ),
                            None,
                        )
                        if active_face is not None:
                            self.selected_target = active_face
                            target_to_look_at = active_face
                        else:
                            target_to_look_at = dict(self.selected_target)
                    else:
                        target_to_look_at = best_face

                    should_track = False
                    if target_to_look_at is not None:
                        if override_tid >= 0:
                            should_track = True
                        elif self.interaction_busy:
                            should_track = True
                        else:
                            target_ips = float(target_to_look_at.get("ips", 0.0))
                            should_track = target_ips >= self.min_track_ips

                    command_target = target_to_look_at if should_track else None

                    # Send I/O outside locks.
                    command_target_for_io = (
                        dict(command_target)
                        if isinstance(command_target, dict)
                        else None
                    )

                    # --- B. DIALOGUE LAYER (Who to interact with) ---
                    if (
                        not self.interaction_busy
                        and override_tid < 0
                        and target_to_look_at
                    ):

                        candidate = target_to_look_at
                        face_id = candidate.get("face_id", "unknown")
                        track_id = candidate.get("track_id", -1)
                        person_id = candidate.get("person_id", face_id)

                        resolved = self._is_face_id_resolved(face_id)

                        if resolved:
                            exec_face_id = (
                                str(person_id)
                                if self._is_face_known(str(person_id))
                                else str(face_id)
                            )

                            if not self._is_face_known(exec_face_id):
                                candidate["social_state"] = "ss1"
                                candidate["eligible"] = (
                                    float(candidate.get("ips", 0.0))
                                    >= self.SS_THRESHOLDS["ss1"]
                                )

                            cd_key = self._cooldown_key(exec_face_id, track_id)
                            last_int = self.last_interaction_time.get(cd_key, 0)

                            if current_time - last_int >= self._effective_cooldown():

                                eligible = candidate.get("eligible", False)
                                ss = candidate.get("social_state", "ss1")
                                area = self._bbox_area(candidate)

                                if self._should_log_target_selection(candidate):
                                    self._db_log(
                                        "target_selection",
                                        {
                                            "track_id": track_id,
                                            "face_id": face_id,
                                            "person_id": person_id,
                                            "bbox_area": area,
                                            "ips": float(candidate.get("ips", 0.0)),
                                            "ss": ss,
                                            "eligible": eligible,
                                            "context_label": self.current_context_label,
                                            "reason": "candidate_ready",
                                            "last_greeted_ts": candidate.get(
                                                "last_greeted_ts"
                                            ),
                                        },
                                    )

                                if eligible and ss != "ss4":
                                    can_try_exec = True

                                    if current_time < self._next_exec_rpc_try_ts:
                                        can_try_exec = False
                                    elif (
                                        self.executive_control_rpc is None
                                        or self.executive_control_rpc.getOutputCount() == 0
                                    ):
                                        self._next_exec_rpc_try_ts = current_time + self.exec_rpc_retry_sec
                                        if not self._exec_rpc_offline_logged:
                                            self._log("INFO", "exec: offline, skip")
                                            self._exec_rpc_offline_logged = True
                                        can_try_exec = False
                                    elif self._exec_rpc_offline_logged:
                                        self._log("INFO", "exec: online")
                                        self._exec_rpc_offline_logged = False

                                    if can_try_exec:
                                        pending_exec_check = {
                                            "track_id": track_id,
                                            "person_id": person_id,
                                            "cd_key": cd_key,
                                            "exec_face_id": exec_face_id,
                                        }

            self._send_target_to_facetracker(command_target_for_io)
            self._log_status_tick(command_target_for_io)

            if pending_exec_check is not None:
                im_status = self._executive_control_status()
                if not isinstance(im_status, dict):
                    self._next_exec_rpc_try_ts = current_time + self.exec_rpc_retry_sec
                elif not im_status.get("busy", False):
                    with self._interaction_lock:
                        with self.state_lock:
                            if not self.interaction_busy:
                                track_id = pending_exec_check["track_id"]
                                person_id = pending_exec_check["person_id"]
                                cd_key = pending_exec_check["cd_key"]
                                candidate = next(
                                    (
                                        f
                                        for f in self.current_faces
                                        if f.get("track_id") == track_id
                                    ),
                                    None,
                                )
                                if candidate is not None:
                                    now2 = time.time()
                                    last_int = self.last_interaction_time.get(cd_key, 0)
                                    if (
                                        now2 - last_int >= self._effective_cooldown()
                                        and candidate.get("eligible", False)
                                        and candidate.get("social_state", "ss1") != "ss4"
                                    ):
                                        exec_face_id = pending_exec_check.get(
                                            "exec_face_id",
                                            str(
                                                candidate.get(
                                                    "person_id",
                                                    candidate.get("face_id", "unknown"),
                                                )
                                            ),
                                        )
                                        if not self._is_face_known(exec_face_id):
                                            candidate["social_state"] = "ss1"
                                        candidate["exec_face_id"] = exec_face_id

                                        self.selected_target = candidate
                                        self.selected_bbox_last = candidate["bbox"]
                                        self.interaction_busy = True

                                        ss = candidate.get("social_state", "ss1")
                                        ips_val = float(candidate.get("ips", 0.0))
                                        self._log(
                                            "INFO",
                                            f"try: rpc t{track_id} {person_id} {ss} ips={ips_val:.2f}",
                                        )
                                        self._log(
                                            "INFO",
                                            f"pick: t{track_id} {person_id} {ss} ips={ips_val:.2f}",
                                        )

                                        self.interaction_thread = threading.Thread(
                                            target=self._run_interaction_thread,
                                            args=(dict(candidate),),
                                            daemon=True,
                                        )
                                        self.interaction_thread.start()



            self._publish_debug()
            self._consecutive_errors = 0
            return True

        except Exception as e:
            self._consecutive_errors += 1
            self._log("ERROR", f"loop err: {e}")
            if self._consecutive_errors >= self._max_consecutive_errors:
                return False
            return True

    def _log_status_tick(self, target_face: Optional[Dict[str, Any]]):
        target_tid = target_face.get("track_id", -1) if target_face else -1
        target_id = (
            target_face.get("person_id", target_face.get("face_id", "-"))
            if target_face
            else "-"
        )
        target_ss = target_face.get("social_state", "-") if target_face else "-"

        key = (int(target_tid), str(target_id), str(target_ss))
        if key == self._last_target_key:
            return

        face_count = len(self.current_faces)
        busy = "busy" if self.interaction_busy else "idle"
        ips = float(target_face.get("ips", 0.0)) if target_face else 0.0
        ov = self.rpc_override_track_id
        override_str = f" ov=t{ov}" if ov >= 0 else ""

        line = f"state: {busy} faces={face_count} look=t{target_tid} id={target_id} {target_ss} ips={ips:.2f}{override_str}"
        self._log("INFO", line)
        self._last_status_line = line
        self._last_target_key = key

    def _should_log_target_selection(self, face: Dict[str, Any]) -> bool:
        now = time.time()
        track_id = int(face.get("track_id", -1))
        person_id = str(face.get("person_id", face.get("face_id", "unknown")))
        ss = str(face.get("social_state", "ss1"))
        eligible = 1 if bool(face.get("eligible", False)) else 0
        ips = float(face.get("ips", 0.0))

        key = (track_id, person_id, ss, eligible)
        if key != self._last_target_log_key:
            self._last_target_log_key = key
            self._last_target_log_ips = ips
            self._last_target_log_ts = now
            return True

        if abs(ips - self._last_target_log_ips) >= self.TARGET_LOG_IPS_DELTA:
            self._last_target_log_ips = ips
            self._last_target_log_ts = now
            return True

        if now - self._last_target_log_ts >= self.TARGET_LOG_MIN_PERIOD_SEC:
            self._last_target_log_ips = ips
            self._last_target_log_ts = now
            return True

        return False

    # ==================== Context-Aware Cooldown ====================

    def _effective_cooldown(self) -> float:
        """Return the interaction cooldown based on the latest STM context label.

        label == 1  → lively/active scene  → shorter cooldown (more interactions)
        label == 0  → calm/quiet scene     → longer  cooldown (fewer interactions)
        label == -1 → not yet received or noise cluster → default cooldown
        """
        if self.current_context_label == 1:
            return self.cooldown_lively
        elif self.current_context_label == 0:
            return self.cooldown_calm
        return self.cooldown_default

    # ==================== Landmark Parsing ====================

    def _read_landmarks(self) -> List[Dict[str, Any]]:
        faces = []
        bottle = self.landmarks_port.read(False)
        if not bottle:
            if (
                self._latest_landmarks_ts > 0
                and (time.time() - self._latest_landmarks_ts) <= self.landmarks_stale_sec
            ):
                return list(self._latest_landmarks)
            return faces
        for i in range(bottle.size()):
            face_btl = bottle.get(i)
            if not face_btl.isList():
                continue
            face_data = self._parse_face_bottle(face_btl.asList())
            if face_data:
                faces.append(face_data)

        # Cache even empty parsed frames so real "no-face" observations can propagate.
        self._latest_landmarks = list(faces)
        self._latest_landmarks_ts = time.time()
        return faces

    def _parse_face_bottle(self, bottle: yarp.Bottle) -> Optional[Dict[str, Any]]:
        if not bottle:
            return None
        data = asdict(self.FaceSnapshot())
        try:
            i = 0
            while i < bottle.size():
                item = bottle.get(i)
                if item.isString():
                    key = item.asString()
                    if i + 1 < bottle.size():
                        nxt = bottle.get(i + 1)
                        if key == "face_id" and nxt.isString():
                            data["face_id"] = nxt.asString()
                            i += 2
                        elif key == "track_id" and (nxt.isInt32() or nxt.isInt64()):
                            data["track_id"] = nxt.asInt32()
                            i += 2
                        elif key in ("distance", "attention") and nxt.isString():
                            data[key] = nxt.asString()
                            i += 2
                        elif (
                            key in ("pitch", "yaw", "roll", "cos_angle", "time_in_view")
                            and nxt.isFloat64()
                        ):
                            data[key] = nxt.asFloat64()
                            i += 2
                        elif key == "is_talking" and (nxt.isInt32() or nxt.isInt64()):
                            data["is_talking"] = nxt.asInt32()
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                elif item.isList():
                    nested = item.asList()
                    if nested.size() >= 2:
                        key = (
                            nested.get(0).asString() if nested.get(0).isString() else ""
                        )
                        if key == "bbox" and nested.size() >= 5:
                            data["bbox"] = (
                                nested.get(1).asFloat64(),
                                nested.get(2).asFloat64(),
                                nested.get(3).asFloat64(),
                                nested.get(4).asFloat64(),
                            )
                        elif key == "gaze_direction" and nested.size() >= 4:
                            data["gaze_direction"] = (
                                nested.get(1).asFloat64(),
                                nested.get(2).asFloat64(),
                                nested.get(3).asFloat64(),
                            )
                    i += 1
                else:
                    i += 1
            return data
        except Exception as e:
            self._log("WARNING", f"Failed to parse face bottle: {e}")
            return None



    # ==================== State Computation ====================

    def _compute_face_states(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        today = self._get_today_date()
        new_area_history = {}

        for face in faces:
            face_id = face["face_id"]
            track_id = face["track_id"]
            person_id = self.track_to_person.get(track_id, face_id) or face_id
            face["person_id"] = person_id

            is_known = self._is_face_known(face_id) or self._is_face_known(person_id)
            face["is_known"] = is_known

            lg_entry = self.get_last_greeted_entry(str(person_id), track_id=track_id)
            face["last_greeted_info"] = lg_entry
            face["last_greeted_ts"] = lg_entry.get("timestamp") if lg_entry else None

            greeted_today = (
                self._was_greeted_today(person_id, today) if is_known else False
            )
            talked_today = (
                self._was_talked_today(person_id, today) if is_known else False
            )
            face["greeted_today"] = greeted_today
            face["talked_today"] = talked_today

            face["social_state"] = self._compute_social_state(
                is_known, greeted_today, talked_today
            )
            face["ips"] = self._calculate_ips(face, person_id)
            face["eligible"] = self._is_eligible(face)

            new_area_history[track_id] = self._bbox_area(face)

        active = {f["track_id"] for f in faces if f.get("track_id", -1) >= 0}
        self.track_to_person = {
            t: p for t, p in self.track_to_person.items() if t in active
        }
        self.area_history = new_area_history
        return faces

    def _is_face_known(self, face_id: str) -> bool:
        if not face_id:
            return False
        norm = str(face_id).strip().lower()
        if norm in ("unknown", "unmatched", "recognizing", ""):
            return False
        if norm.startswith("unknown:"):
            return False
        if norm.isdigit():
            return False
        return True

    def _cooldown_key(self, person_id: str, track_id: int) -> str:
        pid = str(person_id or "").strip()
        return pid if self._is_face_known(pid) else f"unknown:{track_id}"

    def _was_greeted_today(self, person_id: str, today: date) -> bool:
        if person_id not in self.greeted_today:
            return False
        try:
            dt = datetime.fromisoformat(self.greeted_today[person_id])
            return dt.astimezone(self.TIMEZONE).date() == today
        except Exception:
            return False

    def _was_talked_today(self, person_id: str, today: date) -> bool:
        if person_id not in self.talked_today:
            return False
        try:
            dt = datetime.fromisoformat(self.talked_today[person_id])
            return dt.astimezone(self.TIMEZONE).date() == today
        except Exception:
            return False

    def _compute_social_state(
        self, is_known: bool, greeted_today: bool, talked_today: bool
    ) -> str:
        """New 4-state model. Unknown people are always ss1."""
        if not is_known:
            return "ss1"
        if not greeted_today:
            return "ss2"
        if not talked_today:
            return "ss3"
        return "ss4"

    def _is_eligible(self, face: Dict[str, Any]) -> bool:
        ss = face.get("social_state", "ss1")
        ips = face.get("ips", 0.0)

        if ss == "ss4":
            return False

        threshold = self.SS_THRESHOLDS.get(ss, 1.0)
        return ips >= threshold

    # ==================== IPS & Habituation Math ====================

    def _get_person_weights(self, person_id: str) -> Dict[str, float]:
        """Fetch personalized weights, or fallback to baseline for strangers."""
        if (
            person_id in self.learning_data
            and "weights" in self.learning_data[person_id]
        ):
            return self.learning_data[person_id]["weights"]
        return self.BASELINE_WEIGHTS.copy()

    def _calculate_ips_variables(self, face: Dict[str, Any]) -> Dict[str, float]:
        """Converts raw landmark data into normalized 0.0-1.0 scoring variables."""
        IMG_W, IMG_H = 640.0, 480.0
        MAX_AREA = IMG_W * IMG_H
        MAX_DIST = math.hypot(IMG_W / 2, IMG_H / 2)
        VEL_SENSITIVITY = 10.0

        x, y, w, h = face.get("bbox", (0, 0, 0, 0))
        track_id = face.get("track_id", -1)
        cos_angle = face.get("cos_angle", 0.0)

        # 1. Proximity (Face Height Ratio)
        s_prox = min(1.0, h / IMG_H) if IMG_H > 0 else 0.0

        # 2. Centricity (Distance to center)
        cx, cy = x + (w / 2), y + (h / 2)
        dist_to_center = math.hypot(cx - (IMG_W / 2), cy - (IMG_H / 2))
        s_cent = max(0.0, 1.0 - (dist_to_center / MAX_DIST)) if MAX_DIST > 0 else 0.0

        # 3. Approach Velocity (Change in area)
        current_area = w * h
        prev_area = self.area_history.get(track_id, current_area)
        raw_vel = (current_area - prev_area) / MAX_AREA if MAX_AREA > 0 else 0.0
        s_vel = min(1.0, max(0.0, raw_vel * VEL_SENSITIVITY))

        # 4. Gaze
        s_gaze = max(0.0, cos_angle)

        return {"prox": s_prox, "cent": s_cent, "vel": s_vel, "gaze": s_gaze}

    def _calculate_ips(self, face: Dict[str, Any], person_id: str) -> float:
        """Calculates final IPS using personal weights and Habituation Decay."""
        vars_norm = self._calculate_ips_variables(face)
        weights = self._get_person_weights(person_id)

        # Base Formula
        base_ips = (
            weights["prox"] * vars_norm["prox"]
            + weights["cent"] * vars_norm["cent"]
            + weights["vel"] * vars_norm["vel"]
            + weights["gaze"] * vars_norm["gaze"]
        )

        # Habituation (e^-lambda * t_idle)
        time_in_view = face.get("time_in_view", 0.0)
        cd_key = self._cooldown_key(person_id, face.get("track_id", -1))
        last_int_time = self.last_interaction_time.get(cd_key, 0)

        time_since_int = (
            time.time() - last_int_time if last_int_time > 0 else float("inf")
        )
        t_idle = min(time_in_view, time_since_int)

        habituation_multiplier = math.exp(-self.HABITUATION_LAMBDA * t_idle)
        ips = base_ips * habituation_multiplier

        # Hysteresis Bonus
        if (
            face.get("track_id", -1) != -1
            and face.get("track_id") == self.current_target_track_id
        ):
            ips += self.IPS_HYSTERESIS_BONUS

        return ips

    # ==================== Face Selection (Best IPS) ====================

    @staticmethod
    def _is_face_id_resolved(face_id: str) -> bool:
        """A face_id is resolved when it's no longer recognizing/unmatched/empty."""
        if not face_id:
            return False
        return face_id.lower() not in ("recognizing", "unmatched", "")

    @staticmethod
    def _bbox_area(face: Dict[str, Any]) -> float:
        _, _, w, h = face.get("bbox", (0, 0, 0, 0))
        return w * h

    def _select_best_face(
        self, faces: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select face with highest IPS score."""
        if not faces:
            self.current_target_track_id = -1
            return None

        best = max(faces, key=lambda f: f.get("ips", 0.0))
        if self._bbox_area(best) <= 0:
            self.current_target_track_id = -1
            return None

        self.current_target_track_id = best.get("track_id", -1)
        return best

    # ==================== Target Command Streaming ====================

    def _send_target_to_facetracker(self, face: Optional[Dict[str, Any]]):
        """Send target command [track_id, ips] to vision."""
        if self.vision_cmd_port.getOutputCount() == 0:
            return

        if face is None:
            track_id = -1
            ips = 0.0
        else:
            track_id = int(face.get("track_id", -1))
            ips = float(face.get("ips", 0.0))

        # Event-driven output: only on start/stop/switch.
        if track_id == self._last_sent_track_id:
            return

        bottle = self.vision_cmd_port.prepare()
        bottle.clear()
        bottle.addInt32(track_id)
        bottle.addFloat64(ips)
        self.vision_cmd_port.write()

        self._last_sent_track_id = track_id
        self._last_sent_ips = ips

        if track_id >= 0:
            self._log("INFO", f"track: start t{track_id} ips={ips:.2f}")
        else:
            self._log("INFO", "track: stop")

    # ==================== Interaction Execution ====================

    def _run_interaction_thread(self, target: Dict[str, Any]):
        did_run_interaction = False
        started_at = time.time()
        attempt_id = uuid.uuid4().hex
        result: Optional[Dict] = None
        track_id = target.get("track_id", -1)
        raw_face_id = str(target.get("face_id", "unknown"))
        face_id = str(
            target.get(
                "exec_face_id",
                target.get("person_id", target.get("face_id", "unknown")),
            )
        )
        ss = target.get("social_state", "ss1")
        try:
            self._log("INFO", f"talk: start t{track_id} {face_id} {ss}")

            result = self._run_executive_control(track_id, face_id, ss)
            did_run_interaction = isinstance(result, dict)
            if result:
                self._log("INFO", f"talk: done ok={bool(result.get('success'))}")
                self._process_interaction_result(result, target)
            else:
                self._log("WARNING", "talk: no result")

        except Exception as e:
            self._log("ERROR", f"talk err: {e}")
        finally:
            duration_sec = max(0.0, time.time() - started_at)
            final_state = result.get("final_state") if isinstance(result, dict) else None
            abort_reason = result.get("abort_reason") if isinstance(result, dict) else None
            exec_interaction_id = (
                result.get("interaction_id") if isinstance(result, dict) else None
            )
            attempt = self.InteractionAttempt(
                attempt_id=attempt_id,
                track_id=track_id,
                face_id=raw_face_id,
                person_id=face_id,
                start_ss=ss,
                success=int(bool(result and result.get("success"))),
                final_state=final_state,
                abort_reason=abort_reason,
                exec_interaction_id=exec_interaction_id,
                duration_sec=duration_sec,
            )
            self._db_log("interaction_attempt", asdict(attempt))
            with self._interaction_lock:
                with self.state_lock:
                    self.interaction_busy = False
                    self.selected_target = None
                    self.selected_bbox_last = None
                    final_id = str(
                        self.track_to_person.get(
                            target.get("track_id", -1), target.get("face_id", "unknown")
                        )
                    )
                    cd_key = self._cooldown_key(final_id, target.get("track_id", -1))
                    if did_run_interaction:
                        self.last_interaction_time[cd_key] = time.time()
            self._log("INFO", "talk: end")

    def _prewarm_rpc_connections(self):
        time.sleep(1.0)
        for attempt in range(5):
            try:
                if (
                    self.executive_control_rpc
                    and self.executive_control_rpc.getOutputCount() > 0
                ):
                    cmd = yarp.Bottle()
                    cmd.addString("status")
                    reply = yarp.Bottle()
                    self.executive_control_rpc.write(cmd, reply)
                    break
            except Exception:
                time.sleep(2.0)

    def _run_executive_control(
        self, track_id: int, face_id: str, start_state: str
    ) -> Optional[Dict]:
        try:
            if self.executive_control_rpc.getOutputCount() == 0:
                self._log("WARNING", "exec: not connected")
                return None
            cmd = yarp.Bottle()
            cmd.addString("run")
            cmd.addInt32(track_id)
            cmd.addString(face_id)
            cmd.addString(start_state)
            reply = yarp.Bottle()
            self._log("INFO", "exec: run")
            if self.executive_control_rpc.write(cmd, reply):
                if reply.size() >= 2:
                    status = reply.get(0).asString()
                    json_str = reply.get(1).asString()
                    if status == "ok":
                        try:
                            parsed = json.loads(json_str)
                            if (
                                isinstance(parsed, dict)
                                and parsed.get("error")
                                == "responsive_interaction_running"
                            ):
                                self._log(
                                    "INFO",
                                    "exec: busy, skip",
                                )
                                return None
                            return parsed
                        except json.JSONDecodeError as e:
                            self._log("ERROR", f"exec json err: {e}")
            return None
        except Exception as e:
            self._log("ERROR", f"exec rpc err: {e}")
            return None

    def _executive_control_status(self) -> Optional[Dict]:
        try:
            if self.executive_control_rpc.getOutputCount() == 0:
                return None
            cmd = yarp.Bottle()
            cmd.addString("status")
            reply = yarp.Bottle()
            if not self.executive_control_rpc.write(cmd, reply):
                return None
            if reply.size() < 2:
                return None
            status = reply.get(0).asString()
            payload = reply.get(1).asString()
            if status != "ok":
                return None
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if isinstance(data, dict):
                return data
            return None
        except Exception as e:
            self._log("WARNING", f"executiveControl status check failed: {e}")
            return None

    def _process_interaction_result(self, result: Dict, target: Dict):
        try:
            extracted_name = (
                result.get("name") if result.get("name_extracted") else None
            )
            initial_ss = result.get("initial_state", "ss1")
            final_ss = result.get("final_state", initial_ss)

            greeted = final_ss in ("ss3", "ss4") and initial_ss != "ss4"
            talked = final_ss == "ss4" and initial_ss != "ss4"

            track_id = target["track_id"]
            person_id = str(
                target.get("person_id") or target.get("face_id") or "unknown"
            )

            if extracted_name:
                person_id = str(extracted_name)
                with self.state_lock:
                    self.last_interaction_time[f"unknown:{track_id}"] = time.time()
                    self.last_interaction_time[person_id] = time.time()

            now_iso = datetime.now(self.TIMEZONE).isoformat()

            with self._memory_lock:
                with self.state_lock:
                    self.track_to_person[track_id] = person_id
                    is_known_id = self._is_face_known(person_id)
                    if greeted and is_known_id:
                        self.greeted_today[person_id] = now_iso
                    if talked and is_known_id:
                        self.talked_today[person_id] = now_iso

            if greeted:
                self._enqueue_save("greeted")
            if talked:
                self._enqueue_save("talked")

            self._update_learning_weights(result, person_id)

            if initial_ss != final_ss:
                self._db_log(
                    "ss_change",
                    {
                        "person_id": person_id,
                        "old_ss": initial_ss,
                        "new_ss": final_ss,
                    },
                )

        except Exception as e:
            self._log("ERROR", f"process_interaction_result failed: {e}")

    def _update_learning_weights(self, result: Dict, person_id: str):
        """Continuously shifts a person's personality weights based on success."""
        if not self._is_face_known(person_id):
            return

        success = result.get("success", False)
        name_extracted = result.get("name_extracted", False)
        abort_reason = result.get("abort_reason")

        weights = self._get_person_weights(person_id)
        old_weights = dict(weights)
        lr = self.WEIGHT_SHIFT_RATE

        if success:
            shift = lr * (2.0 if name_extracted else 1.0)
            weights["prox"] = min(1.0, weights["prox"] + shift)
            weights["vel"] = min(1.0, weights["vel"] + shift)
            weights["gaze"] = max(0.0, weights["gaze"] - (shift * 0.5))
            outcome = "success"
            reason = "name_extracted" if name_extracted else "success"
            self._log(
                "INFO", f"Success! '{person_id}' weights shifted towards Proactive."
            )
        else:
            shift = lr * (2.0 if abort_reason == "face_disappeared" else 1.0)
            weights["prox"] = max(0.0, weights["prox"] - shift)
            weights["vel"] = max(0.0, weights["vel"] - shift)
            weights["gaze"] = min(1.0, weights["gaze"] + shift)
            outcome = "failure"
            reason = abort_reason or "failure"
            self._log(
                "INFO", f"Failure! '{person_id}' weights shifted towards Reactive."
            )

        now_iso = datetime.now(self.TIMEZONE).isoformat()
        with self._memory_lock:
            self.learning_data[person_id] = {"weights": weights, "updated_at": now_iso}

        self._enqueue_save("learning")

        delta = self.LearningDelta(
            person_id=person_id,
            reward_delta=shift if success else -shift,
            outcome=outcome,
            reason=reason,
            success=int(bool(success)),
            abort_reason=abort_reason,
            name_extracted=int(bool(name_extracted)),
            exec_interaction_id=result.get("interaction_id"),
            old_prox=old_weights.get("prox"),
            old_cent=old_weights.get("cent"),
            old_vel=old_weights.get("vel"),
            old_gaze=old_weights.get("gaze"),
            new_prox=weights.get("prox"),
            new_cent=weights.get("cent"),
            new_vel=weights.get("vel"),
            new_gaze=weights.get("gaze"),
        )
        self._db_log("learning_change", asdict(delta))

    # ==================== Last Greeted (fresh read) ====================

    def _refresh_last_greeted_snapshot(self):
        """Load last_greeted.json safely into snapshot."""
        try:
            if self.last_greeted_path.exists():
                with open(self.last_greeted_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._last_greeted_lock:
                    self._last_greeted_snapshot = self._normalize_last_greeted_snapshot(
                        data
                    )
            else:
                with self._last_greeted_lock:
                    self._last_greeted_snapshot = {}
        except Exception as e:
            self._log("WARNING", f"last_greeted refresh failed: {e}")

    def _last_greeted_refresh_loop(self):
        """Background loop refreshing last_greeted snapshot every 0.2s (5Hz)."""
        while self._running:
            self._refresh_last_greeted_snapshot()
            time.sleep(0.2)

    def _normalize_last_greeted_snapshot(self, data: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}

        if not isinstance(data, list):
            return {}

        latest_by_person: Dict[str, Dict[str, Any]] = {}
        latest_by_person_ts: Dict[str, float] = {}
        min_ts = float("-inf")

        for entry in data:
            if not isinstance(entry, dict):
                continue

            key = entry.get("assigned_code_or_name") or entry.get("face_id")
            if not key:
                track_id = entry.get("track_id")
                if track_id is not None:
                    key = f"unknown:{track_id}"
            if not key:
                continue

            entry_ts = entry.get("timestamp")
            try:
                ts = (
                    datetime.fromisoformat(entry_ts).timestamp() if entry_ts else min_ts
                )
            except Exception:
                ts = min_ts

            prev_ts = latest_by_person_ts.get(str(key), min_ts)
            if ts >= prev_ts:
                latest_by_person[str(key)] = entry
                latest_by_person_ts[str(key)] = ts

        return latest_by_person

    def get_last_greeted_entry(
        self, person_id: str, track_id: Optional[int] = None
    ) -> Optional[Dict]:
        """Get last greeted entry for a person from the snapshot."""
        with self._last_greeted_lock:
            entry = self._last_greeted_snapshot.get(person_id)
            if entry:
                return entry
            if track_id is not None:
                return self._last_greeted_snapshot.get(f"unknown:{track_id}")
        return None

    # ==================== JSON File Management ====================

    def _load_all_json_files(self):
        with self._memory_lock:
            self.greeted_today = self._load_json(self.greeted_path, {})
            self.talked_today = self._load_json(self.talked_path, {})
            learning_raw = self._load_json(self.learning_path, {"people": {}})
            self.learning_data = learning_raw.get("people", {})

            self.greeted_today = self._prune_to_today(self.greeted_today)
            self.talked_today = self._prune_to_today(self.talked_today)
        self._log(
            "INFO",
            f"Loaded – greeted:{len(self.greeted_today)} "
            f"talked:{len(self.talked_today)} "
            f"learning:{len(self.learning_data)}",
        )

    def _reload_memory_from_disk_and_prune_today(self):
        """Re-read memory JSON files from disk and prune greeted/talked to today.
        Called on day-change to pick up any writes that happened outside this process.
        """
        with self._memory_lock:
            self.greeted_today = self._load_json(self.greeted_path, {})
            self.talked_today = self._load_json(self.talked_path, {})
            learning_raw = self._load_json(self.learning_path, {"people": {}})
            self.learning_data = learning_raw.get("people", {})

            self.greeted_today = self._prune_to_today(self.greeted_today)
            self.talked_today = self._prune_to_today(self.talked_today)
        self._log(
            "INFO",
            f"Day-change reload – greeted:{len(self.greeted_today)} "
            f"talked:{len(self.talked_today)} "
            f"learning:{len(self.learning_data)}",
        )

    def _save_all_json_files(self):
        self._save_greeted_json()
        self._save_talked_json()
        self._save_learning_json()

    def _load_json(self, path: Path, default: Any) -> Any:
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self._log("WARNING", f"Failed to load {path}: {e}")
        return default

    def _save_json_atomic(self, path: Path, data: Any):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(suffix=".json", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, path)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except Exception as e:
            self._log("ERROR", f"Failed to save {path}: {e}")

    def _save_greeted_json(self):
        # Snapshot in-process state first (no I/O under _memory_lock).
        with self._memory_lock:
            in_memory = dict(self.greeted_today)
        # Cross-process exclusive lock: read → merge → write atomically so that
        # entries written by executiveControl's responsive path are never lost.
        lock_path = str(self.greeted_path) + ".lock"
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf, fcntl.LOCK_EX)
            try:
                on_disk = self._load_json(self.greeted_path, {})
                merged = {**(on_disk if isinstance(on_disk, dict) else {}), **in_memory}
                self._save_json_atomic(self.greeted_path, merged)
            finally:
                fcntl.flock(_lf, fcntl.LOCK_UN)

    def _save_talked_json(self):
        with self._memory_lock:
            data = dict(self.talked_today)
        self._save_json_atomic(self.talked_path, data)

    def _save_learning_json(self):
        with self._memory_lock:
            data = {"people": dict(self.learning_data)}
        self._save_json_atomic(self.learning_path, data)

    def _prune_to_today(self, d: Dict[str, str]) -> Dict[str, str]:
        today = self._get_today_date()
        out = {}
        for k, ts in d.items():
            try:
                dt = datetime.fromisoformat(ts).astimezone(self.TIMEZONE)
                if dt.date() == today:
                    out[k] = ts
            except Exception:
                pass
        return out

    # ---- Background I/O worker ----

    def _enqueue_save(self, kind: str):
        """Enqueue a JSON save task for the background worker."""
        self._queue_put_drop_oldest(self._io_queue, kind, f"IO queue ({kind})")

    def _io_worker(self):
        """Drain async JSON save queue."""
        while True:
            try:
                item = self._io_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue
            if item is None:
                break
            try:
                if item == "greeted":
                    self._save_greeted_json()
                elif item == "talked":
                    self._save_talked_json()
                elif item == "learning":
                    self._save_learning_json()
            except Exception as e:
                self._log("ERROR", f"IO worker save failed ({item}): {e}")

    # ==================== SQLite DB Logging ====================

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS target_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, track_id INTEGER, face_id TEXT,
                person_id TEXT, bbox_area REAL, ips REAL,
                ss TEXT, eligible INTEGER, context_label INTEGER, reason TEXT,
                last_greeted_ts TEXT
            )""")

            c.execute("""CREATE TABLE IF NOT EXISTS ss_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, person_id TEXT, old_ss TEXT, new_ss TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS learning_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                person_id TEXT,
                reward_delta REAL,
                outcome TEXT,
                reason TEXT,
                success INTEGER,
                abort_reason TEXT,
                name_extracted INTEGER,
                exec_interaction_id TEXT,
                old_prox REAL,
                old_cent REAL,
                old_vel REAL,
                old_gaze REAL,
                new_prox REAL,
                new_cent REAL,
                new_vel REAL,
                new_gaze REAL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS interaction_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                attempt_id TEXT,
                track_id INTEGER,
                face_id TEXT,
                person_id TEXT,
                start_ss TEXT,
                success INTEGER,
                final_state TEXT,
                abort_reason TEXT,
                exec_interaction_id TEXT,
                duration_sec REAL
            )""")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_target_selections_time ON target_selections(timestamp)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_target_selections_track ON target_selections(track_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_target_selections_person ON target_selections(person_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_attempts_time ON interaction_attempts(timestamp)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_attempts_exec_id ON interaction_attempts(exec_interaction_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_learning_changes_time ON learning_changes(timestamp)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_learning_changes_person ON learning_changes(person_id)"
            )

            self._ensure_learning_changes_columns(conn)
            conn.commit()
            conn.close()
            self._log("INFO", f"DB ready: {self.db_path}")
        except Exception as e:
            self._log("ERROR", f"DB init failed: {e}")

    def _ensure_learning_changes_columns(self, conn: sqlite3.Connection):
        """Add missing learning_changes columns for existing DBs."""
        try:
            rows = conn.execute("PRAGMA table_info(learning_changes)").fetchall()
            existing = {r[1] for r in rows}
            required = {
                "outcome": "TEXT",
                "reason": "TEXT",
                "success": "INTEGER",
                "abort_reason": "TEXT",
                "name_extracted": "INTEGER",
                "exec_interaction_id": "TEXT",
                "old_prox": "REAL",
                "old_cent": "REAL",
                "old_vel": "REAL",
                "old_gaze": "REAL",
                "new_prox": "REAL",
                "new_cent": "REAL",
                "new_vel": "REAL",
                "new_gaze": "REAL",
            }
            for col, typ in required.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE learning_changes ADD COLUMN {col} {typ}")
        except Exception as e:
            self._log("WARNING", f"learning_changes migration skipped: {e}")

    def _db_log(self, table: str, data: Dict):
        data["timestamp"] = datetime.now(self.TIMEZONE).isoformat()
        self._queue_put_drop_oldest(
            self._db_queue, (table, data), f"DB queue (table={table})"
        )

    def _queue_put_drop_oldest(self, q: queue.Queue, item: Any, label: str):
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            self._log("WARNING", f"{label} full, dropping oldest")
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            self._log("WARNING", f"{label} still full, dropping new item")

    def _open_db_connection(self, timeout: float) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout)
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
                    conn = self._open_db_connection(timeout=5.0)
                c = conn.cursor()
                if table == "target_selection":
                    c.execute(
                        "INSERT INTO target_selections (timestamp,track_id,face_id,person_id,bbox_area,ips,ss,eligible,context_label,reason,last_greeted_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            data["timestamp"],
                            data["track_id"],
                            data["face_id"],
                            data["person_id"],
                            data["bbox_area"],
                            data.get("ips"),
                            data["ss"],
                            int(data["eligible"]),
                            data.get("context_label"),
                            data.get("reason"),
                            data.get("last_greeted_ts"),
                        ),
                    )
                elif table == "ss_change":
                    c.execute(
                        "INSERT INTO ss_changes (timestamp,person_id,old_ss,new_ss) VALUES (?,?,?,?)",
                        (
                            data["timestamp"],
                            data["person_id"],
                            data["old_ss"],
                            data["new_ss"],
                        ),
                    )
                elif table == "learning_change":
                    c.execute(
                        """INSERT INTO learning_changes
                        (timestamp,person_id,reward_delta,outcome,reason,success,abort_reason,
                         name_extracted,exec_interaction_id,
                         old_prox,old_cent,old_vel,old_gaze,
                         new_prox,new_cent,new_vel,new_gaze)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            data["timestamp"],
                            data["person_id"],
                            data["reward_delta"],
                            data.get("outcome"),
                            data.get("reason"),
                            data.get("success"),
                            data.get("abort_reason"),
                            data.get("name_extracted"),
                            data.get("exec_interaction_id"),
                            data.get("old_prox"),
                            data.get("old_cent"),
                            data.get("old_vel"),
                            data.get("old_gaze"),
                            data.get("new_prox"),
                            data.get("new_cent"),
                            data.get("new_vel"),
                            data.get("new_gaze"),
                        ),
                    )
                elif table == "interaction_attempt":
                    c.execute(
                        "INSERT INTO interaction_attempts (timestamp,attempt_id,track_id,face_id,person_id,start_ss,success,final_state,abort_reason,exec_interaction_id,duration_sec) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            data["timestamp"],
                            data.get("attempt_id"),
                            data.get("track_id"),
                            data.get("face_id"),
                            data.get("person_id"),
                            data.get("start_ss"),
                            int(data.get("success", 0)),
                            data.get("final_state"),
                            data.get("abort_reason"),
                            data.get("exec_interaction_id"),
                            data.get("duration_sec"),
                        ),
                    )
                conn.commit()
            except Exception as e:
                self._log("ERROR", f"DB write failed ({table}): {e}")
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

    # ==================== Debug Output ====================

    def _publish_debug(self):
        if self.debug_port.getOutputCount() == 0:
            return
        try:
            btl = yarp.Bottle()
            btl.clear()
            btl.addString("status")
            btl.addString("busy" if self.interaction_busy else "idle")
            btl.addString("face_count")
            btl.addInt32(len(self.current_faces))
            with self.state_lock:
                if self.selected_target:
                    btl.addString("selected_face_id")
                    btl.addString(self.selected_target.get("face_id", "?"))
                    btl.addString("selected_track_id")
                    btl.addInt32(self.selected_target.get("track_id", -1))
            self.debug_port.write(btl)
        except Exception as e:
            self._log("WARNING", f"Failed to publish debug: {e}")

    # ==================== Utilities ====================

    def _get_today_date(self) -> date:
        return datetime.now(self.TIMEZONE).date()

    def _log(self, level: str, message: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{level}] {message}")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("[ERROR] YARP network not available – start yarpserver first.")
        sys.exit(1)

    module = SalienceNetworkModule()
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("alwaysOn")
    rf.configure(sys.argv)

    print("=" * 55)
    print(" SalienceNetworkModule – Adaptive IPS Face Selection")
    print("=" * 55)
    print(" ss1=unknown  ss2=known/not-greeted")
    print(" ss3=known/greeted/no-talk  ss4=ultimate")
    print()
    print(" yarp connect /alwayson/vision/landmarks:o /alwayson/salienceNetwork/landmarks:i")
    print(" yarp connect /alwayson/salienceNetwork/targetCmd:o /alwayson/vision/targetCmd:i")
    print("=" * 55)

    try:
        if not module.runModule(rf):
            print("[ERROR] Module failed to run.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        module.interruptModule()
        module.close()
        yarp.Network.fini()
        print("[INFO] Module closed.")
