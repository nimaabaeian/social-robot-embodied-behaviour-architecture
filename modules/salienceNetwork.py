# Copyright (c) 2026 Nima Abaeian
#
# Author: Nima Abaeian
# Organization: Istituto Italiano di Tecnologia
# Lab: Cognitive Architecture for Collaborative Technologies
# License: GNU GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""salienceNetwork.py - real-time face selection and interaction gating."""

import fcntl
import json
import math
import os
import queue
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    import yarp
except ImportError:
    print("[ERROR] YARP Python bindings required.")
    sys.exit(1)

_MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
_ALWAYSON_DIR = os.path.dirname(_MODULE_DIR)


class ExperimentMetadata:
    def __init__(self, *, log_cb=None):
        self._log = log_cb or (lambda level, msg: None)
        self.run_id = (os.getenv("ALWAYSON_RUN_ID") or "").strip() or uuid.uuid4().hex
        self.is_test_run = self._parse_bool(os.getenv("ALWAYSON_IS_TEST_RUN"), default=False)
        valid_env = os.getenv("ALWAYSON_VALID_FOR_ANALYSIS")
        self.valid_for_analysis = self._parse_bool(valid_env, default=not self.is_test_run)

    @staticmethod
    def _parse_bool(value: Optional[str], *, default: bool) -> bool:
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def experiment_fields(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "is_test_run": int(self.is_test_run),
            "valid_for_analysis": int(self.valid_for_analysis),
        }


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
        hunger_state: Optional[str] = None
        is_proactive: int = 1

    @dataclass(frozen=True)
    class HomeostaticLearningDelta:
        person_id: str
        reward_delta: float
        outcome: str
        reason: str
        trigger_mode: Optional[str]
        hunger_state_start: Optional[str]
        hunger_state_end: Optional[str]
        stomach_level_start: Optional[float]
        stomach_level_end: Optional[float]
        active_energy_cost: Optional[float]
        meals_eaten_count: Optional[int]
        n_turns: Optional[int]
        exec_interaction_id: Optional[str]
        affinity_before: Optional[float]
        affinity_after: Optional[float]

    # ==================== Adaptive IPS Constants ====================
    # Baseline IPS weights
    BASELINE_WEIGHTS = {"prox": 0.5, "cent": 0.15, "vel": 0.3, "gaze": 0.5}

    # Minimum IPS by social state
    SS_THRESHOLDS = {
        "ss1": 0.80, # Stranger: stricter hurdle (less proactive)
        "ss2": 0.65, # Known, not greeted: more proactive
        "ss3": 0.70, # Known, greeted, no talk: still proactive, but less than ss2
        "ss4": 0.85, # Known, talked: re-engage with ss3 action when highly salient
    }

    HABITUATION_LAMBDA = 0.06  # Habituation decay (~11.6 s half-life, only active after dwell)

    # Per-person affinity in [-1,+1]: an EMA of normalized reward. >0 feeds the drive
    # (be more proactive), <0 is an energy sink (be more conservative).
    AFFINITY_ALPHA = 0.25             # EMA weight on the newest interaction
    AFFINITY_REWARD_SCALE = 25.0      # reward mapped to |1.0| (~one medium meal)
    AFFINITY_NEUTRAL_BAND = 0.2       # |reward| within this carries no signal
    AFFINITY_THRESHOLD_GAIN = 0.15    # how far affinity shifts the eligibility threshold
    AFFINITY_THRESHOLD_FLOOR = 0.50   # floor so a feeder can't trip on noise
    ENGAGED_MIN_TURNS = 2             # turns that count as real engagement
    TARGET_LOG_MIN_PERIOD_SEC = 1.0
    TARGET_LOG_IPS_DELTA = 0.15
    FACE_IPS_LOG_PERIOD_SEC = 0.5
    FACE_IPS_LOG_DELTA = 0.03
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
        self._run_started_mono: float = time.monotonic()

        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

        self.executive_control_rpc_name = "/executiveControl"

        self.homeostatic_learning_path = Path(_MODULE_DIR) / "memory" / "homeostatic_learning.json"
        self.greeted_path      = Path(_MODULE_DIR) / "memory" / "greeted_today.json"
        self.talked_path       = Path(_MODULE_DIR) / "memory" / "talked_today.json"
        self.last_greeted_path = Path(_MODULE_DIR) / "memory" / "last_greeted.json"

        # YARP ports
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.debug_port: Optional[yarp.Port] = None
        self.vision_cmd_port: Optional[yarp.BufferedPortBottle] = None
        self.executive_control_rpc: Optional[yarp.RpcClient] = None
        self._exec_rpc_local_name: str = ""
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
        self.interaction_busy = False
        self.interaction_thread: Optional[threading.Thread] = None

        self.area_history: Dict[int, float] = {}  # Maps track_id to previous bbox area
        self.current_target_track_id: int = -1  # For applying Hysteresis
        self._attention_target_since: float = 0.0  # When current_target_track_id last changed
        self._target_decay_track_id: int = -1
        self._target_decay_started_at: float = 0.0
        self._recovery_tracks: Dict[int, Tuple[float, float]] = {}  # track_id → (start_time, multiplier_at_switch)

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
        self.track_stop_hysteresis: float = 0.1
        self.track_switch_hysteresis: float = 0.1
        self.min_gaze_dwell_sec: float = 5.0  # Min hold after a switch before another IPS switch (human-like dwell)
        self.habituation_competitor_margin: float = 0.2  # Habituate only when a competitor is within this IPS margin of the target
        self.track_stop_debounce_sec: float = 2.0
        self.exec_rpc_retry_sec: float = 1.0
        self.unknown_ss1_wait_sec: float = 5.0

        self.current_context_label: int = -1

        # Cache recent landmarks for sparse upstream publishing.
        self._latest_landmarks: List[Dict[str, Any]] = []
        self._latest_landmarks_ts: float = 0.0
        self.landmarks_stale_sec: float = 0.30

        self.greeted_today: Dict[str, str] = {}
        self.talked_today: Dict[str, str] = {}
        self.homeostatic_profiles: Dict[str, Dict] = {}

        self._last_greeted_snapshot: Dict[str, Dict[str, Any]] = {}
        self._last_greeted_lock = threading.Lock()

        self.track_to_person: Dict[int, str] = {}
        self._current_day: Optional[date] = None

        self.verbose_debug = False
        self.ports_connected_logged = False
        self.status_log_period_sec: float = 1.0
        self._last_target_key: Tuple[int, str, str] = (-2, "", "")
        self._last_sent_track_id: int = -99999
        self._last_sent_ips: float = -1.0
        self._last_sent_target_ts: float = 0.0
        self.target_cmd_keepalive_sec: float = 0.5
        self._vision_cmd_was_connected: bool = False
        self._next_exec_rpc_try_ts: float = 0.0
        self._exec_rpc_offline_logged: bool = False
        self._last_target_log_key: Tuple[int, str, str, int] = (-2, "", "", -1)
        self._last_target_log_ips: float = -1.0
        self._last_target_log_ts: float = 0.0
        self._last_face_ips_log: Dict[Tuple[int, str], Tuple[float, float, float]] = {}

        self._io_queue: queue.Queue = queue.Queue(maxsize=self.IO_QUEUE_MAXSIZE)
        self._io_thread: Optional[threading.Thread] = None

        self.experiment = ExperimentMetadata(log_cb=self._log)
        self.db_path = str(Path(_MODULE_DIR) / "data_collection" / "salience_network.db")
        self._db_queue: queue.Queue = queue.Queue()
        self._db_thread: Optional[threading.Thread] = None
        self._context_connected_logged = False
        self._unknown_ss1_since: Dict[int, float] = {}
        self._last_hunger_state: str = "HS0"
        self._active_interaction_attempt_id: Optional[str] = None
        self._active_interaction_track_id: int = -1
        self._active_interaction_person_id: str = ""

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
            if rf.check("homeostatic_learning_path"):
                self.homeostatic_learning_path = Path(
                    rf.find("homeostatic_learning_path").asString()
                )
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
            if rf.check("track_stop_hysteresis"):
                self.track_stop_hysteresis = max(
                    0.0, rf.find("track_stop_hysteresis").asFloat64()
                )
            if rf.check("track_switch_hysteresis"):
                self.track_switch_hysteresis = max(
                    0.0, rf.find("track_switch_hysteresis").asFloat64()
                )
            if rf.check("min_gaze_dwell_sec"):
                self.min_gaze_dwell_sec = max(
                    0.0, rf.find("min_gaze_dwell_sec").asFloat64()
                )
            if rf.check("habituation_competitor_margin"):
                self.habituation_competitor_margin = max(
                    0.0, rf.find("habituation_competitor_margin").asFloat64()
                )
            if rf.check("track_stop_debounce_sec"):
                self.track_stop_debounce_sec = max(
                    0.0, rf.find("track_stop_debounce_sec").asFloat64()
                )
            if rf.check("target_cmd_keepalive_sec"):
                self.target_cmd_keepalive_sec = max(
                    0.1, rf.find("target_cmd_keepalive_sec").asFloat64()
                )
            if rf.check("exec_rpc_retry_sec"):
                self.exec_rpc_retry_sec = max(
                    0.1, rf.find("exec_rpc_retry_sec").asFloat64()
                )
            if rf.check("unknown_ss1_wait_sec"):
                self.unknown_ss1_wait_sec = max(
                    0.0, rf.find("unknown_ss1_wait_sec").asFloat64()
                )

            # --- Open ports ---
            def _open_port(attr: str, cls, name: str) -> bool:
                port = cls()
                next_log_ts = 0.0
                while not port.open(name):
                    if not self._running:
                        try:
                            port.close()
                        except Exception:
                            pass
                        return False
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
                f"/alwayson/{self.module_name}/landmarks:i"):
                return False

            if not _open_port(
                "vision_cmd_port",
                yarp.BufferedPortBottle,
                f"/alwayson/{self.module_name}/targetCmd:o"):
                return False

            if not _open_port(
                "debug_port", yarp.Port, f"/alwayson/{self.module_name}/debug:o"
            ):
                return False

            handle_name = f"/{self.module_name}"
            handle_next_log_ts = 0.0
            while not self.handle_port.open(handle_name):
                if not self._running:
                    return False
                now = time.time()
                if now >= handle_next_log_ts:
                    self._log("INFO", f"Waiting local port availability: {handle_name}")
                    handle_next_log_ts = now + 2.0
                time.sleep(0.5)

            self.stm_context_port = yarp.BufferedPortBottle()
            _ctx_local = f"/alwayson/{self.module_name}/context:i"
            ctx_next_log_ts = 0.0
            while not self.stm_context_port.open(_ctx_local):
                if not self._running:
                    try:
                        self.stm_context_port.close()
                    except Exception:
                        pass
                    return False
                now = time.time()
                if now >= ctx_next_log_ts:
                    self._log("INFO", f"Waiting local port availability: {_ctx_local}")
                    ctx_next_log_ts = now + 2.0
                time.sleep(0.5)

            self.executive_control_rpc = yarp.RpcClient()
            exec_rpc_local = f"/{self.module_name}/executiveControl:rpc"
            self._exec_rpc_local_name = exec_rpc_local
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
                and self.landmarks_port.getInputCount() > 0),
            (
                f"/alwayson/{self.module_name}/targetCmd:o -> /alwayson/vision/targetCmd:i",
                lambda: self.vision_cmd_port is not None
                and self.vision_cmd_port.getOutputCount() > 0),
        ]

        pending = [name for name, ok in checks if not ok()]
        if pending:
            pending_labels = ", ".join(
                item.split(" -> ")[0].split("/")[-1] for item in pending
            )
            self._log(
                "INFO",
                f"Startup without blocking; pending required ports: {pending_labels}")
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
        if self.executive_control_rpc:
            self.executive_control_rpc.interrupt()
        if self.facetracker_rpc:
            self.facetracker_rpc.interrupt()
        if self.interaction_thread and self.interaction_thread.is_alive():
            self.interaction_thread.join(timeout=5.0)
        self._enqueue_save("greeted")
        self._enqueue_save("talked")
        self._enqueue_save("homeostatic_learning")
        self._queue_put_drop_oldest(self._io_queue, None, "IO queue close")
        if self._io_thread:
            self._io_thread.join(timeout=5.0)
        self._db_queue.put(None)
        if self._db_thread:
            self._db_thread.join(timeout=3.0)
        if self.facetracker_rpc and self._running:
            # Send 'sus' before shutting down, only try once so we don't delay close()
            self._send_facetracker_cmd("sus", retries=1)
        if self.executive_control_rpc:
            self.executive_control_rpc.close()
        if self.facetracker_rpc:
            self.facetracker_rpc.close()

        for port in [
            self.landmarks_port,
            self.debug_port,
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

        if command == "ambient_feed":
            if cmd.size() < 2:
                reply.addString("error")
                reply.addString("usage: ambient_feed <json_payload>")
                return True
            try:
                payload = json.loads(cmd.get(1).asString())
            except json.JSONDecodeError as e:
                reply.addString("error")
                reply.addString(f"invalid json: {e}")
                return True
            if not isinstance(payload, dict):
                reply.addString("error")
                reply.addString("payload must be a JSON object")
                return True
            person_id = self._resolve_attention_target_person()
            self._apply_homeostatic_learning(payload, person_id)
            reply.addString("ok")
            reply.addString(person_id or "unknown")
            return True

        if command == "interaction_result":
            if cmd.size() < 2:
                reply.addString("error")
                reply.addString("usage: interaction_result <json_payload>")
                return True
            try:
                payload = json.loads(cmd.get(1).asString())
            except json.JSONDecodeError as e:
                reply.addString("error")
                reply.addString(f"invalid json: {e}")
                return True
            if not isinstance(payload, dict):
                reply.addString("error")
                reply.addString("payload must be a JSON object")
                return True
            person_id = self._resolve_result_person_id(payload)
            self._process_external_interaction_result(payload, person_id)
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
                if reply.size() > 0 and reply.get(0).asVocab32() == yarp.createVocab32(
                    'o', 'k'
                ):
                    self._log("INFO", f"Sent '{command}' to /faceTracker successfully")
                    return
                else:
                    self._log(
                        "WARNING",
                        f"'/faceTracker' replied unexpectedly to '{command}': {reply.toString()}")
                    return
            else:
                if attempt < retries - 1:
                    time.sleep(1.0)
                else:
                    self._log(
                        "WARNING",
                        f"Could not connect to /faceTracker to send '{command}'")

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
                    override_tid = self.rpc_override_track_id
                    target_to_look_at = self._choose_attention_target(
                        self.current_faces, override_tid
                    )
                    command_target = self._hydrate_tracking_target(target_to_look_at)

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

                        if self._can_attempt_interaction(candidate, current_time):
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
                                        })

                                if eligible:
                                    can_try_exec = True

                                    if current_time < self._next_exec_rpc_try_ts:
                                        can_try_exec = False
                                    elif not self._ensure_exec_rpc_connected(log_on_fail=False):
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
                self._log(
                    "INFO",
                    (
                        "try: start_check "
                        f"t{pending_exec_check['track_id']} "
                        f"{pending_exec_check['person_id']}"
                    ))
                im_status = self._executive_control_status()
                if isinstance(im_status, dict):
                    hs_val = im_status.get("hunger_state")
                    if hs_val and str(hs_val).startswith("HS"):
                        self._last_hunger_state = str(hs_val)
                if not isinstance(im_status, dict):
                    self._log("INFO", "try: blocked status_unavailable")
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
                                    None)
                                if candidate is not None:
                                    now2 = time.time()
                                    last_int = self.last_interaction_time.get(cd_key, 0)
                                    if (
                                        now2 - last_int >= self._effective_cooldown()
                                        and candidate.get("eligible", False)
                                    ):
                                        exec_face_id = pending_exec_check.get(
                                            "exec_face_id",
                                            str(
                                                candidate.get(
                                                    "person_id",
                                                    candidate.get("face_id", "unknown"))
                                            ))
                                        if not self._is_face_known(exec_face_id):
                                            candidate["social_state"] = "ss1"
                                        candidate["exec_face_id"] = exec_face_id

                                        self.selected_target = candidate
                                        self.interaction_busy = True

                                        ss = candidate.get("social_state", "ss1")
                                        ips_val = float(candidate.get("ips", 0.0))
                                        self._log(
                                            "INFO",
                                            f"try: rpc t{track_id} {person_id} {ss} ips={ips_val:.2f}")
                                        self._log(
                                            "INFO",
                                            f"pick: t{track_id} {person_id} {ss} ips={ips_val:.2f}")

                                        self.interaction_thread = threading.Thread(
                                            target=self._run_interaction_thread,
                                            args=(dict(candidate),),
                                            daemon=True)
                                        self.interaction_thread.start()
                                    else:
                                        self._log(
                                            "INFO",
                                            (
                                                "try: blocked after_recheck "
                                                f"cooldown_ok={now2 - last_int >= self._effective_cooldown()} "
                                                f"eligible={bool(candidate.get('eligible', False))} "
                                                f"ss={candidate.get('social_state', 'ss1')}"
                                            ))
                                else:
                                    self._log(
                                        "INFO",
                                        f"try: blocked candidate_missing t{track_id}")
                            else:
                                self._log("INFO", "try: blocked salience_busy")
                else:
                    busy_mode = str(im_status.get("busy_mode", ""))
                    busy_reason = str(im_status.get("busy_reason", ""))
                    if busy_mode or busy_reason:
                        self._log(
                            "INFO",
                            f"try: blocked executive_busy mode={busy_mode} reason={busy_reason}")
                    else:
                        self._log("INFO", "try: blocked executive_busy")



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

    def _should_log_face_ips_event(self, face: Dict[str, Any], now: float) -> bool:
        track_id = int(face.get("track_id", -1))
        if track_id < 0:
            return False
        face_id = str(face.get("face_id", "unknown"))
        key = (track_id, face_id)
        ips = float(face.get("ips", 0.0))
        habituation_multiplier = float(face.get("habituation_multiplier", 1.0))
        last = self._last_face_ips_log.get(key)
        if last is None:
            self._last_face_ips_log[key] = (now, ips, habituation_multiplier)
            return True
        last_ts, last_ips, last_habituation = last
        if (
            now - last_ts >= self.FACE_IPS_LOG_PERIOD_SEC
            or abs(ips - last_ips) >= self.FACE_IPS_LOG_DELTA
            or abs(habituation_multiplier - last_habituation) >= self.FACE_IPS_LOG_DELTA
        ):
            self._last_face_ips_log[key] = (now, ips, habituation_multiplier)
            return True
        return False

    def _log_face_ips_events(self, faces: List[Dict[str, Any]]) -> None:
        now = time.time()
        active_track_id = self._active_attention_track_id()
        for face in faces:
            if not self._should_log_face_ips_event(face, now):
                continue
            person_id = str(face.get("person_id", face.get("face_id", "unknown")))
            vars_norm = self._calculate_ips_variables(face)
            weights = self.BASELINE_WEIGHTS
            self._db_log(
                "face_ips_event",
                {
                    "track_id": int(face.get("track_id", -1)),
                    "face_id": str(face.get("face_id", "unknown")),
                    "person_id": person_id,
                    "social_state": str(face.get("social_state", "ss1")),
                    "is_known": int(bool(face.get("is_known", False))),
                    "eligible": int(bool(face.get("eligible", False))),
                    "is_active_target": int(int(face.get("track_id", -1)) == active_track_id),
                    "bbox_area": self._bbox_area(face),
                    "ips": float(face.get("ips", 0.0)),
                    "ips_before_habituation": float(
                        face.get("ips_before_habituation", face.get("ips", 0.0))
                    ),
                    "habituation_applied": int(bool(face.get("habituation_applied", False))),
                    "habituation_multiplier": float(face.get("habituation_multiplier", 1.0)),
                    "habituation_elapsed_sec": float(face.get("habituation_elapsed_sec", 0.0)),
                    "habituation_ips_delta": float(face.get("habituation_ips_delta", 0.0)),
                    "stimulus_type": (
                        "habituation_decay"
                        if face.get("habituation_applied", False)
                        else "visual_salience"
                    ),
                    "context_label": self.current_context_label,
                    "prox_score": vars_norm["prox"],
                    "cent_score": vars_norm["cent"],
                    "vel_score": vars_norm["vel"],
                    "gaze_score": vars_norm["gaze"],
                    "weight_prox": weights["prox"],
                    "weight_cent": weights["cent"],
                    "weight_vel": weights["vel"],
                    "weight_gaze": weights["gaze"],
                })

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

        # Upstream multi-face association can transiently emit only unmatched faces
        # for one frame. Do not replace a fresh valid tracked snapshot with an
        # all-unmatched snapshot, or attention oscillates between target and none.
        has_valid_tracks = any(int(f.get("track_id", -1)) >= 0 for f in faces)
        prev_has_valid_tracks = any(
            int(f.get("track_id", -1)) >= 0 for f in self._latest_landmarks
        )
        if (
            faces
            and not has_valid_tracks
            and prev_has_valid_tracks
            and self._latest_landmarks_ts > 0
            and (time.time() - self._latest_landmarks_ts) <= self.landmarks_stale_sec
        ):
            return list(self._latest_landmarks)

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
                                nested.get(4).asFloat64())
                        elif key == "gaze_direction" and nested.size() >= 4:
                            data["gaze_direction"] = (
                                nested.get(1).asFloat64(),
                                nested.get(2).asFloat64(),
                                nested.get(3).asFloat64())
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
            face["ips"] = self._calculate_ips(face, person_id, apply_habituation=False)
            face["habituation_applied"] = False
            face["habituation_multiplier"] = 1.0
            face["habituation_elapsed_sec"] = 0.0
            face["ips_before_habituation"] = float(face.get("ips", 0.0))
            face["habituation_ips_delta"] = 0.0

            new_area_history[track_id] = self._bbox_area(face)

        # Apply habituation decay only when:
        # 1) no interaction is ongoing,
        # 2) there is more than one face visible.
        if (not self.interaction_busy) and len(faces) > 1:
            current_track_id = self._active_attention_track_id()
            current_face = next(
                (f for f in faces if int(f.get("track_id", -1)) == current_track_id),
                None)
            if current_face is not None:
                person_id = str(
                    current_face.get(
                        "person_id", current_face.get("face_id", "unknown")
                    )
                )
                before_ips = float(current_face.get("ips", 0.0))
                dwell_elapsed = time.time() - self._attention_target_since
                has_competitor = any(
                    int(f.get("track_id", -1)) != current_track_id
                    and float(f.get("ips", 0.0)) >= before_ips - self.habituation_competitor_margin
                    for f in faces
                )
                decay = (
                    self._habituation_multiplier(current_face, person_id)
                    if has_competitor and dwell_elapsed >= self.min_gaze_dwell_sec
                    else 1.0
                )
                after_ips = before_ips * decay
                current_face["habituation_applied"] = True
                current_face["habituation_multiplier"] = decay
                current_face["habituation_elapsed_sec"] = max(
                    0.0, time.time() - self._target_decay_started_at
                )
                current_face["ips_before_habituation"] = before_ips
                current_face["habituation_ips_delta"] = after_ips - before_ips
                current_face["ips"] = after_ips
            else:
                self._target_decay_track_id = -1
                self._target_decay_started_at = 0.0
        else:
            self._target_decay_track_id = -1
            self._target_decay_started_at = 0.0
            self._recovery_tracks.clear()

        active_track_id = self._active_attention_track_id()
        now = time.time()
        for face in faces:
            track_id = int(face.get("track_id", -1))
            if track_id == active_track_id:
                self._recovery_tracks.pop(track_id, None)
                continue
            recovery = self._recovery_tracks.get(track_id)
            if recovery is not None:
                rec_start, m0 = recovery
                elapsed = max(0.0, now - rec_start)
                rec_multiplier = 1.0 - (1.0 - m0) * math.exp(-self.HABITUATION_LAMBDA * elapsed)
                face["ips"] = float(face.get("ips", 0.0)) * rec_multiplier

        for face in faces:
            face["eligible"] = self._is_eligible(face)

        self._log_face_ips_events(faces)

        active = {f["track_id"] for f in faces if f.get("track_id", -1) >= 0}
        self.track_to_person = {
            t: p for t, p in self.track_to_person.items() if t in active
        }
        self._unknown_ss1_since = {
            t: ts for t, ts in self._unknown_ss1_since.items() if t in active
        }
        self._recovery_tracks = {
            t: rec for t, rec in self._recovery_tracks.items() if t in active
        }
        self._last_face_ips_log = {
            key: val for key, val in self._last_face_ips_log.items() if key[0] in active
        }
        self.area_history = new_area_history
        return faces

    def _can_attempt_interaction(self, face: Dict[str, Any], now_ts: float) -> bool:
        """Gate interaction start to allow identity resolution for unresolved ss1 faces."""
        track_id = int(face.get("track_id", -1))
        social_state = str(face.get("social_state", "ss1"))
        face_id = str(face.get("face_id", ""))

        resolved = self._is_face_id_resolved(face_id)
        if resolved or social_state != "ss1":
            if track_id >= 0:
                self._unknown_ss1_since.pop(track_id, None)
            return True

        if track_id < 0:
            return False

        first_seen = self._unknown_ss1_since.get(track_id)
        if first_seen is None:
            self._unknown_ss1_since[track_id] = now_ts
            return False

        return (now_ts - first_seen) >= self.unknown_ss1_wait_sec

    def _is_face_known(self, face_id: str) -> bool:
        if not face_id:
            return False
        norm = str(face_id).strip().lower()
        if norm in ("unknown", "unmatched", "recognizing", "unresolved", ""):
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
        person_id = str(face.get("person_id", face.get("face_id", "")))
        return ips >= self._effective_threshold(ss, person_id)

    def _effective_threshold(self, ss: str, person_id: str) -> float:
        """Eligibility threshold lowered for liked feeders, raised for energy sinks."""
        base = self.SS_THRESHOLDS.get(ss, 1.0)
        affinity = self._person_affinity(person_id)
        return max(self.AFFINITY_THRESHOLD_FLOOR, base - self.AFFINITY_THRESHOLD_GAIN * affinity)

    @staticmethod
    def _clamp_affinity(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _person_affinity(self, person_id: str) -> float:
        """Return a person's learned affinity in [-1, +1] (0 for strangers)."""
        profile = self.homeostatic_profiles.get(person_id, {})
        if isinstance(profile, dict) and isinstance(profile.get("affinity"), (int, float)):
            return self._clamp_affinity(profile["affinity"])
        return 0.0

    # ==================== IPS & Habituation Math ====================

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

    def _calculate_ips(
        self, face: Dict[str, Any], person_id: str, apply_habituation: bool = True
    ) -> float:
        """Calculates IPS from fixed perceptual weights and optional Habituation Decay.

        Personalization lives in the eligibility threshold (see _effective_threshold),
        not the IPS itself, so the perceptual score stays comparable across people.
        """
        vars_norm = self._calculate_ips_variables(face)
        weights = self.BASELINE_WEIGHTS

        # Base Formula
        base_ips = (
            weights["prox"] * vars_norm["prox"]
            + weights["cent"] * vars_norm["cent"]
            + weights["vel"] * vars_norm["vel"]
            + weights["gaze"] * vars_norm["gaze"]
        )

        ips = base_ips
        if apply_habituation:
            ips *= self._habituation_multiplier(face, person_id)

        return ips

    def _habituation_multiplier(self, face: Dict[str, Any], person_id: str) -> float:
        """Decay only the continuously attended target during multi-face idle periods."""
        track_id = int(face.get("track_id", -1))
        if track_id < 0:
            return 1.0

        now = time.time()
        if self._target_decay_track_id != track_id:
            old_track_id = self._target_decay_track_id
            if old_track_id >= 0 and self._target_decay_started_at > 0.0:
                elapsed = max(0.0, now - self._target_decay_started_at)
                self._recovery_tracks[old_track_id] = (
                    now, math.exp(-self.HABITUATION_LAMBDA * elapsed)
                )
            self._target_decay_track_id = track_id
            self._target_decay_started_at = now
            return 1.0

        elapsed = max(0.0, now - self._target_decay_started_at)
        return math.exp(-self.HABITUATION_LAMBDA * elapsed)

    def _set_attention_target(self, track_id: int) -> None:
        """Assign the attention target, timestamping any change for dwell gating."""
        if track_id != self.current_target_track_id:
            self._attention_target_since = time.time()
        self.current_target_track_id = track_id

    def _active_attention_track_id(self) -> int:
        """Return the track that is actually holding the robot's attention."""
        if self.interaction_busy and self.selected_target is not None:
            return int(self.selected_target.get("track_id", -1))
        if self._last_sent_track_id >= 0:
            return int(self._last_sent_track_id)
        return int(self.current_target_track_id)

    def _resolve_attention_target_person(self) -> str:
        """Resolve the person the robot is currently gaze-tracking.

        Used to credit ambient feeds: the feeder is always the tracked person.
        Returns a known person id when resolvable, else 'unknown' (which the
        homeostatic-learning path will log and skip).
        """
        track_id = self._active_attention_track_id()
        if track_id < 0:
            return "unknown"
        with self.state_lock:
            person_id = self.track_to_person.get(track_id, "")
        if not self._is_face_known(person_id):
            face = self._find_face_by_track_id(track_id)
            if face is not None:
                face_id = str(face.get("face_id", "")).strip()
                if self._is_face_known(face_id):
                    person_id = face_id
        return person_id if self._is_face_known(person_id) else "unknown"

    def _find_face_by_track_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        if track_id < 0:
            return None
        return next(
            (f for f in self.current_faces if int(f.get("track_id", -1)) == track_id),
            None)

    def _hydrate_tracking_target(
        self, target: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Prefer current full face state over synthetic track-only placeholders."""
        if target is None:
            return None

        hydrated = dict(target)
        track_id = int(hydrated.get("track_id", -1))
        live_face = self._find_face_by_track_id(track_id)
        if live_face is not None:
            merged = dict(hydrated)
            merged.update(live_face)
            return merged

        if (
            track_id >= 0
            and float(hydrated.get("ips", 0.0)) <= 0.0
            and track_id == self._last_sent_track_id
            and self._last_sent_ips > 0.0
        ):
            hydrated["ips"] = float(self._last_sent_ips)

        return hydrated

    def _visible_tracked_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            f
            for f in faces
            if int(f.get("track_id", -1)) >= 0
        ]

    def _choose_attention_target(
        self, faces: List[Dict[str, Any]], override_tid: int
    ) -> Optional[Dict[str, Any]]:
        """Simple tracking policy: decay current target, then track the highest visible IPS."""
        if override_tid >= 0:
            override_face = self._find_face_by_track_id(override_tid)
            self._set_attention_target(override_tid if override_face is not None else -1)
            return override_face if override_face is not None else {"track_id": override_tid, "ips": 0.0}

        if self.interaction_busy and self.selected_target:
            active_track_id = int(self.selected_target.get("track_id", -1))
            active_face = self._find_face_by_track_id(active_track_id)
            self._set_attention_target(active_track_id if active_face is not None else -1)
            return active_face if active_face is not None else dict(self.selected_target)

        visible_faces = self._visible_tracked_faces(faces)
        if not visible_faces:
            self._set_attention_target(-1)
            return None

        best_face = max(visible_faces, key=lambda f: float(f.get("ips", 0.0)))
        active_track_id = self._active_attention_track_id()
        active_face = self._find_face_by_track_id(active_track_id)
        if active_face is not None:
            active_ips = float(active_face.get("ips", 0.0))
            best_ips = float(best_face.get("ips", 0.0))
            best_track_id = int(best_face.get("track_id", -1))
            # Hold the current target if the challenger is within the hysteresis
            # margin, or if the minimum gaze dwell time has not yet elapsed. The
            # dwell gate bounds switch frequency directly, so habituation cannot
            # erode the margin into rapid oscillation between two faces.
            dwell_elapsed = time.time() - self._attention_target_since
            if (
                best_track_id != active_track_id
                and len(visible_faces) > 1
                and (
                    best_ips <= (active_ips + self.track_switch_hysteresis)
                    or dwell_elapsed < self.min_gaze_dwell_sec
                )
            ):
                self._set_attention_target(active_track_id)
                return active_face

            self._set_attention_target(int(best_face.get("track_id", -1)))
            return best_face

        if float(best_face.get("ips", 0.0)) < self.min_track_ips:
            self._set_attention_target(-1)
            return None

        self._set_attention_target(int(best_face.get("track_id", -1)))
        return best_face

    # ==================== Face Selection (Best IPS) ====================

    @staticmethod
    def _is_face_id_resolved(face_id: str) -> bool:
        """A face_id is resolved when it's no longer recognizing/unmatched/unknown/empty."""
        if not face_id:
            return False
        norm = face_id.lower()
        if norm.startswith("unknown:"):
            return False
        return norm not in ("recognizing", "unmatched", "unknown", "")

    @staticmethod
    def _bbox_area(face: Dict[str, Any]) -> float:
        _, _, w, h = face.get("bbox", (0, 0, 0, 0))
        return w * h

    # ==================== Target Command Streaming ====================

    def _send_target_to_facetracker(self, face: Optional[Dict[str, Any]]):
        """Send target command [track_id, ips] to vision."""
        now = time.time()
        if self.vision_cmd_port.getOutputCount() == 0:
            # If downstream reconnects later, force the next command to be resent.
            self._vision_cmd_was_connected = False
            self._last_sent_track_id = -99999
            self._last_sent_target_ts = 0.0
            return

        if not self._vision_cmd_was_connected:
            self._vision_cmd_was_connected = True
            self._last_sent_track_id = -99999
            self._last_sent_target_ts = 0.0

        if face is None:
            track_id = -1
            ips = 0.0
        else:
            track_id = int(face.get("track_id", -1))
            ips = float(face.get("ips", 0.0))

        # Event-driven output with keepalive resend to avoid missed downstream events.
        same_track = track_id == self._last_sent_track_id
        keepalive_due = (
            same_track
            and track_id >= 0
            and (now - self._last_sent_target_ts) >= self.target_cmd_keepalive_sec
        )
        if same_track and not keepalive_due:
            return

        bottle = self.vision_cmd_port.prepare()
        bottle.clear()
        bottle.addInt32(track_id)
        bottle.addFloat64(ips)
        self.vision_cmd_port.write()

        self._last_sent_track_id = track_id
        self._last_sent_ips = ips
        self._last_sent_target_ts = now

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
                target.get("person_id", target.get("face_id", "unknown")))
        )
        ss = target.get("social_state", "ss1")
        with self._interaction_lock:
            self._active_interaction_attempt_id = attempt_id
            self._active_interaction_track_id = int(track_id)
            self._active_interaction_person_id = face_id
        self._db_log(
            "interaction_state_event",
            {
                "event_type": "start",
                "attempt_id": attempt_id,
                "exec_interaction_id": None,
                "track_id": track_id,
                "face_id": raw_face_id,
                "person_id": face_id,
                "social_state": ss,
                "success": None,
                "abort_reason": None,
                "duration_sec": 0.0,
                "hunger_state": self._last_hunger_state,
            })
        try:
            self._log("INFO", f"talk: start t{track_id} {face_id} {ss}")

            result = self._run_executive_control(track_id, face_id, "ss3" if ss == "ss4" else ss)
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
                hunger_state=self._last_hunger_state,
                is_proactive=1)
            self._db_log("interaction_attempt", asdict(attempt))
            self._db_log(
                "interaction_state_event",
                {
                    "event_type": "end",
                    "attempt_id": attempt_id,
                    "exec_interaction_id": exec_interaction_id,
                    "track_id": track_id,
                    "face_id": raw_face_id,
                    "person_id": face_id,
                    "social_state": ss,
                    "success": int(bool(result and result.get("success"))),
                    "abort_reason": abort_reason,
                    "duration_sec": duration_sec,
                    "hunger_state": self._last_hunger_state,
                })
            with self._interaction_lock:
                with self.state_lock:
                    self.interaction_busy = False
                    self.selected_target = None
                    self._active_interaction_attempt_id = None
                    self._active_interaction_track_id = -1
                    self._active_interaction_person_id = ""
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
                if self._ensure_exec_rpc_connected(log_on_fail=False):
                    cmd = yarp.Bottle()
                    cmd.addString("status")
                    reply = yarp.Bottle()
                    self.executive_control_rpc.write(cmd, reply)
                    break
            except Exception:
                time.sleep(2.0)

    def _ensure_exec_rpc_connected(self, log_on_fail: bool = True) -> bool:
        try:
            if self.executive_control_rpc is None:
                return False
            if self.executive_control_rpc.getOutputCount() > 0:
                return True
            local = self._exec_rpc_local_name or f"/{self.module_name}/executiveControl:rpc"
            ok = yarp.Network.connect(local, self.executive_control_rpc_name)
            if ok:
                return self.executive_control_rpc.getOutputCount() > 0
            if log_on_fail:
                self._log("DEBUG", f"exec: reconnect failed ({local} -> {self.executive_control_rpc_name})")
            return False
        except Exception as e:
            if log_on_fail:
                self._log("WARNING", f"exec: reconnect error: {e}")
            return False

    def _run_executive_control(
        self, track_id: int, face_id: str, start_state: str
    ) -> Optional[Dict]:
        try:
            if not self._ensure_exec_rpc_connected(log_on_fail=False):
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
                                == "reactive_interaction_running"
                            ):
                                self._log(
                                    "INFO",
                                    "exec: busy, skip")
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
            if not self._ensure_exec_rpc_connected(log_on_fail=False):
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
            initial_ss = str(target.get("social_state") or result.get("initial_state", "ss1"))
            if initial_ss == "ss4":
                # ss4 re-engagement uses an ss3 action but the person stays ss4.
                final_ss = "ss4"
            else:
                final_ss = result.get("final_state", initial_ss)

            greeted = final_ss in ("ss3", "ss4") and initial_ss != "ss4"
            talked = final_ss == "ss4" and initial_ss != "ss4"

            track_id = target["track_id"]
            person_id = str(
                target.get("person_id") or target.get("face_id") or "unknown"
            )
            resolved_person_id = self._resolve_result_person_id(result)
            if self._is_face_known(resolved_person_id):
                person_id = resolved_person_id

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

            self._apply_homeostatic_learning(result, person_id)

            if initial_ss != final_ss:
                self._db_log(
                    "ss_change",
                    {
                        "person_id": person_id,
                        "old_ss": initial_ss,
                        "new_ss": final_ss,
                    })

        except Exception as e:
            self._log("ERROR", f"process_interaction_result failed: {e}")

    def _resolve_result_person_id(self, result: Dict[str, Any]) -> str:
        """Resolve a stable known person ID from an interaction result payload."""
        for key in ("person_id", "name", "resolved_face_id", "face_id"):
            value = result.get(key)
            if value is None:
                continue
            candidate = str(value).strip()
            if self._is_face_known(candidate):
                return candidate

        try:
            track_id = int(result.get("track_id", -1))
        except (TypeError, ValueError):
            track_id = -1
        if track_id >= 0:
            with self.state_lock:
                candidate = self.track_to_person.get(track_id, "")
            if self._is_face_known(candidate):
                return str(candidate)
        return "unknown"

    def _process_external_interaction_result(self, result: Dict[str, Any], person_id: str) -> None:
        """Handle completed interactions submitted by executiveControl reactive paths."""
        try:
            try:
                track_id = int(result.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1

            initial_ss = str(result.get("initial_state", "ss1"))
            final_ss = str(result.get("final_state", initial_ss))
            greeted = final_ss in ("ss3", "ss4") and initial_ss != "ss4"
            talked = final_ss == "ss4" and initial_ss != "ss4"
            now_iso = datetime.now(self.TIMEZONE).isoformat()

            if self._is_face_known(person_id):
                with self._memory_lock:
                    with self.state_lock:
                        if track_id >= 0:
                            self.track_to_person[track_id] = person_id
                        if greeted:
                            self.greeted_today[person_id] = now_iso
                        if talked:
                            self.talked_today[person_id] = now_iso
                if greeted:
                    self._enqueue_save("greeted")
                if talked:
                    self._enqueue_save("talked")

            self._apply_homeostatic_learning(result, person_id)
        except Exception as e:
            self._log("ERROR", f"process_external_interaction_result failed: {e}")

    @staticmethod
    def _result_bool(result: Dict[str, Any], key: str, default: bool = True) -> bool:
        value = result.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", "disabled", "hs0"}
        return default

    @staticmethod
    def _result_float(result: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(result.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _result_int(result: Dict[str, Any], key: str, default: int = 0) -> int:
        try:
            return int(result.get(key, default))
        except (TypeError, ValueError):
            return default

    def _log_homeostatic_delta(
        self,
        result: Dict[str, Any],
        person_id: str,
        reward: float,
        outcome: str,
        reason: str,
        affinity_before: Optional[float],
        affinity_after: Optional[float]) -> None:
        delta = self.HomeostaticLearningDelta(
            person_id=person_id,
            reward_delta=reward,
            outcome=outcome,
            reason=reason,
            trigger_mode=result.get("trigger_mode"),
            hunger_state_start=result.get("hunger_state_start"),
            hunger_state_end=result.get("hunger_state_end"),
            stomach_level_start=self._result_float(result, "stomach_level_start", 100.0),
            stomach_level_end=self._result_float(result, "stomach_level_end", 100.0),
            active_energy_cost=self._result_float(result, "active_energy_cost", 0.0),
            meals_eaten_count=self._result_int(result, "meals_eaten_count", 0),
            n_turns=self._result_int(result, "n_turns", 0),
            exec_interaction_id=result.get("interaction_id"),
            affinity_before=affinity_before,
            affinity_after=affinity_after)
        self._db_log("homeostatic_learning_change", asdict(delta))

    def _apply_homeostatic_learning(self, result: Dict[str, Any], person_id: str) -> None:
        """Update a person's affinity (EMA of normalized reward) from one interaction.

        Fed -> reinforce; cost while barely engaged -> penalize; cost after a real chat
        the person left, or a tiny delta -> neutral, no change.
        """
        person_id = str(person_id or "").strip()
        reward = self._result_float(result, "homeostatic_reward", 0.0)

        if not self._is_face_known(person_id):
            self._log_homeostatic_delta(
                result, person_id or "unknown", reward, "skipped", "unknown_person", None, None)
            return

        if (not self._result_bool(result, "hunger_drive_enabled", True)
                or str(result.get("hunger_state_start", "")).upper() == "HS0"):
            a = self._person_affinity(person_id)
            self._log_homeostatic_delta(result, person_id, reward, "skipped", "hunger_disabled", a, a)
            return

        n_turns = self._result_int(result, "n_turns", 0)
        band = self.AFFINITY_NEUTRAL_BAND
        positive = reward > band
        early_abandonment = reward < -band and n_turns < self.ENGAGED_MIN_TURNS

        if not positive and not early_abandonment:
            a = self._person_affinity(person_id)
            reason = "target_left_after_engagement" if reward < -band else "neutral_delta"
            self._log_homeostatic_delta(result, person_id, reward, "neutral", reason, a, a)
            self._log("INFO", f"affinity neutral: {person_id} reason={reason} (turns={n_turns} reward={reward:.1f})")
            return

        r_norm = self._clamp_affinity(reward / self.AFFINITY_REWARD_SCALE)
        outcome = "drive_reduced" if positive else "drive_depleted"
        reason = "food_received" if positive else "no_response"

        with self._memory_lock:
            profile = self.homeostatic_profiles.get(person_id, {})
            interactions_prev = self._result_int(profile, "interactions", 0) if isinstance(profile, dict) else 0
            affinity_before = self._person_affinity(person_id)
            affinity_after = self._clamp_affinity(
                r_norm if interactions_prev <= 0
                else affinity_before + self.AFFINITY_ALPHA * (r_norm - affinity_before))
            self.homeostatic_profiles[person_id] = {
                "affinity": affinity_after,
                "interactions": interactions_prev + 1,
                "last_reward": reward,
                "last_outcome": outcome,
                "last_trigger_mode": str(result.get("trigger_mode", "proactive")),
                "updated_at": datetime.now(self.TIMEZONE).isoformat(),
            }

        self._enqueue_save("homeostatic_learning")
        self._log_homeostatic_delta(result, person_id, reward, outcome, reason, affinity_before, affinity_after)
        self._log(
            "INFO",
            f"affinity: {person_id} {affinity_before:+.2f} -> {affinity_after:+.2f} "
            f"(reward={reward:.1f} r_norm={r_norm:+.2f} {outcome})")

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

    def _load_memory_from_disk(self, log_prefix: str) -> None:
        """Read memory JSON files from disk and prune greeted/talked to today."""
        with self._memory_lock:
            self.greeted_today = self._prune_to_today(
                self._load_json(self.greeted_path, {})
            )
            self.talked_today = self._prune_to_today(
                self._load_json(self.talked_path, {})
            )
            learning_raw = self._load_json(
                self.homeostatic_learning_path,
                {"schema_version": 1, "people": {}})
            people = learning_raw.get("people", {}) if isinstance(learning_raw, dict) else {}
            self.homeostatic_profiles = people if isinstance(people, dict) else {}
        self._log(
            "INFO",
            f"{log_prefix} – greeted:{len(self.greeted_today)} "
            f"talked:{len(self.talked_today)} "
            f"homeostatic_profiles:{len(self.homeostatic_profiles)}")

    def _load_all_json_files(self):
        self._load_memory_from_disk("Loaded")

    def _reload_memory_from_disk_and_prune_today(self):
        """Re-read memory on day-change to pick up out-of-process writes."""
        self._load_memory_from_disk("Day-change reload")

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

    def _save_homeostatic_learning_json(self):
        with self._memory_lock:
            data = {"schema_version": 1, "people": dict(self.homeostatic_profiles)}
        self._save_json_atomic(self.homeostatic_learning_path, data)

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
                elif item == "homeostatic_learning":
                    self._save_homeostatic_learning_json()
            except Exception as e:
                self._log("ERROR", f"IO worker save failed ({item}): {e}")

    # ==================== SQLite DB Logging ====================

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys=ON")
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS target_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                timestamp_local TEXT NOT NULL,
                timezone TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                monotonic_sec REAL NOT NULL,
                run_elapsed_sec REAL NOT NULL,
                day_rome TEXT NOT NULL,
                run_id TEXT,
                is_test_run INTEGER NOT NULL DEFAULT 0 CHECK (is_test_run IN (0,1)),
                valid_for_analysis INTEGER NOT NULL DEFAULT 1 CHECK (valid_for_analysis IN (0,1)),
                track_id INTEGER NOT NULL,
                face_id TEXT NOT NULL,
                person_id TEXT NOT NULL,
                bbox_area REAL,
                ips REAL,
                ss TEXT NOT NULL CHECK (ss IN ('ss1','ss2','ss3','ss4')),
                eligible INTEGER NOT NULL CHECK (eligible IN (0,1)),
                context_label INTEGER,
                reason TEXT,
                last_greeted_ts TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS face_ips_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                timestamp_local TEXT NOT NULL,
                timezone TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                monotonic_sec REAL NOT NULL,
                run_elapsed_sec REAL NOT NULL,
                day_rome TEXT NOT NULL,
                run_id TEXT,
                is_test_run INTEGER NOT NULL DEFAULT 0 CHECK (is_test_run IN (0,1)),
                valid_for_analysis INTEGER NOT NULL DEFAULT 1 CHECK (valid_for_analysis IN (0,1)),
                track_id INTEGER,
                face_id TEXT,
                person_id TEXT,
                social_state TEXT CHECK (social_state IS NULL OR social_state IN ('ss1','ss2','ss3','ss4')),
                is_known INTEGER CHECK (is_known IS NULL OR is_known IN (0,1)),
                eligible INTEGER CHECK (eligible IS NULL OR eligible IN (0,1)),
                is_active_target INTEGER CHECK (is_active_target IS NULL OR is_active_target IN (0,1)),
                bbox_area REAL,
                ips REAL,
                ips_before_habituation REAL,
                habituation_applied INTEGER,
                habituation_multiplier REAL,
                habituation_elapsed_sec REAL,
                habituation_ips_delta REAL,
                stimulus_type TEXT,
                context_label INTEGER,
                prox_score REAL,
                cent_score REAL,
                vel_score REAL,
                gaze_score REAL,
                weight_prox REAL,
                weight_cent REAL,
                weight_vel REAL,
                weight_gaze REAL
            )""")

            c.execute("""CREATE TABLE IF NOT EXISTS ss_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                timestamp_local TEXT NOT NULL,
                timezone TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                monotonic_sec REAL NOT NULL,
                run_elapsed_sec REAL NOT NULL,
                day_rome TEXT NOT NULL,
                run_id TEXT,
                is_test_run INTEGER NOT NULL DEFAULT 0 CHECK (is_test_run IN (0,1)),
                valid_for_analysis INTEGER NOT NULL DEFAULT 1 CHECK (valid_for_analysis IN (0,1)),
                person_id TEXT NOT NULL,
                old_ss TEXT NOT NULL CHECK (old_ss IN ('ss1','ss2','ss3','ss4')),
                new_ss TEXT NOT NULL CHECK (new_ss IN ('ss1','ss2','ss3','ss4'))
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS homeostatic_learning_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                timestamp_local TEXT NOT NULL,
                timezone TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                monotonic_sec REAL NOT NULL,
                run_elapsed_sec REAL NOT NULL,
                day_rome TEXT NOT NULL,
                run_id TEXT,
                is_test_run INTEGER NOT NULL DEFAULT 0 CHECK (is_test_run IN (0,1)),
                valid_for_analysis INTEGER NOT NULL DEFAULT 1 CHECK (valid_for_analysis IN (0,1)),
                person_id TEXT NOT NULL,
                reward_delta REAL NOT NULL,
                outcome TEXT,
                reason TEXT,
                trigger_mode TEXT CHECK (trigger_mode IS NULL OR trigger_mode IN ('proactive','reactive')),
                hunger_state_start TEXT CHECK (hunger_state_start IS NULL OR hunger_state_start IN ('HS0','HS1','HS2','HS3')),
                hunger_state_end TEXT CHECK (hunger_state_end IS NULL OR hunger_state_end IN ('HS0','HS1','HS2','HS3')),
                stomach_level_start REAL,
                stomach_level_end REAL,
                active_energy_cost REAL,
                meals_eaten_count INTEGER,
                n_turns INTEGER,
                exec_interaction_id TEXT,
                affinity_before REAL,
                affinity_after REAL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS interaction_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                timestamp_local TEXT NOT NULL,
                timezone TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                monotonic_sec REAL NOT NULL,
                run_elapsed_sec REAL NOT NULL,
                day_rome TEXT NOT NULL,
                run_id TEXT,
                is_test_run INTEGER NOT NULL DEFAULT 0 CHECK (is_test_run IN (0,1)),
                valid_for_analysis INTEGER NOT NULL DEFAULT 1 CHECK (valid_for_analysis IN (0,1)),
                attempt_id TEXT NOT NULL UNIQUE,
                track_id INTEGER NOT NULL,
                face_id TEXT NOT NULL,
                person_id TEXT NOT NULL,
                start_ss TEXT NOT NULL CHECK (start_ss IN ('ss1','ss2','ss3','ss4')),
                success INTEGER NOT NULL CHECK (success IN (0,1)),
                final_state TEXT CHECK (final_state IS NULL OR final_state IN ('ss1','ss2','ss3','ss4')),
                abort_reason TEXT,
                exec_interaction_id TEXT,
                duration_sec REAL,
                hunger_state TEXT CHECK (hunger_state IS NULL OR hunger_state IN ('HS0','HS1','HS2','HS3')),
                is_proactive INTEGER NOT NULL DEFAULT 1 CHECK (is_proactive IN (0,1))
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS interaction_state_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                timestamp_local TEXT NOT NULL,
                timezone TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                monotonic_sec REAL NOT NULL,
                run_elapsed_sec REAL NOT NULL,
                day_rome TEXT NOT NULL,
                run_id TEXT,
                is_test_run INTEGER NOT NULL DEFAULT 0 CHECK (is_test_run IN (0,1)),
                valid_for_analysis INTEGER NOT NULL DEFAULT 1 CHECK (valid_for_analysis IN (0,1)),
                event_type TEXT NOT NULL CHECK (event_type IN ('start','end')),
                attempt_id TEXT NOT NULL,
                exec_interaction_id TEXT,
                track_id INTEGER NOT NULL,
                face_id TEXT NOT NULL,
                person_id TEXT NOT NULL,
                social_state TEXT NOT NULL CHECK (social_state IN ('ss1','ss2','ss3','ss4')),
                success INTEGER CHECK (success IS NULL OR success IN (0,1)),
                abort_reason TEXT,
                duration_sec REAL,
                hunger_state TEXT CHECK (hunger_state IS NULL OR hunger_state IN ('HS0','HS1','HS2','HS3'))
            )""")
            c.execute(
                "INSERT INTO schema_info(key,value) VALUES('schema_version','4') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
            for key, value in self.experiment.experiment_fields().items():
                c.execute(
                    "INSERT INTO schema_info(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(value) if value is not None else ""))
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_target_selections_time ON target_selections(timestamp_utc)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_target_selections_track ON target_selections(track_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_target_selections_person ON target_selections(person_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_ips_events_time ON face_ips_events(timestamp_utc)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_ips_events_track ON face_ips_events(track_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_ips_events_person ON face_ips_events(person_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_ips_events_valid ON face_ips_events(valid_for_analysis)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_attempts_time ON interaction_attempts(timestamp_utc)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_attempts_exec_id ON interaction_attempts(exec_interaction_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_ia_hunger ON interaction_attempts(hunger_state)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_ia_hs_day ON interaction_attempts(hunger_state, timestamp_utc)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_ia_run ON interaction_attempts(run_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_ia_valid ON interaction_attempts(valid_for_analysis)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_state_events_attempt ON interaction_state_events(attempt_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_homeostatic_learning_changes_time ON homeostatic_learning_changes(timestamp_utc)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_homeostatic_learning_changes_person ON homeostatic_learning_changes(person_id)"
            )

            self._create_analytics_views(conn)
            conn.commit()
            conn.close()
            self._log("INFO", f"DB ready: {self.db_path}")
        except Exception as e:
            self._log("ERROR", f"DB init failed: {e}")

    def _create_analytics_views(self, conn: sqlite3.Connection):
        c = conn.cursor()
        c.execute("DROP VIEW IF EXISTS v_interaction_attempts_clean")
        c.execute(
            """
            CREATE VIEW v_interaction_attempts_clean AS
            SELECT
                attempt_id,
                timestamp_utc,
                timestamp_local,
                timezone,
                timestamp_epoch,
                monotonic_sec,
                run_elapsed_sec,
                day_rome,
                run_id,
                is_test_run,
                valid_for_analysis,
                track_id,
                face_id,
                person_id,
                start_ss,
                CAST(success AS INTEGER) AS success,
                final_state,
                abort_reason,
                CASE
                    WHEN abort_reason IS NULL THEN NULL
                    WHEN abort_reason IN ('no_response_greeting','no_response_conversation','no_response_name') THEN 'no_response'
                    WHEN abort_reason IN ('target_lost','face_disappeared','target_monitor_abort') THEN 'target_lost'
                    WHEN abort_reason LIKE 'exception:%' THEN 'error'
                    ELSE 'system_abort'
                END AS abort_category,
                exec_interaction_id,
                duration_sec,
                COALESCE(hunger_state, 'HS0')        AS hunger_state,
                COALESCE(is_proactive, 1)            AS is_proactive
            FROM interaction_attempts
            WHERE start_ss IN ('ss1', 'ss2', 'ss3')
              AND valid_for_analysis = 1
            """
        )

        c.execute("DROP VIEW IF EXISTS v_interaction_attempts_daily")
        c.execute(
            """
            CREATE VIEW v_interaction_attempts_daily AS
            SELECT
                day_rome,
                run_id, hunger_state,
                start_ss,
                COUNT(*)                                                          AS launched,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)                     AS completed,
                SUM(CASE WHEN final_state = 'ss4' THEN 1 ELSE 0 END)             AS reached_ss4,
                1.0 * SUM(CASE WHEN final_state = 'ss4' THEN 1 ELSE 0 END)
                      / MAX(COUNT(*), 1)                                          AS ss4_rate,
                SUM(CASE WHEN is_proactive = 1 THEN 1 ELSE 0 END)                AS proactive_count,
                SUM(CASE WHEN abort_category='no_response'  THEN 1 ELSE 0 END)   AS ignored_count,
                SUM(CASE WHEN abort_category='target_lost'  THEN 1 ELSE 0 END)   AS target_lost_count,
                SUM(CASE WHEN abort_category='system_abort' THEN 1 ELSE 0 END)   AS system_abort_count,
                SUM(CASE WHEN abort_category='error'        THEN 1 ELSE 0 END)   AS error_count,
                SUM(CASE WHEN abort_reason IS NOT NULL THEN 1 ELSE 0 END)        AS aborted_count,
                AVG(duration_sec)                                                 AS avg_duration_sec
            FROM v_interaction_attempts_clean
            GROUP BY run_id, day_rome, hunger_state, start_ss
            """
        )

        c.execute("DROP VIEW IF EXISTS v_face_ips_timeline")
        c.execute(
            """
            CREATE VIEW v_face_ips_timeline AS
            SELECT
                timestamp_utc,
                timestamp_local,
                timezone,
                timestamp_epoch,
                monotonic_sec,
                run_elapsed_sec,
                day_rome,
                run_id,
                is_test_run,
                valid_for_analysis,
                track_id,
                face_id,
                person_id,
                social_state,
                CAST(is_known AS INTEGER)                    AS is_known,
                CAST(eligible AS INTEGER)                    AS eligible,
                CAST(is_active_target AS INTEGER)            AS is_active_target,
                bbox_area,
                ips,
                ips_before_habituation,
                CAST(habituation_applied AS INTEGER)         AS habituation_applied,
                habituation_multiplier,
                habituation_elapsed_sec,
                habituation_ips_delta,
                stimulus_type,
                context_label,
                prox_score,
                cent_score,
                vel_score,
                gaze_score,
                weight_prox,
                weight_cent,
                weight_vel,
                weight_gaze
            FROM face_ips_events
            """
        )
        c.execute("DROP VIEW IF EXISTS v_target_selections_clean")
        c.execute(
            """
            CREATE VIEW v_target_selections_clean AS
            SELECT
                timestamp_utc,
                timestamp_local,
                timezone,
                timestamp_epoch,
                monotonic_sec,
                run_elapsed_sec,
                day_rome,
                run_id,
                is_test_run,
                valid_for_analysis,
                track_id,
                face_id,
                person_id,
                bbox_area,
                ips,
                ss,
                eligible,
                context_label,
                reason,
                last_greeted_ts
            FROM target_selections
            WHERE valid_for_analysis = 1
            """
        )
        c.execute("DROP VIEW IF EXISTS v_homeostatic_learning_changes_clean")
        c.execute(
            """
            CREATE VIEW v_homeostatic_learning_changes_clean AS
            SELECT
                timestamp_utc,
                timestamp_local,
                timezone,
                timestamp_epoch,
                monotonic_sec,
                run_elapsed_sec,
                day_rome,
                run_id,
                is_test_run,
                valid_for_analysis,
                person_id,
                reward_delta,
                outcome,
                reason,
                trigger_mode,
                hunger_state_start,
                hunger_state_end,
                stomach_level_start,
                stomach_level_end,
                active_energy_cost,
                meals_eaten_count,
                n_turns,
                exec_interaction_id,
                affinity_before,
                affinity_after
            FROM homeostatic_learning_changes
            WHERE valid_for_analysis = 1
            """
        )
        c.execute("DROP VIEW IF EXISTS v_interaction_state_events")
        c.execute(
            """
            CREATE VIEW v_interaction_state_events AS
            SELECT
                timestamp_utc,
                timestamp_local,
                timezone,
                timestamp_epoch,
                monotonic_sec,
                run_elapsed_sec,
                day_rome,
                run_id,
                is_test_run,
                valid_for_analysis,
                event_type,
                attempt_id,
                exec_interaction_id,
                track_id,
                face_id,
                person_id,
                social_state,
                success,
                abort_reason,
                duration_sec,
                hunger_state
            FROM interaction_state_events
            """
        )
        c.execute("DROP VIEW IF EXISTS v_quality_salience_missing_metadata")
        c.execute(
            """
            CREATE VIEW v_quality_salience_missing_metadata AS
            SELECT 'interaction_attempts' AS table_name, id, timestamp_utc, run_id
            FROM interaction_attempts
            WHERE run_id IS NULL
            UNION ALL
            SELECT 'face_ips_events' AS table_name, id, timestamp_utc, run_id
            FROM face_ips_events
            WHERE run_id IS NULL
            """
        )
        c.execute("DROP VIEW IF EXISTS v_quality_attempt_counts")
        c.execute(
            """
            CREATE VIEW v_quality_attempt_counts AS
            SELECT run_id, hunger_state,
                   COUNT(*) AS attempt_count
            FROM interaction_attempts
            GROUP BY run_id, hunger_state
            """
        )

    def _time_fields(
        self,
        *,
        epoch: Optional[float] = None,
        monotonic_sec: Optional[float] = None) -> Dict[str, Any]:
        now_epoch = time.time()
        now_mono = time.monotonic()
        mono = float(now_mono if monotonic_sec is None else monotonic_sec)
        ts_epoch = float(now_epoch - (now_mono - mono) if epoch is None else epoch)
        utc_dt = datetime.fromtimestamp(ts_epoch, timezone.utc)
        local_dt = utc_dt.astimezone(self.TIMEZONE)
        return {
            "timestamp_utc": utc_dt.isoformat(timespec="milliseconds"),
            "timestamp_local": local_dt.isoformat(timespec="milliseconds"),
            "timezone": str(self.TIMEZONE),
            "timestamp_epoch": ts_epoch,
            "monotonic_sec": mono,
            "run_elapsed_sec": mono - self._run_started_mono,
            "day_rome": local_dt.date().isoformat(),
        }

    def _db_log(self, table: str, data: Dict):
        for key, value in self._time_fields().items():
            data.setdefault(key, value)
        for key, value in self.experiment.experiment_fields().items():
            data.setdefault(key, value)
        self._db_queue.put((table, data))

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
        conn.execute("PRAGMA foreign_keys=ON")
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
                        """INSERT INTO target_selections
                        (timestamp_utc,timestamp_local,timezone,timestamp_epoch,
                         monotonic_sec,run_elapsed_sec,day_rome,run_id,
                         is_test_run,valid_for_analysis,track_id,face_id,
                         person_id,bbox_area,ips,ss,eligible,context_label,reason,
                         last_greeted_ts)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            data["timestamp_utc"],
                            data["timestamp_local"],
                            data["timezone"],
                            data["timestamp_epoch"],
                            data["monotonic_sec"],
                            data["run_elapsed_sec"],
                            data["day_rome"],
                            data.get("run_id"),
                            int(data.get("is_test_run", 0)),
                            int(data.get("valid_for_analysis", 1)),
                            data["track_id"],
                            data["face_id"],
                            data["person_id"],
                            data["bbox_area"],
                            data.get("ips"),
                            data["ss"],
                            int(data["eligible"]),
                            data.get("context_label"),
                            data.get("reason"),
                            data.get("last_greeted_ts")))
                elif table == "face_ips_event":
                    c.execute(
                        """INSERT INTO face_ips_events
                        (timestamp_utc,timestamp_local,timezone,timestamp_epoch,
                         monotonic_sec,run_elapsed_sec,day_rome,run_id,
                         is_test_run,valid_for_analysis,
                         track_id,face_id,person_id,social_state,is_known,eligible,
                         is_active_target,bbox_area,ips,ips_before_habituation,
                         habituation_applied,habituation_multiplier,habituation_elapsed_sec,
                         habituation_ips_delta,stimulus_type,context_label,
                         prox_score,cent_score,vel_score,gaze_score,
                         weight_prox,weight_cent,weight_vel,weight_gaze)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            data["timestamp_utc"],
                            data["timestamp_local"],
                            data["timezone"],
                            data["timestamp_epoch"],
                            data["monotonic_sec"],
                            data["run_elapsed_sec"],
                            data["day_rome"],
                            data.get("run_id"),
                            int(data.get("is_test_run", 0)),
                            int(data.get("valid_for_analysis", 1)),
                            data.get("track_id"),
                            data.get("face_id"),
                            data.get("person_id"),
                            data.get("social_state"),
                            int(data.get("is_known", 0)),
                            int(data.get("eligible", 0)),
                            int(data.get("is_active_target", 0)),
                            data.get("bbox_area"),
                            data.get("ips"),
                            data.get("ips_before_habituation"),
                            int(data.get("habituation_applied", 0)),
                            data.get("habituation_multiplier"),
                            data.get("habituation_elapsed_sec"),
                            data.get("habituation_ips_delta"),
                            data.get("stimulus_type"),
                            data.get("context_label"),
                            data.get("prox_score"),
                            data.get("cent_score"),
                            data.get("vel_score"),
                            data.get("gaze_score"),
                            data.get("weight_prox"),
                            data.get("weight_cent"),
                            data.get("weight_vel"),
                            data.get("weight_gaze")))
                elif table == "ss_change":
                    c.execute(
                        """INSERT INTO ss_changes
                        (timestamp_utc,timestamp_local,timezone,timestamp_epoch,
                         monotonic_sec,run_elapsed_sec,day_rome,run_id,
                         is_test_run,valid_for_analysis,
                         person_id,old_ss,new_ss)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            data["timestamp_utc"],
                            data["timestamp_local"],
                            data["timezone"],
                            data["timestamp_epoch"],
                            data["monotonic_sec"],
                            data["run_elapsed_sec"],
                            data["day_rome"],
                            data.get("run_id"),
                            int(data.get("is_test_run", 0)),
                            int(data.get("valid_for_analysis", 1)),
                            data["person_id"],
                            data["old_ss"],
                            data["new_ss"]))
                elif table == "homeostatic_learning_change":
                    c.execute(
                        """INSERT INTO homeostatic_learning_changes
                        (timestamp_utc,timestamp_local,timezone,timestamp_epoch,
                         monotonic_sec,run_elapsed_sec,day_rome,run_id,
                         is_test_run,valid_for_analysis,
                         person_id,reward_delta,outcome,reason,trigger_mode,
                         hunger_state_start,hunger_state_end,stomach_level_start,stomach_level_end,
                         active_energy_cost,meals_eaten_count,n_turns,exec_interaction_id,
                         affinity_before,affinity_after)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            data["timestamp_utc"],
                            data["timestamp_local"],
                            data["timezone"],
                            data["timestamp_epoch"],
                            data["monotonic_sec"],
                            data["run_elapsed_sec"],
                            data["day_rome"],
                            data.get("run_id"),
                            int(data.get("is_test_run", 0)),
                            int(data.get("valid_for_analysis", 1)),
                            data["person_id"],
                            data["reward_delta"],
                            data.get("outcome"),
                            data.get("reason"),
                            data.get("trigger_mode"),
                            data.get("hunger_state_start"),
                            data.get("hunger_state_end"),
                            data.get("stomach_level_start"),
                            data.get("stomach_level_end"),
                            data.get("active_energy_cost"),
                            data.get("meals_eaten_count"),
                            data.get("n_turns"),
                            data.get("exec_interaction_id"),
                            data.get("affinity_before"),
                            data.get("affinity_after")))
                elif table == "interaction_attempt":
                    c.execute(
                        """INSERT INTO interaction_attempts
                        (timestamp_utc,timestamp_local,timezone,timestamp_epoch,
                         monotonic_sec,run_elapsed_sec,day_rome,run_id,
                         is_test_run,valid_for_analysis,
                         attempt_id,track_id,face_id,person_id,start_ss,
                         success,final_state,abort_reason,exec_interaction_id,duration_sec,
                         hunger_state,is_proactive)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            data["timestamp_utc"],
                            data["timestamp_local"],
                            data["timezone"],
                            data["timestamp_epoch"],
                            data["monotonic_sec"],
                            data["run_elapsed_sec"],
                            data["day_rome"],
                            data.get("run_id"),
                            int(data.get("is_test_run", 0)),
                            int(data.get("valid_for_analysis", 1)),
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
                            data.get("hunger_state", "HS0"),
                            int(data.get("is_proactive", 1))))
                elif table == "interaction_state_event":
                    c.execute(
                        """INSERT INTO interaction_state_events
                        (timestamp_utc,timestamp_local,timezone,timestamp_epoch,
                         monotonic_sec,run_elapsed_sec,day_rome,run_id,
                         is_test_run,valid_for_analysis,
                         event_type,attempt_id,
                         exec_interaction_id,track_id,face_id,person_id,social_state,
                         success,abort_reason,duration_sec,hunger_state)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            data["timestamp_utc"],
                            data["timestamp_local"],
                            data["timezone"],
                            data["timestamp_epoch"],
                            data["monotonic_sec"],
                            data["run_elapsed_sec"],
                            data["day_rome"],
                            data.get("run_id"),
                            int(data.get("is_test_run", 0)),
                            int(data.get("valid_for_analysis", 1)),
                            data.get("event_type"),
                            data.get("attempt_id"),
                            data.get("exec_interaction_id"),
                            data.get("track_id"),
                            data.get("face_id"),
                            data.get("person_id"),
                            data.get("social_state"),
                            int(data["success"]) if data.get("success") is not None else None,
                            data.get("abort_reason"),
                            data.get("duration_sec"),
                            data.get("hunger_state")))
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
        line = f"[{ts}] [{level}] {message}\n"
        try:
            print(line, end="")
        except RuntimeError:
            # Ctrl+C can interrupt an in-flight buffered print and YARP may re-enter
            # shutdown hooks; fall back to a low-level write in that case.
            try:
                os.write(2, line.encode("utf-8", errors="replace"))
            except Exception:
                pass


# ==================== Main Entry Point ====================

def _run_module_with_main_thread_signals(
    module: SalienceNetworkModule, rf: yarp.ResourceFinder
) -> bool:
    stop_requested = threading.Event()
    run_done = threading.Event()
    run_state: Dict[str, Any] = {"result": False, "error": None}

    def _runner():
        try:
            run_state["result"] = bool(module.runModule(rf))
        except BaseException as e:
            run_state["error"] = e
        finally:
            run_done.set()

    def _handle_signal(signum, _frame):
        if stop_requested.is_set():
            return
        stop_requested.set()
        try:
            os.write(
                2,
                f"\n[INFO] SalienceNetworkModule received signal {signum}, shutting down.\n".encode(
                    "utf-8", errors="replace"
                ))
        except Exception:
            pass

    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

    run_thread = threading.Thread(
        target=_runner,
        name=f"{module.module_name}-runModule",
        daemon=True)
    run_thread.start()

    interrupted = False
    shutdown_deadline: Optional[float] = None
    try:
        while not run_done.wait(0.2):
            if stop_requested.is_set() and not interrupted:
                interrupted = True
                shutdown_deadline = time.time() + 2.0
                module.interruptModule()
            elif (
                interrupted
                and shutdown_deadline is not None
                and time.time() >= shutdown_deadline
            ):
                break
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        module.interruptModule()
        module.close()

    run_thread.join(timeout=0.5)
    if run_state["error"] is not None:
        raise run_state["error"]
    return bool(run_state["result"])

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
        if not _run_module_with_main_thread_signals(module, rf):
            print("[ERROR] Module failed to run.")
            sys.exit(1)
    finally:
        yarp.Network.fini()
        print("[INFO] Module closed.")
