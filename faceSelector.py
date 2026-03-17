"""
faceSelector.py – YARP RFModule for Real-Time Face Selection

Continuously reads face landmarks, computes social/spatial/learning states,
selects the biggest-bbox face as target candidate, and triggers interactions
via /interactionManager RPC.

Social States:
  ss1: unknown
  ss2: known, not greeted today
  ss3: known, greeted today, not talked
  ss4: known, greeted today, talked (ultimate)

YARP Connections:
    yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i
    yarp connect /icub/camcalib/left/out    /faceSelector/img:i
"""

import fcntl
import json
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERROR] OpenCV/NumPy required:  pip install opencv-python numpy")
    sys.exit(1)

try:
    import yarp
except ImportError:
    print("[ERROR] YARP Python bindings required.")
    sys.exit(1)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


class FaceSelectorModule(yarp.RFModule):
    """
    Real-time face selection module that:
    - Reads face landmarks from vision system
    - Computes social states (ss1-ss4) and spatial states
    - Maintains learning states (LS1-LS3) per person
    - Selects biggest-bbox face as target candidate
    - Gates interaction start on LS eligibility
    - Triggers interactions via /interactionManager RPC
    - Publishes annotated image with face boxes and states
    - Logs selections and state changes to face_selector.db
    """

    # ==================== Constants ====================

    # Social State string IDs
    SS_DESCRIPTIONS = {
        "ss1": "Unknown",
        "ss2": "Known, Not Greeted",
        "ss3": "Known, Greeted, No Talk",
        "ss4": "Known, Greeted, Talked",
    }

    # Learning State definitions
    LS1 = 1
    LS2 = 2
    LS3 = 3

    LS_NAMES = {1: "LS1", 2: "LS2", 3: "LS3"}

    LS_VALID_DISTANCES = {
        1: {"SO_CLOSE", "CLOSE"},
        2: {"SO_CLOSE", "CLOSE", "FAR"},
    }
    LS_VALID_ATTENTIONS = {
        1: {"MUTUAL_GAZE"},
        2: {"MUTUAL_GAZE", "NEAR_GAZE"}
    }

    LS_MIN_TIME_IN_VIEW = {
        1: 2.0,
        2: 1.0,
    }

    # Colors for drawing (BGR format)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (0, 0, 255)

    TIMEZONE = ZoneInfo("Europe/Rome")

    # ==================== Lifecycle ====================

    def __init__(self):
        super().__init__()

        self.module_name = "faceSelector"
        self.period = 0.05  # 20 Hz
        self._running = True

        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

        self.interaction_manager_rpc_name = "/interactionManager"
        self.interaction_interface_rpc_name = "/interactionInterface"

        self.learning_path = Path("/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/learning.json")
        self.greeted_path = Path("/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/greeted_today.json")
        self.talked_path = Path("/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/talked_today.json")
        self.last_greeted_path = Path("/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/memory/last_greeted.json")

        # YARP ports
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.img_in_port: Optional[yarp.BufferedPortImageRgb] = None
        self.img_out_port: Optional[yarp.BufferedPortImageRgb] = None
        self.debug_port: Optional[yarp.Port] = None
        self.interaction_manager_rpc: Optional[yarp.RpcClient] = None
        self.interaction_interface_rpc: Optional[yarp.RpcClient] = None
        self.stm_context_port: Optional[yarp.BufferedPortBottle] = None  # /alwayson/stm/context:o

        # Image handling
        self.img_width = 640
        self.img_height = 480
        self.last_annotated_frame: Optional[np.ndarray] = None

        # State tracking (thread-safe)
        self.state_lock = threading.Lock()
        self.current_faces: List[Dict[str, Any]] = []
        self.selected_target: Optional[Dict[str, Any]] = None
        self.selected_bbox_last: Optional[Tuple[float, float, float, float]] = None
        self.interaction_busy = False
        self.interaction_thread: Optional[threading.Thread] = None

        # Lock guarding in-memory memory dicts and their file I/O
        self._memory_lock = threading.Lock()
        # Lock guarding interaction_busy transitions and spawn decisions
        self._interaction_lock = threading.Lock()

        # Cooldown — base value and context-driven overrides
        self.last_interaction_time: Dict[str, float] = {}
        self.interaction_cooldown = 5.0   # fallback (kept for back-compat)
        self.cooldown_lively: float = 3.0   # label == 1 : active/lively scene → shorter
        self.cooldown_calm:   float = 15.0  # label == 0 : calm/quiet scene  → longer
        self.cooldown_default: float = 5.0  # label == -1 or unknown

        # STM context (cluster label from /alwayson/stm/context:o)
        self.current_context_label: int = -1  # -1 = not yet received

        # Image frame skip
        self.frame_skip_counter = 0
        self.frame_skip_rate = 0

        # Memory caches
        self.greeted_today: Dict[str, str] = {}
        self.talked_today: Dict[str, str] = {}
        self.learning_data: Dict[str, Dict] = {}

        # Last-greeted snapshot (refreshed by background thread)
        self._last_greeted_snapshot: Dict[str, Dict[str, Any]] = {}
        self._last_greeted_lock = threading.Lock()

        # Session tracking
        self.track_to_person: Dict[int, str] = {}

        # Day tracking
        self._current_day: Optional[date] = None

        # Config flags
        self.verbose_debug = False
        self.ports_connected_logged = False

        # Tracking for face_disappeared penalty
        self.DISAPPEAR_WINDOW_SEC = 30.0
        self.DISAPPEAR_THRESHOLD = 2
        self._disappear_events: Dict[str, List[float]] = {}

        # --- Background I/O ---
        self._io_queue: queue.Queue = queue.Queue()
        self._io_thread: Optional[threading.Thread] = None

        # --- DB logging ---
        self.db_path = "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture/modules/alwaysOn/data_collection/face_selector.db"
        self._db_queue: queue.Queue = queue.Queue()
        self._db_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ configure
    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            try:
                self.setName(self.module_name)
            except Exception:
                pass

            if rf.check("interaction_manager_rpc"):
                self.interaction_manager_rpc_name = rf.find("interaction_manager_rpc").asString()
            if rf.check("interaction_interface_rpc"):
                self.interaction_interface_rpc_name = rf.find("interaction_interface_rpc").asString()
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

            # --- Open ports ---
            port_specs = [
                ("landmarks_port", yarp.BufferedPortBottle,   f"/{self.module_name}/landmarks:i"),
                ("img_in_port",    yarp.BufferedPortImageRgb, f"/{self.module_name}/img:i"),
                ("img_out_port",   yarp.BufferedPortImageRgb, f"/{self.module_name}/img:o"),
            ]
            for attr, cls, name in port_specs:
                port = cls()
                if not port.open(name):
                    self._log("ERROR", f"Failed to open port: {name}")
                    return False
                setattr(self, attr, port)

            self.debug_port = yarp.Port()
            if not self.debug_port.open(f"/{self.module_name}/debug:o"):
                self._log("ERROR", f"Failed to open debug port")
                return False

            self.stm_context_port = yarp.BufferedPortBottle()
            if not self.stm_context_port.open(f"/{self.module_name}/context:i"):
                self._log("ERROR", "Failed to open STM context port")
                return False
            # Non-critical: try to auto-connect; continue even if STM is not yet running
            _ctx_remote = "/alwayson/stm/context:o"
            _ctx_local  = f"/{self.module_name}/context:i"
            if yarp.Network.connect(_ctx_remote, _ctx_local):
                self._log("INFO", f"STM context port connected: {_ctx_remote} → {_ctx_local}")
            else:
                self._log("WARNING", f"STM context port not yet available – will work without context (cooldown={self.cooldown_default}s)")

            self.interaction_manager_rpc = yarp.RpcClient()
            if not self.interaction_manager_rpc.open(f"/{self.module_name}/interactionManager:rpc"):
                self._log("ERROR", "Failed to open interactionManager RPC port")
                return False

            self.interaction_interface_rpc = yarp.RpcClient()
            if not self.interaction_interface_rpc.open(f"/{self.module_name}/interactionInterface:rpc"):
                self._log("ERROR", "Failed to open interactionInterface RPC port")
                return False

            # --- Auto-connect RPC ---
            for local, remote in [
                (f"/{self.module_name}/interactionManager:rpc",  self.interaction_manager_rpc_name),
                (f"/{self.module_name}/interactionInterface:rpc", self.interaction_interface_rpc_name),
            ]:
                if yarp.Network.connect(local, remote):
                    self._log("INFO", f"RPC connected → {remote}")
                else:
                    self._log("ERROR", f"RPC connect failed → {remote}")

            # --- Load persistent data ---
            self._load_all_json_files()
            self._refresh_last_greeted_snapshot()
            self._current_day = self._get_today_date()

            # --- Start background workers ---
            self._init_db()
            for target, name in [
                (self._io_worker,                  "_io_thread"),
                (self._db_worker,                  "_db_thread"),
                (self._last_greeted_refresh_loop,  "_lg_refresh_thread"),
            ]:
                t = threading.Thread(target=target, daemon=True)
                t.start()
                setattr(self, name, t)

            self._log("INFO", f"FaceSelectorModule ready @ {1.0/self.period:.0f} Hz")
            threading.Thread(target=self._prewarm_rpc_connections, daemon=True).start()
            return True

        except Exception as e:
            self._log("ERROR", f"configure() failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def interruptModule(self) -> bool:
        self._log("INFO", "Interrupting...")
        self._running = False
        for port in [self.landmarks_port, self.img_in_port, self.img_out_port,
                     self.debug_port, self.interaction_manager_rpc, self.interaction_interface_rpc,
                     self.stm_context_port]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        self._log("INFO", "Closing...")
        if self.interaction_thread and self.interaction_thread.is_alive():
            self.interaction_thread.join(timeout=5.0)
        # Enqueue final saves (async via IO worker)
        self._enqueue_save("greeted")
        self._enqueue_save("talked")
        self._enqueue_save("learning")
        # Signal IO worker to stop, then drain/join
        self._io_queue.put(None)
        if self._io_thread:
            self._io_thread.join(timeout=5.0)
            if self._io_thread.is_alive():
                self._log("WARNING", "IO worker did not finish in time – proceeding with shutdown")
        self._db_queue.put(None)
        if self._db_thread:
            self._db_thread.join(timeout=3.0)
            if self._db_thread.is_alive():
                self._log("WARNING", "DB worker did not finish in time – proceeding with shutdown")
        for port in [self.landmarks_port, self.img_in_port, self.img_out_port,
                     self.debug_port, self.interaction_manager_rpc, self.interaction_interface_rpc,
                     self.stm_context_port]:
            if port:
                port.close()
        return True

    def getPeriod(self) -> float:
        return self.period

    # ------------------------------------------------------------------ updateModule
    def updateModule(self) -> bool:
        if not self._running:
            return False

        try:
            landmarks_connected = self.landmarks_port.getInputCount() > 0
            img_connected = self.img_in_port.getInputCount() > 0

            if not landmarks_connected or not img_connected:
                if not self.ports_connected_logged:
                    lm = "OK" if landmarks_connected else "waiting"
                    img = "OK" if img_connected else "waiting"
                    self._log("INFO", f"Waiting for ports  landmarks:{lm}  img:{img}")
                    self.ports_connected_logged = True
                return True

            if self.ports_connected_logged:
                self._log("INFO", "✓ Input ports connected – starting processing")
                self.ports_connected_logged = False

            # Day-change check
            today = self._get_today_date()
            if self._current_day != today:
                self._log("INFO", f"Day change: {self._current_day} → {today}")
                self._reload_memory_from_disk_and_prune_today()
                self._enqueue_save("greeted")
                self._enqueue_save("talked")
                self._current_day = today

            # 0. Read STM context (non-blocking – best-effort)
            if self.stm_context_port is not None:
                ctx_btl = self.stm_context_port.read(False)
                if ctx_btl is not None:
                    self.current_context_label = ctx_btl.get(2).asInt8()
                    self._log("DEBUG", f"STM context updated → label={self.current_context_label}")

            # 1. Read landmarks
            faces = self._read_landmarks()

            # 2. Read image
            frame = None
            self.frame_skip_counter += 1
            if self.frame_skip_counter >= self.frame_skip_rate:
                self.frame_skip_counter = 0
                frame = self._read_image()

            # 3. Compute states
            with self.state_lock:
                self.current_faces = self._compute_face_states(faces)

            # 4. Select target (biggest bbox → wait resolve → LS gate → interact)
            #    Policy: ALWAYS pick biggest bbox. If unresolved, wait (don't switch
            #    to smaller resolved faces). Only interact once the biggest resolves.
            current_time = time.time()
            with self._interaction_lock:
                with self.state_lock:
                    if not self.interaction_busy:
                        candidate = self._select_biggest_face(self.current_faces)
                        if candidate:
                            face_id = candidate.get("face_id", "unknown")
                            track_id = candidate.get("track_id", -1)
                            person_id = candidate.get("person_id", face_id)
                            bbox = candidate.get("bbox", (0, 0, 0, 0))
                            area = bbox[2] * bbox[3]
                            resolved = self._is_face_id_resolved(face_id)

                            # If face_id not yet resolved → skip this cycle (don't fall back)
                            if not resolved:
                                if self.verbose_debug:
                                    self._log("DEBUG",
                                        f"Biggest face track={track_id} still resolving "
                                        f"(face_id='{face_id}', area={area:.0f}), waiting...")
                            else:
                                # Face is resolved — proceed with interaction checks
                                cd_key = str(person_id) if self._is_face_known(str(person_id)) else f"unknown:{track_id}"
                                last_int = self.last_interaction_time.get(cd_key, 0)

                                if current_time - last_int < self._effective_cooldown():
                                    if self.verbose_debug:
                                        self._log("DEBUG", f"{person_id} in cooldown (label={self.current_context_label}, cd={self._effective_cooldown():.1f}s)")
                                else:
                                    ss = candidate.get("social_state", "ss1")
                                    ls = candidate.get("learning_state", self.LS1)
                                    eligible = candidate.get("eligible", False)
                                    lg_ts = candidate.get("last_greeted_ts")

                                    # Log selection to DB
                                    self._db_log("target_selection", {
                                        "track_id": track_id,
                                        "face_id": face_id,
                                        "person_id": person_id,
                                        "bbox_area": area,
                                        "ss": ss, "ls": ls, "eligible": eligible,
                                        "last_greeted_ts": lg_ts,
                                    })

                                    if eligible:
                                        if ss == "ss4":
                                            if self.verbose_debug:
                                                self._log("DEBUG", f"{person_id} is ss4 – skipping")
                                        else:
                                            # Pre-spawn busy check (cooldown applied regardless)
                                            self.last_interaction_time[cd_key] = current_time

                                            im_status = self._interaction_manager_status()
                                            if isinstance(im_status, dict) and not im_status.get("busy", False):
                                                # Status confirmed not busy — spawn
                                                self.selected_target = candidate
                                                self.selected_bbox_last = candidate["bbox"]
                                                self.interaction_busy = True

                                                self._log("INFO",
                                                    f">>> TARGET: {person_id} "
                                                    f"(track={track_id}, {ss}, LS{ls}, area={area:.0f})")

                                                self.interaction_thread = threading.Thread(
                                                    target=self._run_interaction_thread,
                                                    args=(candidate,), daemon=True)
                                                self.interaction_thread.start()
                                            else:
                                                # Status is None (RPC error/not connected) or busy — skip spawn
                                                if im_status is None:
                                                    self._log("WARNING", "InteractionManager status unavailable – skipping")
                                                else:
                                                    self._log("INFO", "InteractionManager busy (pre-spawn) – skipping")
                                                # interaction_busy stays False; cooldown already applied
                                    elif self.verbose_debug:
                                        self._log("DEBUG", f"{person_id} not eligible (LS{ls})")

            # 5. Annotate & publish image
            if frame is not None:
                with self.state_lock:
                    annotated = self._annotate_image(frame, self.current_faces, self.selected_target)
                self.last_annotated_frame = annotated
                self._publish_image(annotated)
            elif self.last_annotated_frame is not None:
                self._publish_image(self.last_annotated_frame)

            # 6. Debug
            self._publish_debug()

            self._consecutive_errors = 0
            return True

        except Exception as e:
            self._consecutive_errors += 1
            self._log("ERROR", f"Error in updateModule: {e}")
            import traceback; traceback.print_exc()
            if self._consecutive_errors >= self._max_consecutive_errors:
                self._log("CRITICAL", "Too many consecutive errors, stopping")
                return False
            return True

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
            return faces
        for i in range(bottle.size()):
            face_btl = bottle.get(i)
            if not face_btl.isList():
                continue
            face_data = self._parse_face_bottle(face_btl.asList())
            if face_data:
                faces.append(face_data)
        return faces

    def _parse_face_bottle(self, bottle: yarp.Bottle) -> Optional[Dict[str, Any]]:
        if not bottle:
            return None
        data = {
            "face_id": "unknown", "track_id": -1,
            "bbox": (0.0, 0.0, 0.0, 0.0),
            "distance": "UNKNOWN", "gaze_direction": (0.0, 0.0, 1.0),
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "cos_angle": 0.0,
            "attention": "AWAY", "is_talking": 0, "time_in_view": 0.0,
        }
        try:
            i = 0
            while i < bottle.size():
                item = bottle.get(i)
                if item.isString():
                    key = item.asString()
                    if i + 1 < bottle.size():
                        nxt = bottle.get(i + 1)
                        if key == "face_id" and nxt.isString():
                            data["face_id"] = nxt.asString(); i += 2
                        elif key == "track_id" and (nxt.isInt32() or nxt.isInt64()):
                            data["track_id"] = nxt.asInt32(); i += 2
                        elif key in ("distance", "attention") and nxt.isString():
                            data[key] = nxt.asString(); i += 2
                        elif key in ("pitch", "yaw", "roll", "cos_angle", "time_in_view") and nxt.isFloat64():
                            data[key] = nxt.asFloat64(); i += 2
                        elif key == "is_talking" and (nxt.isInt32() or nxt.isInt64()):
                            data["is_talking"] = nxt.asInt32(); i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                elif item.isList():
                    nested = item.asList()
                    if nested.size() >= 2:
                        key = nested.get(0).asString() if nested.get(0).isString() else ""
                        if key == "bbox" and nested.size() >= 5:
                            data["bbox"] = (nested.get(1).asFloat64(), nested.get(2).asFloat64(),
                                            nested.get(3).asFloat64(), nested.get(4).asFloat64())
                        elif key == "gaze_direction" and nested.size() >= 4:
                            data["gaze_direction"] = (nested.get(1).asFloat64(),
                                                      nested.get(2).asFloat64(),
                                                      nested.get(3).asFloat64())
                    i += 1
                else:
                    i += 1
            return data
        except Exception as e:
            self._log("WARNING", f"Failed to parse face bottle: {e}")
            return None

    # ==================== Image Handling ====================

    def _read_image(self) -> Optional[np.ndarray]:
        yimg = self.img_in_port.read(False)
        if not yimg:
            return None
        w, h = yimg.width(), yimg.height()
        if w <= 0 or h <= 0:
            return None
        self.img_width, self.img_height = w, h
        try:
            try:
                img_bytes = yimg.toBytes()
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                expected = h * w * 3
                if len(arr) >= expected:
                    return arr[:expected].reshape((h, w, 3)).copy()
            except AttributeError:
                pass
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    pixel = yimg.pixel(x, y)
                    rgb[y, x] = [pixel.r, pixel.g, pixel.b]
            return rgb
        except Exception as e:
            self._log("WARNING", f"Failed to convert image: {e}")
            return None

    def _publish_image(self, frame_rgb: np.ndarray):
        if self.img_out_port.getOutputCount() == 0:
            return
        try:
            h, w, _ = frame_rgb.shape
            out = self.img_out_port.prepare()
            out.resize(w, h)
            try:
                out.fromBytes(frame_rgb.tobytes())
            except AttributeError:
                for y in range(h):
                    for x in range(w):
                        pixel = out.pixel(x, y)
                        r, g, b = frame_rgb[y, x]
                        pixel.r, pixel.g, pixel.b = int(r), int(g), int(b)
            self.img_out_port.write()
        except Exception as e:
            self._log("WARNING", f"Failed to publish image: {e}")

    def _annotate_image(self, frame_rgb: np.ndarray, faces: List[Dict],
                        selected: Optional[Dict]) -> np.ndarray:
        annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        selected_track_id = selected["track_id"] if selected else None
        selected_found = False

        for face in faces:
            x, y, w, h = face["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            x = max(0, min(x, self.img_width - 1))
            y = max(0, min(y, self.img_height - 1))
            w = max(0, min(w, self.img_width - x))
            h = max(0, min(h, self.img_height - y))

            track_id = face["track_id"]
            if track_id == selected_track_id and self.interaction_busy:
                color = self.COLOR_GREEN
                selected_found = True
                self.selected_bbox_last = face["bbox"]
            elif face.get("eligible", False):
                color = self.COLOR_YELLOW
            else:
                color = self.COLOR_WHITE

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            ss = face.get("social_state", "ss1")
            ls = face.get("learning_state", 1)
            person_id = face.get("person_id", face.get("face_id", "?"))
            display_id = person_id if person_id != "unknown" else face.get("face_id", "?")

            label1 = f"{display_id} (T:{track_id})"
            lg_ts = face.get("last_greeted_ts")
            lg_short = lg_ts[11:16] if lg_ts else "never"  # HH:MM or "never"
            label2 = f"{ss} | LS{ls} | LG:{lg_short}"
            label3 = f"{face.get('distance','?')}/{face.get('attention','?')[:3]}"
            area = face["bbox"][2] * face["bbox"][3]
            label4 = f"area={area:.0f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, th = 0.45, 1
            self._draw_label(annotated, label1, (x, y - 58), font, fs, color, th)
            self._draw_label(annotated, label2, (x, y - 41), font, fs, color, th)
            self._draw_label(annotated, label3, (x, y - 24), font, fs, color, th)
            self._draw_label(annotated, label4, (x, y - 7), font, fs, color, th)

            if track_id == selected_track_id and self.interaction_busy:
                cv2.putText(annotated, "ACTIVE", (x + w - 55, y + 15),
                            font, 0.5, self.COLOR_GREEN, 2)

        if self.interaction_busy and not selected_found and self.selected_bbox_last:
            x, y, w, h = [int(v) for v in self.selected_bbox_last]
            x = max(0, min(x, self.img_width - 1))
            y = max(0, min(y, self.img_height - 1))
            w = max(0, min(w, self.img_width - x))
            h = max(0, min(h, self.img_height - y))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), self.COLOR_GREEN, 2)
            cv2.putText(annotated, "ACTIVE (LOST)", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_GREEN, 2)

        status = "BUSY" if self.interaction_busy else "IDLE"
        status_color = self.COLOR_GREEN if self.interaction_busy else self.COLOR_WHITE
        cv2.putText(annotated, f"Status: {status} | Faces: {len(faces)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def _draw_label(self, img, text, pos, font, scale, color, thickness):
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 1, y), font, scale, color, thickness)

    # ==================== State Computation ====================

    def _compute_face_states(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        today = self._get_today_date()
        for face in faces:
            face_id = face["face_id"]
            track_id = face["track_id"]
            person_id = self.track_to_person.get(track_id, face_id) or face_id
            face["person_id"] = person_id

            is_known = self._is_face_known(face_id) or self._is_face_known(person_id)
            face["is_known"] = is_known

            # Read last_greeted info from background-refreshed snapshot
            lg_entry = self.get_last_greeted_entry(str(person_id), track_id=track_id)
            face["last_greeted_info"] = lg_entry  # None if never greeted
            face["last_greeted_ts"] = lg_entry.get("timestamp") if lg_entry else None

            greeted_today = self._was_greeted_today(person_id, today) if is_known else False
            talked_today = self._was_talked_today(person_id, today) if is_known else False
            face["greeted_today"] = greeted_today
            face["talked_today"] = talked_today

            face["social_state"] = self._compute_social_state(is_known, greeted_today, talked_today)
            face["learning_state"] = self._get_learning_state(person_id)
            face["eligible"] = self._is_eligible(face)

        # Prune stale track mappings
        active = {f["track_id"] for f in faces if f.get("track_id", -1) >= 0}
        self.track_to_person = {t: p for t, p in self.track_to_person.items() if t in active}
        return faces

    def _is_face_known(self, face_id: str) -> bool:
        if not face_id or face_id.lower() in ("unknown", "unmatched", "recognizing"):
            return False
        return True

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

    def _compute_social_state(self, is_known: bool, greeted_today: bool, talked_today: bool) -> str:
        """New 4-state model. Unknown people are always ss1."""
        if not is_known:
            return "ss1"
        if not greeted_today:
            return "ss2"
        if not talked_today:
            return "ss3"
        return "ss4"

    def _get_learning_state(self, person_id: str) -> int:
        if person_id in self.learning_data:
            return self.learning_data[person_id].get("ls", self.LS1)
        return self.LS1

    def _is_eligible(self, face: Dict[str, Any]) -> bool:
        ss = face.get("social_state", "ss1")
        ls = face.get("learning_state", self.LS1)
        # ss4 already at ultimate - not eligible for new interaction
        if ss == "ss4":
            return False

        if ls == self.LS3:
            return True

        distance = face.get("distance", "UNKNOWN")
        attention = face.get("attention", "AWAY")
        time_in_view = face.get("time_in_view", 0.0)

        if distance not in self.LS_VALID_DISTANCES.get(ls, set()):
            return False
        if attention not in self.LS_VALID_ATTENTIONS.get(ls, set()):
            return False
        min_tiv = self.LS_MIN_TIME_IN_VIEW.get(ls, 0.0)
        if time_in_view < min_tiv:
            return False
        return True

    # ==================== Face Selection (BIGGEST BBOX) ====================

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

    def _select_biggest_face(self, faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select face with biggest bbox area.

        POLICY: Always returns the biggest-bbox face regardless of face_id
        resolution status. The caller must check face_id resolution separately.
        Never filters by face_id. Returns None only if faces is empty or all
        have zero-area bboxes.
        """
        if not faces:
            return None
        biggest = max(faces, key=self._bbox_area)
        if self._bbox_area(biggest) <= 0:
            return None
        return biggest

    # ==================== Interaction Execution ====================

    def _run_interaction_thread(self, target: Dict[str, Any]):
        did_run_interaction = False
        try:
            track_id = target["track_id"]
            face_id  = str(target.get("person_id", target["face_id"]))
            ss       = target.get("social_state", "ss1")

            status = self._interaction_manager_status()
            if not isinstance(status, dict) or status.get("busy", False):
                self._log("INFO", "InteractionManager unavailable or busy – skipping proactive")
                return

            if ss == "ss4":
                self._log("INFO", f"{face_id} is ss4 – skipping")
                return

            self._log("INFO", f"--- START: {face_id} (track={track_id}, {ss}) ---")

            did_run_interaction = True
            self._execute_interaction_interface("ao_start")

            try:
                result = self._run_interaction_manager(track_id, face_id, ss)
                if result:
                    self._log("INFO", f"Result: success={result.get('success')}")
                    self._process_interaction_result(result, target)
                else:
                    self._log("WARNING", "No result from interactionManager")
            finally:
                self._execute_interaction_interface("ao_stop")

        except Exception as e:
            self._log("ERROR", f"Interaction thread error: {e}")
            import traceback; traceback.print_exc()
        finally:
            with self._interaction_lock:
                with self.state_lock:
                    self.interaction_busy = False
                    self.selected_target = None
                    self.selected_bbox_last = None
                    final_id = str(self.track_to_person.get(
                        target.get("track_id", -1),
                        target.get("face_id", "unknown")))
                    cd_key = final_id if self._is_face_known(final_id) else f"unknown:{target.get('track_id', -1)}"
                    if did_run_interaction:
                        self.last_interaction_time[cd_key] = time.time()
            self._log("INFO", "--- INTERACTION COMPLETE ---")

    def _prewarm_rpc_connections(self):
        """Send a no-op ping to both RPC servers so TCP setup happens at startup
        rather than on the first real interaction command."""
        time.sleep(1.0)  # brief wait for servers to register
        for attempt in range(5):
            try:
                if (self.interaction_manager_rpc and
                        self.interaction_manager_rpc.getOutputCount() > 0):
                    cmd = yarp.Bottle()
                    cmd.addString("status")
                    reply = yarp.Bottle()
                    self.interaction_manager_rpc.write(cmd, reply)
                    self._log("INFO", "InteractionManager RPC pre-warmed")
                    break
            except Exception as e:
                self._log("DEBUG", f"IM RPC pre-warm attempt {attempt + 1} failed: {e}")
                time.sleep(2.0)

        for attempt in range(5):
            try:
                if (self.interaction_interface_rpc and
                        self.interaction_interface_rpc.getOutputCount() > 0):
                    # interactionInterface doesn't have a status command — just open
                    # the connection by calling getOutputCount(); TCP is established
                    # when addOutput() was called, so the loop above already covers it.
                    self._log("INFO", "InteractionInterface RPC pre-warmed")
                    break
            except Exception as e:
                self._log("DEBUG", f"IF RPC pre-warm attempt {attempt + 1} failed: {e}")
                time.sleep(2.0)

    def _execute_interaction_interface(self, command: str) -> bool:
        try:
            if self.interaction_interface_rpc.getOutputCount() == 0:
                self._log("WARNING", "interactionInterface not connected")
                return False
            cmd = yarp.Bottle()
            cmd.addString("exe")
            cmd.addString(command)
            reply = yarp.Bottle()
            self.interaction_interface_rpc.write(cmd, reply)
            return True
        except Exception as e:
            self._log("ERROR", f"interactionInterface exception: {e}")
            return False

    def _run_interaction_manager(self, track_id: int, face_id: str, start_state: str) -> Optional[Dict]:
        try:
            if self.interaction_manager_rpc.getOutputCount() == 0:
                self._log("WARNING", "interactionManager not connected")
                return None
            cmd = yarp.Bottle()
            cmd.addString("run")
            cmd.addInt32(track_id)
            cmd.addString(face_id)
            cmd.addString(start_state)
            reply = yarp.Bottle()
            self._log("INFO", "RPC → interactionManager (waiting...)")
            if self.interaction_manager_rpc.write(cmd, reply):
                if reply.size() >= 2:
                    status   = reply.get(0).asString()
                    json_str = reply.get(1).asString()
                    if status == "ok":
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and parsed.get("error") == "responsive_interaction_running":
                                self._log("INFO", "interactionManager busy with responsive – skipping")
                                return None
                            return parsed
                        except json.JSONDecodeError as e:
                            self._log("ERROR", f"RPC JSON parse failed: {e}")
            return None
        except Exception as e:
            self._log("ERROR", f"interactionManager RPC error: {e}")
            return None

    def _interaction_manager_status(self) -> Optional[Dict]:
        try:
            if self.interaction_manager_rpc.getOutputCount() == 0:
                return None
            cmd = yarp.Bottle()
            cmd.addString("status")
            reply = yarp.Bottle()
            if not self.interaction_manager_rpc.write(cmd, reply):
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
            self._log("WARNING", f"interactionManager status check failed: {e}")
            return None

    def _process_interaction_result(self, result: Dict, target: Dict):
        """Update greeted/talked state, LS, and cooldown from interaction result."""
        try:
            extracted_name = result.get("name") if result.get("name_extracted") else None
            initial_ss = result.get("initial_state", "ss1")
            final_ss   = result.get("final_state", initial_ss)

            greeted = final_ss in ("ss3", "ss4") and initial_ss != "ss4"
            talked  = final_ss == "ss4"            and initial_ss != "ss4"

            track_id  = target["track_id"]
            person_id = str(target.get("person_id") or target.get("face_id") or "unknown")

            if extracted_name:
                person_id = str(extracted_name)
                with self.state_lock:
                    self.last_interaction_time[f"unknown:{track_id}"] = time.time()
                    self.last_interaction_time[person_id]               = time.time()

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
                self._log("INFO", f"Greeted: '{person_id}'")
            if talked:
                self._enqueue_save("talked")
                self._log("INFO", f"Talked:  '{person_id}'")

            delta = self._compute_reward(result, person_id)
            self._log("INFO", f"Reward delta = {delta:+d}  ({person_id})")
            self._update_learning_state(person_id, delta)

            if initial_ss != final_ss:
                self._db_log("ss_change", {
                    "person_id": person_id, "old_ss": initial_ss, "new_ss": final_ss,
                })

        except Exception as e:
            self._log("ERROR", f"process_interaction_result failed: {e}")
            import traceback; traceback.print_exc()

    def _face_disappear_penalty(self, person_id: str) -> int:
        now = time.time()
        with self.state_lock:
            events = self._disappear_events.get(person_id, [])
            # Prune old events outside window
            events = [ts for ts in events if now - ts <= self.DISAPPEAR_WINDOW_SEC]
            events.append(now)
            self._disappear_events[person_id] = events

            if len(events) >= self.DISAPPEAR_THRESHOLD:
                return -2
            return -1

    def _compute_reward(self, result: Dict, person_id: Optional[str] = None) -> int:
        """Compute reward delta from new compact format."""
        success = result.get("success", False)
        name_extracted = result.get("name_extracted", False)
        abort_reason = result.get("abort_reason")

        if success:
            return 2 if name_extracted else 1
        
        if abort_reason == "not_responded":
            return -1
        elif abort_reason == "face_disappeared":
            if person_id:
                return self._face_disappear_penalty(person_id)
            else:
                return -1
            
        return -1 # Default failure penalty

    def _update_learning_state(self, person_id: str, delta: int):
        if delta == 0:
            return

        with self._memory_lock:
            current_ls = self.learning_data.get(person_id, {}).get("ls", self.LS1)

        old_ls = current_ls
        if delta > 0:
            new_ls = min(3, current_ls + 1)
        elif delta < 0:
            new_ls = max(1, current_ls - 1)
        else:
            new_ls = current_ls

        now_iso = datetime.now(self.TIMEZONE).isoformat()
        with self._memory_lock:
            self.learning_data[person_id] = {"ls": new_ls, "updated_at": now_iso}

        self._enqueue_save("learning")

        if new_ls != old_ls:
            self._log("INFO", f"LS: '{person_id}' LS{old_ls} → LS{new_ls}")
            self._db_log("ls_change", {
                "person_id": person_id, "old_ls": old_ls,
                "new_ls": new_ls, "reward_delta": delta,
            })
        else:
            self._log("INFO", f"LS: '{person_id}' unchanged at LS{current_ls} (delta={delta:+d})")

    # ==================== Last Greeted (fresh read) ====================

    def _refresh_last_greeted_snapshot(self):
        """Load last_greeted.json safely into snapshot."""
        try:
            if self.last_greeted_path.exists():
                with open(self.last_greeted_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._last_greeted_lock:
                    self._last_greeted_snapshot = self._normalize_last_greeted_snapshot(data)
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
                ts = datetime.fromisoformat(entry_ts).timestamp() if entry_ts else min_ts
            except Exception:
                ts = min_ts

            prev_ts = latest_by_person_ts.get(str(key), min_ts)
            if ts >= prev_ts:
                latest_by_person[str(key)] = entry
                latest_by_person_ts[str(key)] = ts

        return latest_by_person

    def get_last_greeted_entry(self, person_id: str, track_id: Optional[int] = None) -> Optional[Dict]:
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
            self.talked_today  = self._load_json(self.talked_path,  {})
            learning_raw       = self._load_json(self.learning_path, {"people": {}})
            self.learning_data = learning_raw.get("people", {})

            # Clamp LS values to [1, 3]
            for pdata in self.learning_data.values():
                if "ls" in pdata:
                    pdata["ls"] = max(1, min(3, pdata["ls"]))

            self.greeted_today = self._prune_to_today(self.greeted_today)
            self.talked_today  = self._prune_to_today(self.talked_today)
        self._log("INFO",
            f"Loaded – greeted:{len(self.greeted_today)} "
            f"talked:{len(self.talked_today)} "
            f"learning:{len(self.learning_data)}")

    def _reload_memory_from_disk_and_prune_today(self):
        """Re-read memory JSON files from disk and prune greeted/talked to today.
        Called on day-change to pick up any writes that happened outside this process.
        """
        with self._memory_lock:
            self.greeted_today = self._load_json(self.greeted_path, {})
            self.talked_today  = self._load_json(self.talked_path,  {})
            learning_raw       = self._load_json(self.learning_path, {"people": {}})
            self.learning_data = learning_raw.get("people", {})

            for pdata in self.learning_data.values():
                if "ls" in pdata:
                    pdata["ls"] = max(1, min(3, pdata["ls"]))

            self.greeted_today = self._prune_to_today(self.greeted_today)
            self.talked_today  = self._prune_to_today(self.talked_today)
        self._log("INFO",
            f"Day-change reload – greeted:{len(self.greeted_today)} "
            f"talked:{len(self.talked_today)} "
            f"learning:{len(self.learning_data)}")

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
        # entries written by interactionManager's responsive path are never lost.
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
        self._io_queue.put(kind)

    def _io_worker(self):
        """Background thread draining IO save queue."""
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
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS target_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, track_id INTEGER, face_id TEXT,
                person_id TEXT, bbox_area REAL, ss TEXT, ls INTEGER, eligible INTEGER,
                last_greeted_ts TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS ss_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, person_id TEXT, old_ss TEXT, new_ss TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS ls_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, person_id TEXT, old_ls INTEGER,
                new_ls INTEGER, reward_delta INTEGER
            )""")
            conn.commit()
            conn.close()
            self._log("INFO", f"DB ready: {self.db_path}")
        except Exception as e:
            self._log("ERROR", f"DB init failed: {e}")

    def _db_log(self, table: str, data: Dict):
        data["timestamp"] = datetime.now(self.TIMEZONE).isoformat()
        self._db_queue.put((table, data))

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
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                c = conn.cursor()
                if table == "target_selection":
                    c.execute(
                        "INSERT INTO target_selections (timestamp,track_id,face_id,person_id,bbox_area,ss,ls,eligible,last_greeted_ts) VALUES (?,?,?,?,?,?,?,?,?)",
                        (data["timestamp"], data["track_id"], data["face_id"],
                         data["person_id"], data["bbox_area"], data["ss"],
                         data["ls"], int(data["eligible"]),
                         data.get("last_greeted_ts")))
                elif table == "ss_change":
                    c.execute(
                        "INSERT INTO ss_changes (timestamp,person_id,old_ss,new_ss) VALUES (?,?,?,?)",
                        (data["timestamp"], data["person_id"], data["old_ss"], data["new_ss"]))
                elif table == "ls_change":
                    c.execute(
                        "INSERT INTO ls_changes (timestamp,person_id,old_ls,new_ls,reward_delta) VALUES (?,?,?,?,?)",
                        (data["timestamp"], data["person_id"], data["old_ls"],
                         data["new_ls"], data["reward_delta"]))
                conn.commit()
                conn.close()
            except Exception as e:
                self._log("ERROR", f"DB write failed ({table}): {e}")

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

    module = FaceSelectorModule()
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("alwaysOn")
    rf.configure(sys.argv)

    print("=" * 55)
    print(" FaceSelectorModule – Biggest-BBox Face Selection")
    print("=" * 55)
    print(" ss1=unknown  ss2=known/not-greeted")
    print(" ss3=known/greeted/no-talk  ss4=ultimate")
    print()
    print(" yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i")
    print(" yarp connect /alwayson/vision/img:o       /faceSelector/img:i")
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
