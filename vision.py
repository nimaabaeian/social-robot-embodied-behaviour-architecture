import os
import sys
import subprocess
import threading
import urllib.request
import cv2
import numpy as np
import yarp
import time
from datetime import datetime
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import supervision as sv
from ultralytics import YOLO

import logging
import colorlog


def get_colored_logger(name):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("alwaysOn.log")])

    logger = logging.getLogger(name)
    colored_handler = colorlog.StreamHandler()
    colored_handler.setFormatter(colorlog.ColoredFormatter(
        '%(asctime)s %(log_color)s[%(name)s][%(levelname)s] %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple',
        },

    ))
    logger.addHandler(colored_handler)
    return logger

LANDMARK_IDS = [
    1,    # Nose tip
    199,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    291   # Right mouth corner
]

FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),   # Chin
    (-43.3, 32.7, -26.0),  # Left eye left corner
    (43.3, 32.7, -26.0),   # Right eye right corner
    (-28.9, -28.9, -24.1), # Left mouth corner
    (28.9, -28.9, -24.1)   # Right mouth corner
], dtype=np.float64)


class VisionAnalyzer(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)

        self.logger = get_colored_logger("Video Features")
        self.rate = 0.05

        self.img_in_port = yarp.BufferedPortImageRgb()              # Raw images
        self.vision_features_port = yarp.Port()                     # Output Features
        self.landmarks_port = yarp.Port()                           # Per-face detailed information

        # --- Target command from salienceNetwork (lightweight: [track_id, ips_score]) ---
        self.target_cmd_port = yarp.BufferedPortBottle()
        # --- Target box output to FaceTracker (full bounding box bottle) ---
        self.target_box_port = yarp.BufferedPortBottle()

        self.img_in_btl = yarp.ImageRgb()
        self.vision_features_btl = yarp.Bottle()
        self.landmarks_btl = yarp.Bottle()

        self.name = "alwayson/vision"
        self.img_width = 640
        self.img_height = 480
        self.default_width = 640
        self.default_height = 480
        self.input_img_array = None
        self.image = None
        self.opt_flow_buf = deque()
        self.timestamp = None

        self.env_dict = {
            "Faces": 0,
            "People": 0,
            "Motion": 0.0,
            "Light": 0.0,
            "MutualGaze": 0,
        }

        # To make information stable and not instantaneously change
        self.faces_sync_info = 0
        self.exec_time = 0.15  # Execution mean time for the module

        self.mutual_gaze_threshold = 10
        self.max_face_match_distance = 100.0  # Max distance (pixels) for matching MediaPipe to bbox

        self.face_mesh = None
        self.detected_faces = []  # Store face data from YOLO/ByteTrack (bbox, face_id)
        
        # Talking detection based on lip motion
        self.mouth_motion_history = {}  # dict[track_id] -> deque of normalized mouth_open values
        self.mouth_buffer_size = 10  # Number of frames to track for motion detection
        self.talking_threshold = 0.012  # Std threshold for mouth motion (tunable: increase to reduce false positives)
        self.last_seen_track = {}  # dict[track_id] -> timestamp for cleanup
        self.first_seen_track = {}  # dict[track_id] -> timestamp when first seen


        self.default_face_model_url = (
            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
        )
        self.default_face_model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "yolov11n-face.pt",
        )
        self.yolo_model = None
        self.byte_tracker = None
        self.conf_threshold = 0.7
        self.track = True
        self.identify_faces = True
        self.tolerance = 0.62
        self.verbose = False
        self.debug = False
        self.identity_sticky_sec = 1.5
        self.max_enroll_samples = 5
        self.unknown_retry_interval_sec = 2.0
        self.unknown_retry_max_attempts = 3

        self.known_faces = {}
        self.tracked_faces = {}
        self.last_known_identity = {}  # dict[track_id] -> (name, confidence, timestamp)
        self.unknown_retry_state = {}  # dict[track_id] -> (attempt_count, last_retry_ts)
        self._face_identity_lock = threading.Lock()
        self.objects = []
        self.last_frame = None
        self.faces_path = None

        self.face_recognition_available = False
        self.face_recognition = None
        self.auto_download_model = True
        self.fallback_models = ["yolov8n-face.pt"]
        self._warned_non_face_model = False

        # RPC port for face naming commands
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Annotated face detection image output
        self.face_detection_img_port = yarp.BufferedPortImageRgb()
        self.display_buf_image = yarp.ImageRgb()
        self._display_rgb_buffer = None

        # --- Target command state ---
        self._current_target_track_id = -1
        self._current_target_ips = 0.0

        # QR state tracking (emit once per appearance)
        self._active_qr_value = None
        self._qr_missing_scans = 0
        self._qr_lost_reset_scans = 3

    # ==================== face identity methods ====================

    def _pip_install(self, package: str) -> bool:
        python_exec = sys.executable

        print(f"[INFO] Installing {package} using {python_exec}")

        try:
            subprocess.run(
                [python_exec, "-m", "pip", "install", package],
                check=True,
            )
            return True
        except Exception as err:
            print(f"\033[93m[WARNING] Failed to install {package}: {err}\033[00m")
            return False

    def _install_face_recognition_stack(self, install_face_lib: bool, install_models: bool) -> bool:
        ok = True

        ok = self._pip_install("setuptools<70.0.0") and ok

        if install_face_lib:
            ok = self._pip_install("face-recognition") and ok

        if install_models:
            ok = self._pip_install("git+https://github.com/ageitgey/face_recognition_models") and ok

        return ok

    def _initialize_face_recognition(self):
        error_text = ""
        missing_models = False
        missing_face_library = False

        try:
            import face_recognition as fr

            self.face_recognition_available = True
            self.face_recognition = fr
            return
        except (Exception, SystemExit) as err:
            error_text = str(err) if err is not None else ""
            err_lower = error_text.lower()
            missing_models = isinstance(err, SystemExit) or (
                "face_recognition_models" in err_lower
            )
            missing_face_library = (
                "no module named 'face_recognition'" in err_lower
                or "no module named \"face_recognition\"" in err_lower
            )
            self.face_recognition_available = False
            print(f"\033[93m[WARNING] face_recognition unavailable: {error_text}\033[00m")

        if not (missing_face_library or missing_models):
            return

        print("\033[93m[WARNING] Attempting auto-install for face recognition dependencies...\033[00m")

        if not self._install_face_recognition_stack(
            install_face_lib=missing_face_library,
            install_models=missing_models or missing_face_library,
        ):
            return

        try:
            import face_recognition as fr

            self.face_recognition_available = True
            self.face_recognition = fr
            print("[INFO] face_recognition_models installed successfully")
        except (Exception, SystemExit) as err:
            self.face_recognition_available = False
            print(f"\033[93m[WARNING] face_recognition still unavailable after install: {err}\033[00m")

    def _resolve_model_path(self, model_path: str):
        candidates = []

        if os.path.isabs(model_path):
            candidates.append(model_path)
        else:
            current_script_folder = os.path.dirname(os.path.abspath(__file__))
            candidates.extend(
                [
                    model_path,
                    os.path.join(current_script_folder, model_path),
                    os.path.join(current_script_folder, "model", model_path),
                    os.path.join(os.getcwd(), model_path),
                ]
            )

        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return None

    def _build_model_candidates(self, model_path: str, fallback_models_cfg: str):
        candidates = [model_path]

        if fallback_models_cfg:
            fallback_models = [m.strip() for m in fallback_models_cfg.split(",") if m.strip()]
            for fallback in fallback_models:
                if fallback not in candidates:
                    candidates.append(fallback)

        return candidates

    def _filter_face_detections(self, detections, names):
        if len(detections) == 0 or detections.class_id is None:
            return detections

        face_class_ids = []
        if isinstance(names, dict):
            face_class_ids = [class_id for class_id, label in names.items() if str(label).lower() == "face"]
        elif isinstance(names, list):
            face_class_ids = [idx for idx, label in enumerate(names) if str(label).lower() == "face"]

        if not face_class_ids:
            if not self._warned_non_face_model:
                print(
                    "\033[93m[WARNING] Loaded model has no 'face' class label. "
                    "Dropping all detections to enforce face-only output. "
                    "Use a face model for valid face boxes.\033[00m"
                )
                self._warned_non_face_model = True
            return detections[np.zeros(len(detections), dtype=bool)]

        return detections[np.isin(detections.class_id, face_class_ids)]

    def _initialize_yolo_model(self, model_candidates):
        last_error = None
        perception_dir = os.path.dirname(os.path.abspath(__file__))

        for candidate in model_candidates:
            resolved_path = self._resolve_model_path(candidate)

            if (
                resolved_path is None
                and os.path.isabs(candidate)
                and candidate == self.default_face_model
                and self.auto_download_model
            ):
                if self._download_default_face_model(candidate):
                    resolved_path = candidate

            if resolved_path is None and self.auto_download_model:
                candidate_basename = os.path.basename(candidate)
                if candidate.startswith("http://") or candidate.startswith("https://"):
                    local_candidate = os.path.join(perception_dir, candidate_basename)
                    if self._download_default_face_model(local_candidate):
                        resolved_path = local_candidate
                elif candidate_basename == os.path.basename(self.default_face_model):
                    local_candidate = os.path.join(perception_dir, candidate_basename)
                    if self._download_default_face_model(local_candidate):
                        resolved_path = local_candidate

            if resolved_path is not None:
                model_source = resolved_path
            else:
                print(
                    "\033[93m[WARNING] Model candidate not found locally and remote loading is disabled: "
                    f"{candidate}\033[00m"
                )
                continue

            try:
                self.yolo_model = YOLO(model_source)
                print(f"[INFO] Loaded YOLO model: {resolved_path}")
                return True
            except Exception as err:
                last_error = err
                print(
                    f"\033[93m[WARNING] Failed to load model candidate {candidate}: {err}\033[00m"
                )

        candidates_text = ", ".join(model_candidates)
        print(
            "\033[91m[ERROR] Could not load any YOLO model candidate: "
            f"{candidates_text}\033[00m"
        )
        if last_error is not None:
            print(f"\033[91m[ERROR] Last YOLO load error: {last_error}\033[00m")
        return False

    def _download_default_face_model(self, local_path: str) -> bool:
        try:
            parent_dir = os.path.dirname(local_path)
            os.makedirs(parent_dir, exist_ok=True)
            print(f"[INFO] Downloading default face model to: {local_path}")
            urllib.request.urlretrieve(self.default_face_model_url, local_path)
            return True
        except Exception as err:
            print(
                "\033[93m[WARNING] Failed to download default face model to "
                f"{local_path}: {err}\033[00m"
            )
            return False

    def _load_known_faces(self, faces_path: str):
        database = {}
        if not os.path.exists(faces_path):
            print(f"\033[93m[WARNING] Faces folder does not exist: {faces_path}\033[00m")
            return database

        for filename in sorted(os.listdir(faces_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            file_path = os.path.join(faces_path, filename)
            person_name = os.path.splitext(filename)[0]
            if person_name in database:
                print(
                    f"\033[93m[WARNING] Duplicate face identity basename '{person_name}' in {faces_path}. "
                    f"Keeping first file and skipping {filename}.\033[00m"
                )
                continue
            image = self.face_recognition.load_image_file(file_path)
            encoding = self.face_recognition.face_encodings(image, model="large")
            if len(encoding) > 0:
                database[person_name] = [encoding[0]]

        return database

    def _compare_embeddings(self, frame_bgr, box):
        if not self.face_recognition_available:
            return "unknown", 0.0

        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame_bgr.shape[:2]

        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return "recognizing", 0.0

        cropped_frame = frame_bgr[y1:y2, x1:x2]
        if cropped_frame.size == 0:
            return "recognizing", 0.0

        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        unknown_encoding = self.face_recognition.face_encodings(
            np.array(cropped_rgb), model="large"
        )
        if len(unknown_encoding) == 0:
            return "recognizing", 0.0

        with self._face_identity_lock:
            known_faces_snapshot = {
                name: samples[:] if isinstance(samples, list) else samples
                for name, samples in self.known_faces.items()
            }

        if not known_faces_snapshot:
            return "unknown", 0.0

        unknown_encoding = unknown_encoding[0]
        best_name = None
        best_effective_distance = float("inf")

        for name, stored_samples in known_faces_snapshot.items():
            if isinstance(stored_samples, np.ndarray):
                sample_list = [stored_samples]
            elif isinstance(stored_samples, list):
                sample_list = stored_samples
            else:
                continue

            if len(sample_list) == 0:
                continue

            sample_arr = np.array(sample_list)
            distances = self.face_recognition.face_distance(sample_arr, unknown_encoding)
            if len(distances) == 0:
                continue

            min_distance = float(np.min(distances))
            median_distance = float(np.median(distances))
            effective_distance = 0.5 * (min_distance + median_distance)

            # Gate by best sample, rank by min+median robustness.
            if min_distance <= self.tolerance and effective_distance < best_effective_distance:
                best_effective_distance = effective_distance
                best_name = name

        if best_name is not None:
            confidence = 1.0 - best_effective_distance
            return best_name, confidence

        return "unknown", 0.0

    def _handle_face_naming(self, command, reply):
        """Handle runtime face naming command: name <person_name> id <track_id>."""
        reply.clear()

        if command.size() != 4:
            reply.addString("nack")
            reply.addString("Usage: name <person_name> id <track_id>")
            return

        if command.get(0).asString() != "name" or command.get(2).asString() != "id":
            reply.addString("nack")
            reply.addString("Usage: name <person_name> id <track_id>")
            return

        if not self.track:
            reply.addString("nack")
            reply.addString("Face naming requires --track true")
            return

        if not self.identify_faces:
            reply.addString("nack")
            reply.addString("Face naming requires --identify_faces true")
            return

        if not self.face_recognition_available:
            reply.addString("nack")
            reply.addString("face_recognition library is not available")
            return

        if self.last_frame is None:
            reply.addString("nack")
            reply.addString("No frame available yet")
            return

        person_name = command.get(1).asString()
        track_id = command.get(3).asInt32()

        target_obj = None
        for obj in self.objects:
            if "track_id" in obj and obj["track_id"] == track_id:
                target_obj = obj
                break

        if target_obj is None:
            reply.addString("nack")
            reply.addString(f"Track ID {track_id} not found in current detections")
            return

        box = target_obj["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        h, w = self.last_frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            reply.addString("nack")
            reply.addString("Invalid bounding box")
            return

        face_crop = self.last_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            reply.addString("nack")
            reply.addString("Empty face crop")
            return

        if self.faces_path is None:
            reply.addString("nack")
            reply.addString("Faces path is not configured")
            return

        os.makedirs(self.faces_path, exist_ok=True)
        face_path = os.path.join(self.faces_path, f"{person_name}.jpg")
        cv2.imwrite(face_path, face_crop)

        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encoding = self.face_recognition.face_encodings(face_crop_rgb, model="large")

        if len(encoding) == 0:
            reply.addString("nack")
            reply.addString("Could not extract face encoding from crop")
            return

        with self._face_identity_lock:
            existing_samples = self.known_faces.get(person_name)
            if existing_samples is None:
                self.known_faces[person_name] = [encoding[0]]
            elif isinstance(existing_samples, list):
                existing_samples.append(encoding[0])
                if len(existing_samples) > self.max_enroll_samples:
                    existing_samples = existing_samples[-self.max_enroll_samples:]
                self.known_faces[person_name] = existing_samples
            else:
                self.known_faces[person_name] = [existing_samples, encoding[0]][-self.max_enroll_samples:]

            self.tracked_faces[track_id] = (person_name, 1.0)
            self.last_known_identity[track_id] = (person_name, 1.0, time.time())

        reply.addString("ok")
        reply.addString(face_path)

    def respond(self, command, reply):
        reply.clear()
        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False
        if command.get(0).asString() == "process":
            if command.size() < 2:
                reply.addString("nack")
                reply.addString("Usage: process on/off")
                return True
            # process flag not used in merged module, but keep for compat
            reply.addString("ok")
            return True
        if command.get(0).asString() == "name":
            self._handle_face_naming(command, reply)
            return True
        if command.get(0).asString() == "help":
            reply.addString("Commands: quit | process on/off | name <person_name> id <track_id>")
            return True

        reply.addString("nack")
        return True

    # ==================== Original VisionAnalyzer methods ====================

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        self.name = rf.check("name", yarp.Value(self.name)).asString()
        self.rate = rf.check("rate", yarp.Value("0.05")).asFloat64()
        self.img_width = rf.check("width", yarp.Value(640)).asInt64()
        self.img_height = rf.check("height", yarp.Value(480)).asInt64()
        self.landmark_model_path = rf.check("model", yarp.Value("face_landmarker.task")).asString()

        print(f"IMAGE W: {self.img_width}")
        print(f"IMAGE H: {self.img_height}")
        print(f"RATE: {self.rate}")
        print(f"MODEL PATH: {self.landmark_model_path}")
        
        # Use ResourceFinder to locate the model file in the context directory
        model_full_path = rf.findFileByName(self.landmark_model_path)
        if not model_full_path:
            self.logger.error(f"Could not find model file: {self.landmark_model_path}")
            return False
        print(f"MODEL FULL PATH: {model_full_path}")

        self.img_in_port.open(f'/{self.name}/img:i')
        self.vision_features_port.open(f'/{self.name}/features:o')
        self.landmarks_port.open(f'/{self.name}/landmarks:o')
        self.target_cmd_port.open(f'/{self.name}/targetCmd:i')
        self.target_box_port.open(f'/{self.name}/targetBox:o')

        self.input_img_array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        base_options = python.BaseOptions(
            model_asset_path=model_full_path
        )


        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,

        )

        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

        # --- face identity init ---
        self.conf_threshold = rf.check("conf_threshold", yarp.Value(0.7)).asFloat32()
        self.track = rf.check("track", yarp.Value(True)).asBool()
        self.identify_faces = rf.check("identify_faces", yarp.Value(True)).asBool()
        self.tolerance = rf.check("id_tolerance", yarp.Value(0.62)).asFloat32()
        self.identity_sticky_sec = rf.check("identity_sticky_sec", yarp.Value(1.5)).asFloat64()
        self.max_enroll_samples = rf.check("id_enroll_samples", yarp.Value(5)).asInt64()
        self.unknown_retry_interval_sec = rf.check("unknown_retry_interval_sec", yarp.Value(1.0)).asFloat64()
        self.unknown_retry_max_attempts = rf.check("unknown_retry_max_attempts", yarp.Value(3)).asInt64()
        self.verbose = rf.check("verbose_yolo", yarp.Value(False)).asBool()
        self.debug = rf.check("debug", yarp.Value(False)).asBool()
        self.auto_download_model = rf.check("auto_download_model", yarp.Value(True)).asBool()

        current_script_folder = os.path.dirname(os.path.abspath(__file__))
        default_faces_path = os.path.abspath(
            os.path.join(current_script_folder, "faces")
        )
        faces_path = rf.check("faces_path", yarp.Value(default_faces_path)).asString()
        self.faces_path = faces_path

        yolo_model_path = rf.check("yolo_model", yarp.Value(self.default_face_model)).asString()
        fallback_models_cfg = rf.check(
            "fallback_models",
            yarp.Value(",".join(self.fallback_models)),
        ).asString()
        model_candidates = self._build_model_candidates(yolo_model_path, fallback_models_cfg)

        self._initialize_face_recognition()

        if not self._initialize_yolo_model(model_candidates):
            return False

        if self.track:
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=self.conf_threshold,
                lost_track_buffer=120,
            )

        if self.identify_faces:
            if not self.face_recognition_available:
                print("\033[93m[WARNING] identify_faces requested but face_recognition is unavailable\033[00m")
            else:
                self.known_faces = self._load_known_faces(faces_path)
                print(
                    f"[INFO] Face ID enabled with {len(self.known_faces)} identities from {faces_path}"
                )

        # RPC port for face naming
        rpc_name = rf.check("rpc_name", yarp.Value(f"{self.name}/rpc")).asString()
        self.handle_port.open("/" + rpc_name)

        # Annotated face detection image output
        self.face_detection_img_port.open(f'/{self.name}/faces_view:o')
        self.display_buf_image.resize(self.img_width, self.img_height)

        self.qr_detector = cv2.QRCodeDetector()
        self.qr_port = yarp.BufferedPortBottle()
        self.qr_port.open(f'/{self.name}/qr:o')
        self._qr_seen_frames = 0
        self._qr_lost_reset_scans = rf.check(
            "qr_lost_reset_scans",
            yarp.Value(3),
        ).asInt64()

        self.logger.info("Start processing video (vision monolith)")
        return True

    def getPeriod(self):
        return self.rate

    def updateModule(self):
        self.vision_features_btl.clear()
        self.landmarks_btl.clear()
        has_features_subscriber = self.vision_features_port.getOutputCount() > 0
        has_landmarks_subscriber = self.landmarks_port.getOutputCount() > 0
        has_target_subscriber = self.target_box_port.getOutputCount() > 0
        has_view_subscriber = self.face_detection_img_port.getOutputCount() > 0
        has_qr_subscriber = self.qr_port.getOutputCount() > 0

        if has_features_subscriber or has_landmarks_subscriber or has_target_subscriber or has_view_subscriber or has_qr_subscriber:
            self.img_in_btl = self.img_in_port.read(shouldWait=True)
            if self.img_in_btl:
                self.image = self.__img_yarp_to_cv(self.img_in_btl)
                if has_qr_subscriber:
                    self.detect_qr_codes()
                self.detect_people_obj()        # Run YOLO/ByteTrack directly on self.image
                self.detect_mutual_gaze()       # Count # people looking at the camera and publish per-face details
                self.detect_light()             # Extract from a HSV space, the V component of the image
                self.detect_motion()            # Only presence (no magnitude or orientation)
                if has_view_subscriber:
                    self.draw_and_publish_faces_view()
            self.timestamp = datetime.now().timestamp()

            if has_features_subscriber:
                self.fill_bottle()
                self.vision_features_port.write(self.vision_features_btl)

            if has_landmarks_subscriber:
                self.landmarks_port.write(self.landmarks_btl)

        # --- Target delegation: read command from salienceNetwork, stream bbox to FaceTracker ---
        self._handle_target_command()

        return True

    def detect_qr_codes(self):
        """Throttled QR code detector using OpenCV."""
        if self.image is None:
            return
            
        # Throttling to save CPU: only check 1 in 10 frames
        self._qr_seen_frames += 1
        if self._qr_seen_frames % 10 != 0:
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        raw_val, pts, _ = self.qr_detector.detectAndDecode(gray)

        # No valid QR in this sampled frame: count misses and clear active state
        # after a small hysteresis window to avoid flicker-triggered re-emits.
        if not raw_val:
            self._qr_missing_scans += 1
            if self._qr_missing_scans >= self._qr_lost_reset_scans:
                self._active_qr_value = None
            return

        if raw_val:
            val = raw_val.strip().upper()
            if val:
                # QR is visible in this sampled frame, reset miss counter.
                self._qr_missing_scans = 0

                # Already active in view: suppress repeats.
                if val == self._active_qr_value:
                    return

                btl = self.qr_port.prepare()
                btl.clear()
                btl.addString(val)
                self.qr_port.write()
                self._active_qr_value = val

    def detect_people_obj(self):
        """Run YOLO face detection + ByteTrack + face_recognition directly on self.image."""
        if self.image is None or self.yolo_model is None:
            return

        # Convert to BGR for YOLO (self.image is already BGR from __img_yarp_to_cv)
        frame = self.image.copy()
        self.last_frame = frame.copy()
        current_time = time.time()

        result = self.yolo_model(frame, verbose=self.verbose)[0]
        detections = sv.Detections.from_ultralytics(result)

        if self.conf_threshold:
            detections = detections[detections.confidence > self.conf_threshold]

        detections = self._filter_face_detections(detections, result.names)

        # Expand bounding boxes for face crops
        if len(detections) > 0:
            xyxy = detections.xyxy
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]

            # 10% expansion per side
            expand_w = widths * 0.10
            expand_h = heights * 0.10

            xyxy[:, 0] -= expand_w
            xyxy[:, 1] -= expand_h
            xyxy[:, 2] += expand_w
            xyxy[:, 3] += expand_h

            xyxy[:, 0] = np.maximum(xyxy[:, 0], 0)
            xyxy[:, 1] = np.maximum(xyxy[:, 1], 0)
            xyxy[:, 2] = np.minimum(xyxy[:, 2], frame.shape[1])
            xyxy[:, 3] = np.minimum(xyxy[:, 3], frame.shape[0])

            detections.xyxy = xyxy

        if self.track and len(detections) > 0:
            detections = self.byte_tracker.update_with_detections(detections)

        labels = [result.names[class_id] for class_id in detections.class_id] if len(detections) > 0 else []

        if self.identify_faces and self.track and len(detections) > 0 and detections.tracker_id is not None:
            current_ids = set(detections.tracker_id)
            with self._face_identity_lock:
                lost_ids = [tid for tid in list(self.tracked_faces.keys()) if tid not in current_ids]
                for tid in lost_ids:
                    cached_entry = self.tracked_faces.get(tid)
                    if cached_entry is not None:
                        cached_name, cached_conf = cached_entry
                        if cached_name not in ("recognizing", "unknown"):
                            self.last_known_identity[tid] = (cached_name, float(cached_conf), current_time)
                    del self.tracked_faces[tid]
                    if tid in self.unknown_retry_state:
                        del self.unknown_retry_state[tid]

                stale_ids = [
                    tid
                    for tid, (_, _, ts) in self.last_known_identity.items()
                    if current_time - ts > self.identity_sticky_sec
                ]
                for tid in stale_ids:
                    del self.last_known_identity[tid]

            for tid, box in zip(detections.tracker_id, detections.xyxy):
                with self._face_identity_lock:
                    tracked_entry = self.tracked_faces.get(tid)

                if tracked_entry is None:
                    face_id, id_conf = self._compare_embeddings(frame, box)
                    if face_id in ("unknown", "recognizing"):
                        with self._face_identity_lock:
                            sticky_entry = self.last_known_identity.get(tid)
                        if sticky_entry is not None:
                            sticky_name, sticky_conf, sticky_ts = sticky_entry
                            if current_time - sticky_ts <= self.identity_sticky_sec:
                                face_id, id_conf = sticky_name, sticky_conf

                    with self._face_identity_lock:
                        if face_id in ("unknown", "recognizing"):
                            # Publish unknown while retrying in the background on a timer.
                            self.tracked_faces[tid] = ("unknown", 0.0)
                            self.unknown_retry_state[tid] = (1, current_time)
                        else:
                            self.tracked_faces[tid] = (face_id, id_conf)
                            self.last_known_identity[tid] = (face_id, float(id_conf), current_time)
                            if tid in self.unknown_retry_state:
                                del self.unknown_retry_state[tid]
                else:
                    cached_name, retry_meta = tracked_entry
                    if cached_name in ("recognizing", "unknown"):
                        with self._face_identity_lock:
                            attempts, last_try_ts = self.unknown_retry_state.get(tid, (0, 0.0))

                        should_retry = (
                            attempts < self.unknown_retry_max_attempts
                            and (current_time - last_try_ts) >= self.unknown_retry_interval_sec
                        )

                        if should_retry:
                            face_id, id_conf = self._compare_embeddings(frame, box)

                            if face_id in ("unknown", "recognizing"):
                                with self._face_identity_lock:
                                    self.tracked_faces[tid] = ("unknown", 0.0)
                                    self.unknown_retry_state[tid] = (attempts + 1, current_time)
                            else:
                                with self._face_identity_lock:
                                    self.tracked_faces[tid] = (face_id, id_conf)
                                    self.last_known_identity[tid] = (face_id, float(id_conf), current_time)
                                    if tid in self.unknown_retry_state:
                                        del self.unknown_retry_state[tid]
                        else:
                            # Keep publishing unknown, without transient recognizing labels.
                            with self._face_identity_lock:
                                self.tracked_faces[tid] = ("unknown", 0.0)
                    elif cached_name != "unknown":
                        with self._face_identity_lock:
                            self.last_known_identity[tid] = (cached_name, float(retry_meta), current_time)

        objects = []
        for i in range(len(detections)):
            obj = {
                "class": labels[i],
                "score": float(detections.confidence[i]),
                "box": detections.xyxy[i].tolist(),
            }

            if self.track and detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
                if track_id is not None:
                    obj["track_id"] = int(track_id)
                    with self._face_identity_lock:
                        tracked_entry = self.tracked_faces.get(track_id)
                    if self.identify_faces and tracked_entry is not None:
                        face_id, id_conf = tracked_entry
                        obj["face_id"] = face_id
                        obj["id_confidence"] = float(id_conf)

            objects.append(obj)

        self.objects = objects

        # Populate self.detected_faces for MediaPipe matching (same format as before)
        self.detected_faces = []
        num_faces = 0

        for obj in objects:
            if obj["class"] == "face" and obj["score"] > 0.5:
                num_faces += 1
                box = obj["box"]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                self.detected_faces.append({
                    'face_id': obj.get('face_id', 'unknown'),
                    'track_id': obj.get('track_id', -1),
                    'bbox': (x, y, w, h),
                    'detection_score': obj['score'],
                    'id_confidence': obj.get('id_confidence', 0.0)
                })

        self.env_dict["Faces"] = num_faces
        self.faces_sync_info = time.time()

        # Save detections and labels for drawing later (after MediaPipe fields are computed)
        self.current_detections = detections
        self.current_labels = labels

    def draw_and_publish_faces_view(self):
        if self.face_detection_img_port.getOutputCount() == 0 or self.last_frame is None:
            return

        frame = self.last_frame.copy()
        detections = getattr(self, 'current_detections', None)
        labels = getattr(self, 'current_labels', [])

        if detections is None:
            self._write_annotated_image(frame)
            return

        display_labels = labels.copy()
        if self.track and len(detections) > 0 and detections.tracker_id is not None:
            display_labels = []
            for label, track_id in zip(labels, detections.tracker_id):
                name_str = label
                if self.identify_faces:
                    with self._face_identity_lock:
                        tracked_entry = self.tracked_faces.get(track_id)
                    if tracked_entry is not None:
                        face_id, _ = tracked_entry
                        if face_id != "recognizing":
                            name_str = str(face_id)
                        else:
                            name_str = "recognizing..."

                # Fetch MediaPipe attrs from self.detected_faces
                attn_str = ""
                dist_str = ""
                zone_str = ""
                for fd in self.detected_faces:
                    if fd.get('track_id') == track_id:
                        attn_str = fd.get('attention', '')
                        dist_str = fd.get('distance', '')
                        zone_str = fd.get('zone', '')
                        break

                # Build custom string (name | attention | distance | zone)
                full_label = f"id:{int(track_id)} | {name_str}"
                if attn_str and attn_str != "UNKNOWN":
                    full_label += f" | {attn_str}"
                if dist_str and dist_str != "UNKNOWN":
                    full_label += f" | {dist_str}"
                if zone_str and zone_str != "UNKNOWN":
                    full_label += f" | {zone_str}"

                display_labels.append(full_label)

        if len(detections) > 0:
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
            annotated_image = box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=display_labels,
            )
        else:
            annotated_image = frame

        self._write_annotated_image(annotated_image)

    def _write_annotated_image(self, annotated_image_bgr):
        annotated_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        h, w = annotated_rgb.shape[:2]
        self._display_rgb_buffer = np.ascontiguousarray(annotated_rgb)
        self.display_buf_image = self.face_detection_img_port.prepare()
        self.display_buf_image.resize(w, h)
        self.display_buf_image.setExternal(
            self._display_rgb_buffer.data,
            w,
            h,
        )
        self.face_detection_img_port.write()

    # ==================== Target Command Handling ====================

    def _handle_target_command(self):
        """Read lightweight [track_id, ips_score] from salienceNetwork, find bbox, stream to FaceTracker."""
        # Drain queue to get the freshest command (eliminate lag if commands buffer)
        while True:
            cmd_btl = self.target_cmd_port.read(shouldWait=False)
            if not cmd_btl:
                break
            if cmd_btl.size() >= 2:
                self._current_target_track_id = cmd_btl.get(0).asInt32()
                self._current_target_ips = cmd_btl.get(1).asFloat64()

        # If no valid target, do nothing
        if self._current_target_track_id < 0:
            return

        if self.target_box_port.getOutputCount() == 0:
            return

        # Find the matching bbox in self.detected_faces
        target_face = None
        for face_data in self.detected_faces:
            if face_data['track_id'] == self._current_target_track_id:
                target_face = face_data
                break

        if target_face is None:
            return

        # Construct the exact YARP Bottle format used by the old salienceNetwork:
        # ((class face) (score <ips_score>) (box (<xmin> <ymin> <xmax> <ymax>)))
        x, y, w, h = target_face['bbox']
        x_min, y_min = float(x), float(y)
        x_max, y_max = float(x + w), float(y + h)

        bottle = self.target_box_port.prepare()
        bottle.clear()

        obj_bottle = yarp.Bottle()

        # 1. ("class" "face")
        class_btl = yarp.Bottle()
        class_btl.addString("class")
        class_btl.addString("face")
        obj_bottle.addList().read(class_btl)

        # 2. ("score" ips)
        score_btl = yarp.Bottle()
        score_btl.addString("score")
        score_btl.addFloat64(self._current_target_ips)
        obj_bottle.addList().read(score_btl)

        # 3. ("box" (xmin ymin xmax ymax))
        box_btl = yarp.Bottle()
        box_btl.addString("box")
        coords_btl = yarp.Bottle()
        coords_btl.addFloat64(x_min)
        coords_btl.addFloat64(y_min)
        coords_btl.addFloat64(x_max)
        coords_btl.addFloat64(y_max)

        box_btl.addList().read(coords_btl)
        obj_bottle.addList().read(box_btl)

        bottle.addList().read(obj_bottle)

        self.target_box_port.write()

    # ==================== Original methods (unchanged) ====================

    def detect_light(self):
        if self.image.mean() != 0.0:
            self.opt_flow_add_img(self.image)
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(image)
        cv2.normalize(v, v, 0, 1.0, cv2.NORM_MINMAX)
        self.env_dict["Light"] = round(v.mean(), 2)


    def detect_mutual_gaze(self):

        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        results = self.face_mesh.detect(mp_image)
        img_h, img_w, img_c = self.image.shape

        self.env_dict['MutualGaze'] = 0
        
        # Track which bboxes have been matched (one-to-one matching)
        matched_track_ids = set()
        current_time = time.time()

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:

                face_2d = []

                for idx in LANDMARK_IDS:
                    lm = face_landmarks[idx]
                    x = lm.x * img_w
                    y = lm.y * img_h
                    face_2d.append([x, y])

                face_2d = np.array(face_2d, dtype=np.float64)

                # Calculate face center from landmarks for matching with bboxes
                face_center_x = np.mean(face_2d[:, 0])
                face_center_y = np.mean(face_2d[:, 1])

                focal_length = img_w
                cam_matrix = np.array([
                    [focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_coeffs = np.zeros((4, 1))

                success, rvec, tvec = cv2.solvePnP(
                    FACE_3D_MODEL,
                    face_2d,
                    cam_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not success:
                    continue

                rmat, _ = cv2.Rodrigues(rvec)
                angles, *_ = cv2.RQDecomp3x3(rmat)
                pitch = angles[0]  # up/down
                yaw = angles[1]    # left/right
                roll = angles[2]   # tilt

                # Face forward vector in camera coordinates
                face_forward = rmat @ np.array([0, 0, -1])

                # Camera looks along +Z
                camera_forward = np.array([0, 0, 1])

                cos_angle = np.dot(face_forward, camera_forward)

                # Determine attention state based on cos_angle
                if cos_angle > 0.90:
                    attention = "MUTUAL_GAZE"
                    self.env_dict["MutualGaze"] += 1
                elif cos_angle > 0.7:
                    attention = "NEAR_GAZE"
                else:
                    attention = "AWAY"

                # Match MediaPipe detection with face identity footprint
                matched_face = self._match_face_to_bbox(face_center_x, face_center_y, matched_track_ids)
                
                # Mark this face as matched to prevent duplicate assignments
                if matched_face:
                    matched_track_ids.add(matched_face['track_id'])
                
                # Compute talking detection based on lip motion
                # Landmarks 13 (upper inner lip) and 14 (lower inner lip)
                is_talking = 0
                if len(face_landmarks) > 14:
                    upper_lip = face_landmarks[13]
                    lower_lip = face_landmarks[14]
                    
                    # Compute mouth opening in normalized coordinates
                    mouth_open_raw = np.hypot(upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y)
                    
                    # Normalize by face bbox height if available (scale-invariant)
                    if matched_face:
                        x, y, w, h = matched_face['bbox']
                        # Normalize: mouth_open as fraction of face height
                        mouth_open = mouth_open_raw / (h / self.default_height) if h > 0 else mouth_open_raw
                        track_id = matched_face['track_id']
                    else:
                        # If unmatched, use raw normalized coords (less reliable)
                        mouth_open = mouth_open_raw
                        track_id = -1  # Unmatched
                    
                    # Update motion history for this track_id
                    if track_id != -1:
                        if track_id not in self.mouth_motion_history:
                            self.mouth_motion_history[track_id] = deque(maxlen=self.mouth_buffer_size)
                        
                        self.mouth_motion_history[track_id].append(mouth_open)
                        self.last_seen_track[track_id] = current_time
                        
                        # Compute motion as std of mouth opening history
                        if len(self.mouth_motion_history[track_id]) >= 3:
                            mouth_motion = np.std(self.mouth_motion_history[track_id])
                            # Threshold: tune this for sensitivity (higher = less sensitive)
                            # 0.015 works well for normalized values; adjust if needed
                            is_talking = 1 if mouth_motion > self.talking_threshold else 0
                
                # Compute time_in_view
                if matched_face:
                    track_id = matched_face['track_id']
                    if track_id not in self.first_seen_track:
                        self.first_seen_track[track_id] = current_time
                    time_in_view = current_time - self.first_seen_track[track_id]
                else:
                    time_in_view = 0.0
                
                # Publish per-face landmarks data
                self._publish_landmarks(matched_face, face_forward, pitch, yaw, roll, cos_angle, attention, is_talking, time_in_view)
        
        # Publish data for faces detected by identity tracker but not matched by MediaPipe
        # (e.g., faces too small for landmark detection)
        for face_data in self.detected_faces:
            if face_data['track_id'] not in matched_track_ids:
                # Compute time_in_view for unmatched face
                track_id = face_data['track_id']
                if track_id not in self.first_seen_track:
                    self.first_seen_track[track_id] = current_time
                self.last_seen_track[track_id] = current_time
                time_in_view = current_time - self.first_seen_track[track_id]
                
                # Publish with default/unknown gaze values since no landmarks available
                self._publish_landmarks(
                    face_data=face_data,
                    gaze_direction=np.array([0.0, 0.0, 0.0]),
                    pitch=0.0,
                    yaw=0.0,
                    roll=0.0,
                    cos_angle=0.0,
                    attention="UNKNOWN",
                    is_talking=0,
                    time_in_view=time_in_view
                )

        # Cleanup: remove histories for tracks not seen in over 1 second
        tracks_to_remove = [tid for tid, last_time in self.last_seen_track.items() 
                           if current_time - last_time > 1.0]
        for tid in tracks_to_remove:
            if tid in self.mouth_motion_history:
                del self.mouth_motion_history[tid]
            if tid in self.last_seen_track:
                del self.last_seen_track[tid]
            if tid in self.first_seen_track:
                del self.first_seen_track[tid]

    def _match_face_to_bbox(self, face_x, face_y, matched_track_ids):
        """Match MediaPipe face detection to object recognition face using distance-based scoring.
        
        Implements one-to-one matching: each bbox can only be assigned to one MediaPipe face.
        Computes distance from face center to each bbox center, selects the closest match
        within MAX_FACE_MATCH_DISTANCE threshold. Ties are broken by preferring larger bbox area.
        
        Args:
            face_x: X coordinate of MediaPipe face center
            face_y: Y coordinate of MediaPipe face center
            matched_track_ids: Set of track_ids that have already been matched
            
        Returns:
            Best matching face_data dict or None if no valid match found
        """
        if not self.detected_faces:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for face_data in self.detected_faces:
            # Skip faces that have already been matched (one-to-one constraint)
            if face_data['track_id'] in matched_track_ids:
                continue
            
            x, y, w, h = face_data['bbox']
            
            # Compute bbox center
            bbox_center_x = x + w / 2.0
            bbox_center_y = y + h / 2.0
            
            # Compute Euclidean distance from face center to bbox center
            distance = np.hypot(face_x - bbox_center_x, face_y - bbox_center_y)
            
            # Update best match if this is closer
            if distance < best_distance:
                best_distance = distance
                best_match = face_data
            elif distance == best_distance and best_match is not None:
                # Tie-breaker: prefer larger bbox area for stability
                current_area = w * h
                best_area = best_match['bbox'][2] * best_match['bbox'][3]
                if current_area > best_area:
                    best_match = face_data
        
        # Apply gating threshold to reject poor matches
        if best_distance > self.max_face_match_distance:
            return None
        
        return best_match

    def _publish_landmarks(self, face_data, gaze_direction, pitch, yaw, roll, cos_angle, attention, is_talking, time_in_view):
        """Publish detailed landmarks information for a single face."""
        face_btl = yarp.Bottle()
        
        # Add face identification
        if face_data:
            face_btl.addString("face_id")
            face_btl.addString(face_data['face_id'])
            face_btl.addString("track_id")
            face_btl.addInt32(face_data['track_id'])
            
            # Add bounding box
            bbox_btl = face_btl.addList()
            bbox_btl.addString("bbox")
            x, y, w, h = face_data['bbox']
            bbox_btl.addFloat64(x)
            bbox_btl.addFloat64(y)
            bbox_btl.addFloat64(w)
            bbox_btl.addFloat64(h)
            
            # Add spatial information
            # Compute bbox center
            cx = x + w / 2.0
            cy = y + h / 2.0
            
            # Normalize center using default resolution
            cx_n = cx / self.default_width
            cy_n = cy / self.default_height
            
            # Clamp to [0, 1]
            cx_n = max(0.0, min(1.0, cx_n))
            cy_n = max(0.0, min(1.0, cy_n))
            
            # Determine horizontal zone
            if cx_n < 0.2:
                zone = "FAR_LEFT"
            elif cx_n < 0.4:
                zone = "LEFT"
            elif cx_n < 0.6:
                zone = "CENTER"
            elif cx_n < 0.8:
                zone = "RIGHT"
            else:
                zone = "FAR_RIGHT"
            
            # Add zone
            face_btl.addString("zone")
            face_btl.addString(zone)
            face_data['zone'] = zone
            
            # Determine distance based on face bbox height
            # Normalized height relative to default image height
            h_norm = h / self.default_height
            
            if h_norm > 0.4:
                distance = "SO_CLOSE"
            elif h_norm > 0.2:
                distance = "CLOSE"
            elif h_norm > 0.1:
                distance = "FAR"
            else:
                distance = "VERY_FAR"
            
            # Add distance
            face_btl.addString("distance")
            face_btl.addString(distance)
            
            face_data['distance'] = distance
            face_data['attention'] = attention
        else:
            # Unmatched face - use default/null values for all fields
            face_btl.addString("face_id")
            face_btl.addString("unmatched")
            face_btl.addString("track_id")
            face_btl.addInt32(-1)
            
            # Add empty bounding box
            bbox_btl = face_btl.addList()
            bbox_btl.addString("bbox")
            bbox_btl.addFloat64(0.0)
            bbox_btl.addFloat64(0.0)
            bbox_btl.addFloat64(0.0)
            bbox_btl.addFloat64(0.0)
            
            # Add default zone
            face_btl.addString("zone")
            face_btl.addString("UNKNOWN")
            
            # Add default distance
            face_btl.addString("distance")
            face_btl.addString("UNKNOWN")
        
        # Add gaze direction vector
        gaze_btl = face_btl.addList()
        gaze_btl.addString("gaze_direction")
        gaze_btl.addFloat64(float(gaze_direction[0]))
        gaze_btl.addFloat64(float(gaze_direction[1]))
        gaze_btl.addFloat64(float(gaze_direction[2]))
        
        # Add head orientation angles (degrees)
        face_btl.addString("pitch")
        face_btl.addFloat64(float(pitch))
        face_btl.addString("yaw")
        face_btl.addFloat64(float(yaw))
        face_btl.addString("roll")
        face_btl.addFloat64(float(roll))
        
        # Add cosine angle and attention state
        face_btl.addString("cos_angle")
        face_btl.addFloat64(float(cos_angle))
        face_btl.addString("attention")
        face_btl.addString(attention)
        face_btl.addString("is_talking")
        face_btl.addInt32(is_talking)
        face_btl.addString("time_in_view")
        face_btl.addFloat64(float(time_in_view))
        
        self.landmarks_btl.addList().read(face_btl)

    def __img_yarp_to_cv(self, image):

        if image.width() != self.img_width or image.height() != self.img_height:
            self.logger.warning("imput image has different size from default 640x480, fallback to automatic size detection")
            self.img_width = image.width()
            self.img_height = image.height()
            self.input_img_array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            print(f"New image size: W: {self.img_width}, H: {self.img_height}")


        image.setExternal(self.input_img_array.data, self.img_width, self.img_height)
        img = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
            (self.img_height, self.img_width, 3)).copy()

        # YARP BufferedPortImageRgb provides RGB; normalize internal pipeline to BGR.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.img_width != self.default_width or self.img_height != self.default_height:
            img = cv2.resize(img, (self.default_width, self.default_height))

        return img

    def opt_flow_add_img(self, frame):
        if len(self.opt_flow_buf) < 2:
            # Add new frame to the right part of the queue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.opt_flow_buf.append(frame)
        else:
            # Remove oldest element at the beginning of the queue
            self.opt_flow_buf.popleft()

    def detect_motion(self):
        if len(self.opt_flow_buf) == 2:
            # Dense optical flow estimate
            flow = cv2.calcOpticalFlowFarneback(self.opt_flow_buf[0], self.opt_flow_buf[1],
                                                flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=.2,
                                                flags=0)
            # Compute magnite and angle of 2D vector
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Dim (480, 640)
            # Cast into float64 since cartToPolar is float32 to avoid conflict with types when publishing with Yarp
            self.env_dict["Motion"] = round(mag.mean(), 2).astype(np.float64)

    def fill_bottle(self):
        btl_time = yarp.Bottle()
        btl_time.addString("Time")
        btl_time.addFloat64(self.timestamp)
        self.vision_features_btl.addList().read(btl_time)
        for key, value in self.env_dict.items():
            bottle = yarp.Bottle()
            bottle.addString(key)
            if isinstance(value, int):
                bottle.addInt16(value)
            elif isinstance(value, float):
                bottle.addFloat64(value)
            elif isinstance(value, str):
                bottle.addString(value)
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                # if it is a list, add each element to the bottle
                bottle_list = yarp.Bottle()
                for element in value:
                    if isinstance(element, int):
                        bottle_list.addInt16(element)
                    elif isinstance(element, float):
                        bottle_list.addFloat32(element)
                    elif isinstance(element, str):
                        bottle_list.addString(element)
                self.vision_features_btl.addList().read(bottle_list)
            self.vision_features_btl.addList().read(bottle)

    def interruptModule(self):
        print("stopping the module \n")
        self.img_in_port.interrupt()
        self.vision_features_port.interrupt()
        self.landmarks_port.interrupt()
        self.target_cmd_port.interrupt()
        self.target_box_port.interrupt()
        self.handle_port.interrupt()
        self.face_detection_img_port.interrupt()
        return True

    def close(self):
        print("closing the module \n")
        self.img_in_port.close()
        self.vision_features_port.close()
        self.landmarks_port.close()
        self.target_cmd_port.close()
        self.target_box_port.close()
        self.handle_port.close()
        self.face_detection_img_port.close()
        return True


if __name__ == '__main__':

    logger = get_colored_logger("Video Features")
    # Initialise YARP
    if not yarp.Network.checkNetwork():
        logger.warning("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    vision_analyzer = VisionAnalyzer()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('alwaysOn')

    if rf.configure(sys.argv):
        vision_analyzer.runModule(rf)

    yarp.Network.fini()
    sys.exit(0)
