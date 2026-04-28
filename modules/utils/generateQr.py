#!/usr/bin/env python3
"""
generate_qrs.py - Generate meal QR codes for interactionManager.

Payloads → hunger delta:
  SMALL_MEAL  → +10
  MEDIUM_MEAL → +25
  LARGE_MEAL  → +45

Output PNGs are saved to DEFAULT_OUTPUT_DIR (or --output-dir).
Optionally verifies decode with OpenCV (--verify).
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import qrcode
from qrcode.constants import ERROR_CORRECT_M

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAYLOAD_TO_DELTA: Dict[str, float] = {
    "SMALL_MEAL":  10.0,
    "MEDIUM_MEAL": 25.0,
    "LARGE_MEAL":  45.0,
}

DEFAULT_OUTPUT_DIR = Path(
    "/usr/local/src/robot/cognitiveInteraction/developmental-cognitive-architecture"
    "/modules/alwaysOn/memory"
)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def build_qr(payload: str):
    """Build a QR image for *payload*."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=12,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def generate_qrs(output_dir: Path) -> Dict[str, Path]:
    """Generate one PNG per payload; return {payload: path} mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}
    for payload in PAYLOAD_TO_DELTA:
        path = output_dir / f"{payload.lower()}.png"
        build_qr(payload).save(path)
        generated[payload] = path
        print(f"  [QR] {payload} → {path.name}")
    return generated


def verify_with_cv2(files: Dict[str, Path]) -> Tuple[Optional[bool], str]:
    """Decode each generated image with OpenCV and check the payload matches."""
    try:
        import cv2
    except ImportError:
        return None, "cv2 not available – decode verification skipped"

    detector = cv2.QRCodeDetector()
    for payload, path in files.items():
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, f"Cannot read image: {path}"
        decoded, _, _ = detector.detectAndDecode(img)
        normalized = decoded.strip().upper() if decoded else ""
        if normalized != payload:
            return False, f"Decode mismatch for {path.name}: got '{decoded}', expected '{payload}'"
    return True, "All QR codes verified OK"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate meal QR codes for interactionManager.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output folder (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated images can be decoded by OpenCV.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"[QR] Generating {len(PAYLOAD_TO_DELTA)} QR codes → {output_dir}")
    files = generate_qrs(output_dir)

    if args.verify:
        ok, msg = verify_with_cv2(files)
        if ok is None:
            print(f"[SKIP] {msg}")
            return 0
        tag = "OK" if ok else "FAIL"
        print(f"[{tag}] {msg}")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
