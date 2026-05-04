"""
mouth_tracker.py
================
SpeakMK1 — MediaPipe Face Landmarker mouth tracking module.

Uses the modern MediaPipe Tasks API (mediapipe.tasks.vision.FaceLandmarker)
rather than the deprecated mp.solutions.face_mesh interface.

Requires:
    pip install mediapipe>=0.10.0

Model download (run once):
    wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    # or call MouthTracker.download_model() — see bottom of file.

Extracts clinically relevant orofacial landmarks for SLP articulation analysis:
    - Lip contour (upper + lower)
    - Jaw aperture (chin-to-forehead distance)
    - Teeth visibility proxy (inner lip separation)

Output: a flat feature vector (180,) suitable for linear projection into d_model.

Usage
-----
    tracker = MouthTracker(model_path="face_landmarker.task")

    # Single frame (numpy BGR from OpenCV)
    result = tracker.process_frame(frame)
    vec    = result.to_vector()          # (180,) numpy array
    tensor = result.to_tensor()          # torch.Tensor (180,)

    # Batch of pre-extracted landmark arrays
    vecs = MouthTracker.batch_to_tensor(list_of_results)  # (B, 180)

    # Live webcam demo
    tracker.run_webcam()
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
import pygame

# ── optional torch import ────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ════════════════════════════════════════════════════════════════════════════
# 0.  MODEL ASSET
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL_PATH = "face_landmarker.task"


def download_model(dest: str = _DEFAULT_MODEL_PATH) -> str:
    """
    Download the face_landmarker.task model file if it does not already exist.
    Returns the path to the downloaded file.
    """
    if not os.path.exists(dest):
        print(f"[MouthTracker] Downloading model → {dest} ...")
        urllib.request.urlretrieve(_DEFAULT_MODEL_URL, dest)
        print("[MouthTracker] Download complete.")
    return dest


# ════════════════════════════════════════════════════════════════════════════
# 1.  LANDMARK INDEX DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

# Outer lip contour
LIPS_UPPER: List[int] = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LIPS_LOWER: List[int] = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Jaw aperture: vertical distance between chin (152) and glabella (10)
JAW_OPEN: List[int] = [152, 10]

# Inner lip: upper/lower inner lip edge — proxy for teeth visibility
TEETH_VISIBLE: List[int] = [13, 14]

# All indices in extraction order
ALL_INDICES: List[int] = LIPS_UPPER + LIPS_LOWER + JAW_OPEN + TEETH_VISIBLE
# 10 + 10 + 2 + 2 = 24 landmarks × 3 coords = 72 raw dims

# ── derived feature dimensions ───────────────────────────────────────────────
N_LANDMARKS = len(ALL_INDICES)       # 24
RAW_DIM     = N_LANDMARKS * 3        # 72  (x, y, z per landmark)
DERIVED_DIM = 9                      # scalar clinical features
FEATURE_DIM = RAW_DIM + DERIVED_DIM  # 81
TARGET_DIM  = 180                    # final output dimension (zero-padded)


# ════════════════════════════════════════════════════════════════════════════
# 2.  RESULT DATACLASS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MouthLandmarkResult:
    """
    Extracted mouth landmark data for one frame.

    Attributes
    ----------
    landmarks_xyz : (24, 3) array of normalised (x, y, z) coordinates
    jaw_aperture  : scalar — normalised chin-to-forehead distance
    lip_aperture  : scalar — normalised upper/lower inner lip gap (teeth proxy)
    lip_width     : scalar — normalised outer lip corner distance
    lip_sym_h     : scalar — horizontal symmetry score [0, 1]; 1 = perfectly symmetric
    lip_sym_v     : scalar — vertical symmetry score [0, 1]
    upper_curve   : scalar — mean vertical displacement of upper lip midpoints
    lower_curve   : scalar — mean vertical displacement of lower lip midpoints
    protrusion    : scalar — mean z-depth of lip landmarks (lip protrusion)
    teeth_gap     : scalar — inner lip separation (teeth visibility proxy)
    detected      : bool — whether a face was detected this frame
    """
    landmarks_xyz : np.ndarray          # (24, 3)
    jaw_aperture  : float = 0.0
    lip_aperture  : float = 0.0
    lip_width     : float = 0.0
    lip_sym_h     : float = 0.0
    lip_sym_v     : float = 0.0
    upper_curve   : float = 0.0
    lower_curve   : float = 0.0
    protrusion    : float = 0.0
    teeth_gap     : float = 0.0
    detected      : bool  = False

    SCALAR_NAMES: List[str] = field(default_factory=lambda: [
        "jaw_aperture", "lip_aperture", "lip_width",
        "lip_sym_h", "lip_sym_v",
        "upper_curve", "lower_curve",
        "protrusion", "teeth_gap",
    ], repr=False)

    def to_vector(self) -> np.ndarray:
        """
        Returns flat (TARGET_DIM,) float32 vector.
        Layout: [raw_xyz (72,) | derived_scalars (9,) | zero_pad (...)]
        """
        raw = self.landmarks_xyz.flatten().astype(np.float32)   # (72,)
        scalars = np.array([
            self.jaw_aperture, self.lip_aperture, self.lip_width,
            self.lip_sym_h,    self.lip_sym_v,
            self.upper_curve,  self.lower_curve,
            self.protrusion,   self.teeth_gap,
        ], dtype=np.float32)                                      # (9,)

        combined = np.concatenate([raw, scalars])                 # (81,)
        if len(combined) < TARGET_DIM:
            combined = np.pad(combined, (0, TARGET_DIM - len(combined)))
        return combined[:TARGET_DIM]

    def to_tensor(self) -> "torch.Tensor":
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed.")
        return torch.from_numpy(self.to_vector())

    def scalars_dict(self) -> Dict[str, float]:
        return {
            "jaw_aperture" : self.jaw_aperture,
            "lip_aperture" : self.lip_aperture,
            "lip_width"    : self.lip_width,
            "lip_sym_h"    : self.lip_sym_h,
            "lip_sym_v"    : self.lip_sym_v,
            "upper_curve"  : self.upper_curve,
            "lower_curve"  : self.lower_curve,
            "protrusion"   : self.protrusion,
            "teeth_gap"    : self.teeth_gap,
            "detected"     : float(self.detected),
        }


# ════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE EXTRACTION  (shared by image and live-stream paths)
# ════════════════════════════════════════════════════════════════════════════

_ALL_IDX = np.array(ALL_INDICES, dtype=np.int32)

# Local positions within the 24-element selection
_UPPER_LOCAL = list(range(0, 10))
_LOWER_LOCAL = list(range(10, 20))
_JAW_LOCAL   = [20, 21]
_TEETH_LOCAL = [22, 23]


def _extract(face_landmarks) -> MouthLandmarkResult:
    """
    Build a MouthLandmarkResult from one FaceLandmarkerResult face_landmarks entry.

    Parameters
    ----------
    face_landmarks : list of NormalizedLandmark  (478 items with refined mesh)
    """
    # Convert to (N, 3) array — x, y, z all in [0,1] normalised coords
    all_lm = np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks],
        dtype=np.float32,
    )

    sel   = all_lm[_ALL_IDX]          # (24, 3)
    upper = sel[_UPPER_LOCAL]         # (10, 3)
    lower = sel[_LOWER_LOCAL]         # (10, 3)
    jaw   = sel[_JAW_LOCAL]           # (2, 3)  [chin, forehead]
    teeth = sel[_TEETH_LOCAL]         # (2, 3)  [inner_upper, inner_lower]

    # 1. Jaw aperture
    jaw_aperture = float(abs(jaw[0, 1] - jaw[1, 1]))

    # 2. Lip aperture
    lip_aperture = float(abs(teeth[0, 1] - teeth[1, 1]))

    # 3. Teeth gap (normalised by jaw)
    teeth_gap = lip_aperture / (jaw_aperture + 1e-6)

    # 4. Lip width
    left_corner  = upper[0, :2]
    right_corner = upper[-1, :2]
    lip_width = float(np.linalg.norm(right_corner - left_corner))

    # 5. Horizontal symmetry
    left_half = upper[:5, 1]
    right_half = upper[5:, 1]
    lip_sym_h = float(
        1.0 - np.mean(np.abs(left_half - right_half[::-1])) / (jaw_aperture + 1e-6)
    )
    lip_sym_h = float(np.clip(lip_sym_h, 0.0, 1.0))

    # 6. Vertical symmetry
    upper_y = upper[:, 1]
    lower_y = lower[:, 1]
    mid_y   = (upper_y.mean() + lower_y.mean()) / 2.0
    lip_sym_v = float(
        1.0 - abs(upper_y.mean() - mid_y - (mid_y - lower_y.mean())) / (jaw_aperture + 1e-6)
    )
    lip_sym_v = float(np.clip(lip_sym_v, 0.0, 1.0))

    # 7–8. Lip curvature
    upper_curve = float(np.std(upper[:, 1]))
    lower_curve = float(np.std(lower[:, 1]))

    # 9. Protrusion (mean z-depth)
    protrusion = float(np.mean(np.concatenate([upper[:, 2], lower[:, 2]])))

    return MouthLandmarkResult(
        landmarks_xyz = sel,
        jaw_aperture  = jaw_aperture,
        lip_aperture  = lip_aperture,
        lip_width     = lip_width,
        lip_sym_h     = lip_sym_h,
        lip_sym_v     = lip_sym_v,
        upper_curve   = upper_curve,
        lower_curve   = lower_curve,
        protrusion    = protrusion,
        teeth_gap     = teeth_gap,
        detected      = True,
    )


def _empty_result() -> MouthLandmarkResult:
    return MouthLandmarkResult(
        landmarks_xyz=np.zeros((N_LANDMARKS, 3), dtype=np.float32),
        detected=False,
    )


# ════════════════════════════════════════════════════════════════════════════
# 4.  MOUTH TRACKER
# ════════════════════════════════════════════════════════════════════════════

class MouthTracker:
    """
    MediaPipe Face Landmarker (Tasks API) wrapper for SpeakMK1 orofacial
    landmark extraction.

    Parameters
    ----------
    model_path           : path to face_landmarker.task; auto-downloaded if absent
    running_mode         : RunningMode.IMAGE | VIDEO | LIVE_STREAM
    num_faces            : maximum faces to detect (smoothing only works with 1)
    min_detection_conf   : detection confidence threshold
    min_presence_conf    : face-presence confidence threshold
    min_tracking_conf    : tracking confidence (VIDEO / LIVE_STREAM only)
    output_blendshapes   : enable 52-coefficient blendshape output
    """

    def __init__(
        self,
        model_path          : str   = _DEFAULT_MODEL_PATH,
        running_mode        : RunningMode = RunningMode.IMAGE,
        num_faces           : int   = 1,
        min_detection_conf  : float = 0.5,
        min_presence_conf   : float = 0.5,
        min_tracking_conf   : float = 0.5,
        output_blendshapes  : bool  = False,
    ):
        model_path = download_model(model_path)

        self._running_mode = running_mode

        # Async callback storage for LIVE_STREAM mode
        self._latest_result: Optional[MouthLandmarkResult] = None

        base_options = mp_python.BaseOptions(model_asset_path=model_path)

        options = FaceLandmarkerOptions(
            base_options                    = base_options,
            running_mode                    = running_mode,
            num_faces                       = num_faces,
            min_face_detection_confidence   = min_detection_conf,
            min_face_presence_confidence    = min_presence_conf,
            min_tracking_confidence         = min_tracking_conf,
            output_face_blendshapes         = output_blendshapes,
            result_callback                 = (
                self._live_stream_callback
                if running_mode == RunningMode.LIVE_STREAM
                else None
            ),
        )

        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ── async callback (LIVE_STREAM only) ────────────────────────────────────

    def _live_stream_callback(
        self,
        result,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        if result.face_landmarks:
            self._latest_result = _extract(result.face_landmarks[0])
        else:
            self._latest_result = _empty_result()

    # ── public API ───────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> MouthLandmarkResult:
        """
        Process a single BGR frame (as returned by cv2.VideoCapture.read()).

        For VIDEO mode, timestamp_ms must be provided and monotonically increasing.
        For LIVE_STREAM mode, detection runs asynchronously; the return value
        reflects the PREVIOUS frame's result (call get_latest_result() instead).

        Returns MouthLandmarkResult with detected=False and zeros if no face found.
        """
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._running_mode == RunningMode.IMAGE:
            detection = self._landmarker.detect(mp_image)
        elif self._running_mode == RunningMode.VIDEO:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required for VIDEO running mode.")
            detection = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:  # LIVE_STREAM
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required for LIVE_STREAM running mode.")
            self._landmarker.detect_async(mp_image, timestamp_ms)
            return self._latest_result or _empty_result()

        if not detection.face_landmarks:
            return _empty_result()
        return _extract(detection.face_landmarks[0])

    def process_image_path(self, path: str) -> MouthLandmarkResult:
        """Convenience wrapper for static image files (IMAGE mode only)."""
        if self._running_mode != RunningMode.IMAGE:
            raise RuntimeError("process_image_path requires IMAGE running mode.")
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.process_frame(frame)

    def get_latest_result(self) -> MouthLandmarkResult:
        """Return the most recent result from async LIVE_STREAM processing."""
        return self._latest_result or _empty_result()

    @staticmethod
    def batch_to_tensor(results: List[MouthLandmarkResult]) -> "torch.Tensor":
        """
        Stack a list of MouthLandmarkResults into a (B, TARGET_DIM) tensor.
        Undetected frames produce zero vectors.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed.")
        vecs = np.stack([r.to_vector() for r in results], axis=0)
        return torch.from_numpy(vecs)

    # ── visualisation ─────────────────────────────────────────────────────────

    def draw_landmarks(
        self,
        frame_bgr : np.ndarray,
        result    : MouthLandmarkResult,
    ) -> np.ndarray:
        """Draw extracted landmarks and HUD on a copy of frame_bgr."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        if not result.detected:
            cv2.putText(vis, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        lm = result.landmarks_xyz  # (24, 3), normalised

        def to_px(idx_local: int) -> Tuple[int, int]:
            x, y = lm[idx_local, 0], lm[idx_local, 1]
            return int(x * w), int(y * h)

        # Lip contours
        upper_pts = np.array([to_px(i) for i in _UPPER_LOCAL], dtype=np.int32)
        lower_pts = np.array([to_px(i) for i in _LOWER_LOCAL], dtype=np.int32)
        cv2.polylines(vis, [upper_pts], isClosed=False, color=(0, 255, 100), thickness=2)
        cv2.polylines(vis, [lower_pts], isClosed=False, color=(0, 200, 255), thickness=2)

        for i in _UPPER_LOCAL:
            cv2.circle(vis, to_px(i), 3, (0, 255, 100), -1)
        for i in _LOWER_LOCAL:
            cv2.circle(vis, to_px(i), 3, (0, 200, 255), -1)

        # Jaw aperture line
        chin_px     = to_px(_JAW_LOCAL[0])
        forehead_px = to_px(_JAW_LOCAL[1])
        cv2.line(vis, chin_px, forehead_px, (255, 100, 0), 1)
        cv2.circle(vis, chin_px,     4, (255, 100, 0), -1)
        cv2.circle(vis, forehead_px, 4, (255, 100, 0), -1)

        # Teeth landmarks
        for i in _TEETH_LOCAL:
            cv2.circle(vis, to_px(i), 5, (0, 100, 255), -1)

        # HUD
        scalars = result.scalars_dict()
        hud_lines = [
            f"jaw_aperture : {scalars['jaw_aperture']:.3f}",
            f"lip_aperture : {scalars['lip_aperture']:.3f}",
            f"teeth_gap    : {scalars['teeth_gap']:.3f}",
            f"lip_width    : {scalars['lip_width']:.3f}",
            f"protrusion   : {scalars['protrusion']:.4f}",
            f"lip_sym_h    : {scalars['lip_sym_h']:.3f}",
            f"lip_sym_v    : {scalars['lip_sym_v']:.3f}",
        ]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (260, 20 + 22 * len(hud_lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
        for i, line in enumerate(hud_lines):
            cv2.putText(vis, line, (8, 22 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        return vis

    def run_webcam_pygame(self, camera_index: int = 0) -> None:
        """Live webcam demo using Pygame for display."""
        import time

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        # Get camera properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pygame.init()
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("SpeakMK1 — Mouth Tracker")
        clock = pygame.time.Clock()

        # Spin up a dedicated live-stream instance
        with MouthTracker(running_mode=RunningMode.LIVE_STREAM) as live:
            t0 = time.time()
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_q
                    ):
                        running = False

                ok, frame = cap.read()
                if not ok:
                    break

                ts_ms = int((time.time() - t0) * 1000)
                live.process_frame(frame, timestamp_ms=ts_ms)
                result = live.get_latest_result()
                vis = self.draw_landmarks(frame, result)

                # Convert BGR → RGB for Pygame
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                surface = pygame.surfarray.make_surface(vis_rgb.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(30)

        cap.release()
        pygame.quit()

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ════════════════════════════════════════════════════════════════════════════
# 5.  LINEAR PROJECTION MODULE (plug directly into SpeakMK1)
# ════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    import torch.nn as nn

    class MouthProjection(nn.Module):
        """
        Projects MouthTracker feature vectors into the LLM's token space.

        Usage
        -----
            proj   = MouthProjection(input_dim=180, llm_dim=512)
            vec    = tracker.process_frame(frame).to_tensor()       # (180,)
            token  = proj(vec.unsqueeze(0).unsqueeze(0))            # (1, 1, 512)

        In the multimodal forward pass:
            visual_tok = proj(mouth_vecs)               # (B, 1, 512)
            x = torch.cat([audio_tokens,                # (B, 64, 512)
                           visual_tok,                  # (B,  1, 512)
                           text_emb], dim=1)            # (B, seq, 512)
        """

        def __init__(self, input_dim: int = TARGET_DIM, llm_dim: int = 512):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(input_dim, llm_dim * 2),
                nn.GELU(),
                nn.Linear(llm_dim * 2, llm_dim),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            x : (B, TARGET_DIM) or (B, 1, TARGET_DIM)
            returns : (B, 1, llm_dim)
            """
            if x.dim() == 2:
                x = x.unsqueeze(1)   # (B, 1, dim)
            return self.proj(x)      # (B, 1, llm_dim)


# ════════════════════════════════════════════════════════════════════════════
# 6.  SMOKE TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  SpeakMK1 MouthTracker — smoke test (Tasks API)")
    print("=" * 60)

    print(f"\n  Landmark groups:")
    print(f"    LIPS_UPPER    : {LIPS_UPPER}")
    print(f"    LIPS_LOWER    : {LIPS_LOWER}")
    print(f"    JAW_OPEN      : {JAW_OPEN}")
    print(f"    TEETH_VISIBLE : {TEETH_VISIBLE}")
    print(f"\n  Feature vector dim : {TARGET_DIM}")
    print(f"    raw_xyz  ({N_LANDMARKS} landmarks × 3) : {RAW_DIM}")
    print(f"    derived scalars                  : {DERIVED_DIM}")
    print(f"    zero pad                         : {TARGET_DIM - RAW_DIM - DERIVED_DIM}")

    # ── model download ────────────────────────────────────────────────────
    model_path = download_model()

    # ── synthetic frame (no face expected) ───────────────────────────────
    print("\n  [1] Synthetic frame (random noise — no face expected) ...")
    tracker = MouthTracker(model_path=model_path, running_mode=RunningMode.IMAGE)
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(fake_frame)
    print(f"      detected  : {result.detected}")
    vec = result.to_vector()
    print(f"      vec shape : {vec.shape}")
    print(f"      vec norm  : {np.linalg.norm(vec):.4f}  (should be 0.0 — no face)")

    # ── torch tensor + projection ─────────────────────────────────────────
    if HAS_TORCH:
        print("\n  [2] Torch tensor + MouthProjection ...")
        t = result.to_tensor()
        print(f"      tensor shape : {t.shape}  dtype={t.dtype}")

        proj = MouthProjection(input_dim=TARGET_DIM, llm_dim=512)
        n_params = sum(p.numel() for p in proj.parameters())
        print(f"      MouthProjection params : {n_params:,}")

        dummy_batch = torch.zeros(4, TARGET_DIM)
        out = proj(dummy_batch)
        print(f"      proj output shape : {out.shape}  (expected: [4, 1, 512])")
        assert out.shape == (4, 1, 512), "Shape mismatch!"
        print("      Shape assertion passed ✓")
    else:
        print("\n  [2] PyTorch not installed — skipping tensor test.")

    # ── batch test ────────────────────────────────────────────────────────
    print("\n  [3] Batch stacking (8 synthetic results) ...")
    results = [tracker.process_frame(fake_frame) for _ in range(8)]
    if HAS_TORCH:
        batch_t = MouthTracker.batch_to_tensor(results)
        print(f"      batch tensor shape : {batch_t.shape}  (expected: [8, {TARGET_DIM}])")
        assert batch_t.shape == (8, TARGET_DIM)
        print("      Batch assertion passed ✓")

    tracker.close()

    # ── webcam demo ───────────────────────────────────────────────────────
    if "--webcam" in sys.argv:
        print("\n  [4] Starting webcam demo ...")
        with MouthTracker(model_path=model_path, running_mode=RunningMode.IMAGE) as t:
            t.run_webcam_pygame(camera_index=0)
    else:
        print("\n  Run with --webcam to launch live demo.")

    print("\nDone ✓")