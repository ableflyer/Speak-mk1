"""
This has not been used
"""
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import urllib.request
import cv2
import mediapipe as mp
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Model path — downloads automatically on first run if not present
# Override by setting this to an absolute local path before running.
# ──────────────────────────────────────────────────────────────────────────────

FACE_LANDMARKER_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")

def _ensure_model():
    """Download the face_landmarker.task model if not already present."""
    if Path(FACE_LANDMARKER_MODEL_PATH).exists():
        return
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    print(f"[model] Downloading face_landmarker.task to {FACE_LANDMARKER_MODEL_PATH} ...")
    urllib.request.urlretrieve(url, FACE_LANDMARKER_MODEL_PATH)
    print("[model] Download complete.")

_ensure_model()


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe FaceMesh landmark indices for the mouth region
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ──────────────────────────────────────────────────────────────────────────────

# Outer lip boundary (clockwise from left corner)
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
               37, 39, 40, 185, 61]

# Inner lip boundary (the actual mouth opening)
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
               82, 81, 80, 191, 78]

# Upper teeth landmarks (inside mouth, upper gum line)
UPPER_TEETH_INNER = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14]

# Lower teeth landmarks
LOWER_TEETH_INNER = [14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13]

# Mouth corners
LEFT_CORNER  = 61
RIGHT_CORNER = 291
UPPER_LIP_MID = 0    # top of upper lip
LOWER_LIP_MID = 17   # bottom of lower lip


# ──────────────────────────────────────────────────────────────────────────────
# Colour ranges (HSV, OpenCV scale: H 0-179, S 0-255, V 0-255)
# ──────────────────────────────────────────────────────────────────────────────

# Teeth: bright, low-saturation, high-value whites/yellows
TEETH_HSV_RANGES = [
    # Pure white teeth
    {"lower": np.array([0,   0,  160]), "upper": np.array([179,  60, 255])},
    # Slightly yellow teeth
    {"lower": np.array([15,  10, 130]), "upper": np.array([35,   90, 255])},
]

# Tongue: pinkish-red, moderate saturation
TONGUE_HSV_RANGES = [
    # Pink/red tongue
    {"lower": np.array([0,   60,  60]), "upper": np.array([15,  200, 220])},
    # Wrap-around reds (hue wraps at 179)
    {"lower": np.array([165, 60,  60]), "upper": np.array([179, 200, 220])},
    # Darker pink/mauve
    {"lower": np.array([0,   40,  50]), "upper": np.array([20,  160, 180])},
]


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MouthGeometry:
    """Derived from FaceMesh landmarks; describes mouth openness and shape."""
    mouth_open: bool = False
    opening_height_px: float = 0.0   # vertical gap between upper/lower inner lip
    opening_width_px:  float = 0.0   # horizontal span between corners
    opening_ratio:     float = 0.0   # height / width
    mouth_centre:      tuple = (0, 0)


@dataclass
class SegmentFeatures:
    """Geometric features extracted from teeth and tongue masks."""
    # Teeth
    teeth_area_px:     float = 0.0
    teeth_centroid:    tuple = (0, 0)
    teeth_visible:     bool  = False

    # Tongue
    tongue_area_px:    float = 0.0
    tongue_centroid:   tuple = (0, 0)
    tongue_tip:        tuple = (0, 0)   # topmost point of tongue mask
    tongue_visible:    bool  = False

    # Spatial relationship
    tongue_tip_to_upper_teeth_dist_px: float = 0.0
    tongue_protruding: bool  = False     # tongue tip above upper lip midpoint

    # Mouth geometry
    mouth: MouthGeometry = field(default_factory=MouthGeometry)


# ──────────────────────────────────────────────────────────────────────────────
# Core segmentation class
# ──────────────────────────────────────────────────────────────────────────────

class OralSegmenter:
    """
    Segments teeth and tongue from BGR video frames.

    Parameters
    ----------
    min_detection_confidence : float
        MediaPipe face detection threshold.
    min_tracking_confidence : float
        MediaPipe landmark tracking threshold.
    morph_kernel_size : int
        Morphological kernel size for mask cleanup.
    min_mouth_open_ratio : float
        Minimum height/width ratio to consider mouth open enough to segment.
    debug : bool
        If True, prints per-frame diagnostic info.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence:  float = 0.6,
        morph_kernel_size: int = 5,
        min_mouth_open_ratio: float = 0.06,
        debug: bool = False,
    ):
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.min_mouth_open_ratio = min_mouth_open_ratio
        self.debug = debug

        VisionRunningMode = mp.tasks.vision.RunningMode
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0   # monotonically increasing, required by VIDEO mode

    # ── public API ─────────────────────────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Segment one BGR frame.

        Returns
        -------
        dict with keys:
          'teeth_mask'  : uint8 binary mask, 255 = teeth pixels
          'tongue_mask' : uint8 binary mask, 255 = tongue pixels
          'features'    : SegmentFeatures instance
          'landmarks'   : raw MediaPipe NormalizedLandmarkList or None
        """
        h, w = bgr_frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        result = {
            "teeth_mask":  empty.copy(),
            "tongue_mask": empty.copy(),
            "features":    SegmentFeatures(),
            "landmarks":   None,
        }

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 1   # increment each frame; real ms not required for correctness
        mp_result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not mp_result.face_landmarks:
            if self.debug:
                print("[OralSegmenter] No face detected.")
            return result

        # face_landmarks[0] is a plain list of NormalizedLandmark objects
        landmark_list = mp_result.face_landmarks[0]
        result["landmarks"] = landmark_list

        pts = self._landmark_array(landmark_list, w, h)
        mouth_geo = self._mouth_geometry(pts)
        result["features"].mouth = mouth_geo

        if not mouth_geo.mouth_open:
            if self.debug:
                print(f"[OralSegmenter] Mouth closed (ratio={mouth_geo.opening_ratio:.3f}).")
            return result

        # Build mouth-interior mask from inner lip polygon
        mouth_mask = self._poly_mask(pts[INNER_LIPS], h, w)

        # Segment in HSV within the mouth ROI
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        teeth_mask  = self._hsv_segment(hsv, mouth_mask, TEETH_HSV_RANGES,  "teeth")
        tongue_mask = self._hsv_segment(hsv, mouth_mask, TONGUE_HSV_RANGES, "tongue")

        # Resolve overlap: teeth win over tongue (teeth are more spectrally distinct)
        overlap = cv2.bitwise_and(teeth_mask, tongue_mask)
        tongue_mask = cv2.bitwise_and(tongue_mask, cv2.bitwise_not(overlap))

        result["teeth_mask"]  = teeth_mask
        result["tongue_mask"] = tongue_mask
        result["features"]    = self._extract_features(
            teeth_mask, tongue_mask, mouth_geo, pts
        )

        return result

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _landmark_array(landmarks, w: int, h: int) -> np.ndarray:
        """Convert normalised landmarks to pixel coords, shape (N, 2).
        The Tasks API returns a plain list; the legacy API uses .landmark attribute.
        """
        lm_iter = landmarks if isinstance(landmarks, list) else landmarks.landmark
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in lm_iter],
            dtype=np.float32,
        )
        return pts

    def _mouth_geometry(self, pts: np.ndarray) -> MouthGeometry:
        upper_mid = pts[UPPER_LIP_MID]
        lower_mid = pts[LOWER_LIP_MID]
        left_c    = pts[LEFT_CORNER]
        right_c   = pts[RIGHT_CORNER]

        opening_h = float(np.linalg.norm(upper_mid - lower_mid))
        opening_w = float(np.linalg.norm(left_c - right_c))
        ratio     = opening_h / (opening_w + 1e-6)
        centre    = ((upper_mid + lower_mid) / 2).astype(int)

        return MouthGeometry(
            mouth_open=ratio > self.min_mouth_open_ratio,
            opening_height_px=opening_h,
            opening_width_px=opening_w,
            opening_ratio=ratio,
            mouth_centre=tuple(centre.tolist()),
        )

    @staticmethod
    def _poly_mask(pts: np.ndarray, h: int, w: int) -> np.ndarray:
        """Fill a convex polygon defined by landmark points."""
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly], 255)
        return mask

    def _hsv_segment(
        self,
        hsv: np.ndarray,
        roi_mask: np.ndarray,
        ranges: list,
        label: str,
    ) -> np.ndarray:
        """
        For each HSV range, threshold the frame, combine, mask to ROI,
        then apply morphological cleanup.
        """
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            m = cv2.inRange(hsv, r["lower"], r["upper"])
            combined = cv2.bitwise_or(combined, m)

        # Keep only inside mouth ROI
        combined = cv2.bitwise_and(combined, roi_mask)

        # Morphological cleanup: close small gaps, remove noise
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self.morph_kernel, iterations=1)

        # Keep only the largest connected component (avoids spurious patches)
        combined = self._keep_largest_component(combined)

        if self.debug:
            area = int(np.sum(combined > 0))
            print(f"[OralSegmenter] {label} area = {area} px")

        return combined

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        """Zero-out all connected components except the largest."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        # Component 0 is background — skip it
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, np.uint8(255), np.uint8(0))

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[tuple]:
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    @staticmethod
    def _mask_tip(mask: np.ndarray) -> Optional[tuple]:
        """Return the topmost (smallest y) non-zero pixel — the tongue tip."""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        idx = int(np.argmin(ys))
        return (int(xs[idx]), int(ys[idx]))

    def _extract_features(
        self,
        teeth_mask: np.ndarray,
        tongue_mask: np.ndarray,
        mouth_geo: MouthGeometry,
        pts: np.ndarray,
    ) -> SegmentFeatures:
        feat = SegmentFeatures(mouth=mouth_geo)

        teeth_area = int(np.sum(teeth_mask > 0))
        if teeth_area > 50:
            feat.teeth_visible  = True
            feat.teeth_area_px  = float(teeth_area)
            c = self._mask_centroid(teeth_mask)
            if c:
                feat.teeth_centroid = c

        tongue_area = int(np.sum(tongue_mask > 0))
        if tongue_area > 50:
            feat.tongue_visible  = True
            feat.tongue_area_px  = float(tongue_area)
            c = self._mask_centroid(tongue_mask)
            if c:
                feat.tongue_centroid = c
            tip = self._mask_tip(tongue_mask)
            if tip:
                feat.tongue_tip = tip

                # Distance from tongue tip to upper lip midpoint
                upper_lip_pt = pts[UPPER_LIP_MID]
                dist = float(np.linalg.norm(
                    np.array(tip) - upper_lip_pt
                ))
                feat.tongue_tip_to_upper_teeth_dist_px = dist

                # Protrusion: tongue tip is above (smaller y) upper lip midpoint
                feat.tongue_protruding = tip[1] < upper_lip_pt[1]

        return feat


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

TEETH_COLOUR  = (200, 255, 200)   # light green overlay
TONGUE_COLOUR = (100, 100, 255)   # red-ish overlay
ALPHA = 0.45


def draw_overlay(bgr_frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Render coloured mask overlays and feature annotations onto a copy of the frame.
    """
    vis = bgr_frame.copy()
    h, w = vis.shape[:2]

    teeth_mask  = result["teeth_mask"]
    tongue_mask = result["tongue_mask"]
    feat: SegmentFeatures = result["features"]

    # Coloured overlays
    for mask, colour in [(teeth_mask, TEETH_COLOUR), (tongue_mask, TONGUE_COLOUR)]:
        if mask is None or not np.any(mask):
            continue
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = colour
        vis = cv2.addWeighted(vis, 1.0, overlay, ALPHA, 0)
        # Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, colour, 1)

    # Feature annotations
    def put(text, row, colour=(255, 255, 255)):
        cv2.putText(vis, text, (10, 20 + row * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (10, 20 + row * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, colour, 1, cv2.LINE_AA)

    put(f"Mouth open : {'YES' if feat.mouth.mouth_open else 'NO'} "
        f"(ratio={feat.mouth.opening_ratio:.2f})", 0)
    put(f"Teeth  : {'visible' if feat.teeth_visible else 'hidden'} "
        f"| area={feat.teeth_area_px:.0f}px", 1,
        TEETH_COLOUR if feat.teeth_visible else (120, 120, 120))
    put(f"Tongue : {'visible' if feat.tongue_visible else 'hidden'} "
        f"| area={feat.tongue_area_px:.0f}px", 2,
        TONGUE_COLOUR if feat.tongue_visible else (120, 120, 120))

    if feat.tongue_visible:
        put(f"Tongue tip   : {feat.tongue_tip}", 3)
        put(f"Tip→upper-lip: {feat.tongue_tip_to_upper_teeth_dist_px:.1f}px", 4)
        put(f"Protruding   : {'YES' if feat.tongue_protruding else 'NO'}", 5,
            (50, 255, 50) if feat.tongue_protruding else (200, 200, 200))

    # Draw centroids
    if feat.teeth_visible and feat.teeth_centroid != (0, 0):
        cv2.drawMarker(vis, feat.teeth_centroid, TEETH_COLOUR,
                       cv2.MARKER_CROSS, 12, 2)
    if feat.tongue_visible and feat.tongue_centroid != (0, 0):
        cv2.drawMarker(vis, feat.tongue_centroid, TONGUE_COLOUR,
                       cv2.MARKER_CROSS, 12, 2)
    if feat.tongue_visible and feat.tongue_tip != (0, 0):
        cv2.circle(vis, feat.tongue_tip, 5, (0, 255, 255), -1)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Video processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types produced by cv2/numpy ops."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def process_video(
    source,                         # path string or int (webcam index)
    output_dir: Optional[Path],
    visualise: bool = False,
    save_video: bool = True,
    save_json: bool = True,
    debug: bool = False,
) -> list:
    """
    Process a video source frame-by-frame.

    Returns
    -------
    list of dicts, one per frame, containing serialisable feature data.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[pipeline] Source : {source}")
    print(f"[pipeline] Size   : {width}×{height}  FPS={fps:.1f}  Frames={total}")

    writer = None
    if save_video and output_dir is not None:
        out_path = output_dir / "segmented.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[pipeline] Writing video → {out_path}")

    all_features = []
    frame_idx = 0

    with OralSegmenter(debug=debug) as segmenter:
        segmenter._timestamp_ms = 0  # reset per video
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = segmenter.process_frame(frame)
            feat   = result["features"]

            # Serialise features for JSON output
            feat_dict = asdict(feat)
            feat_dict["frame"] = frame_idx
            all_features.append(feat_dict)

            # Visualisation
            vis_frame = draw_overlay(frame, result) if (visualise or save_video) else frame

            if writer is not None:
                writer.write(vis_frame)

            if visualise:
                cv2.imshow("Speak MK1 — Oral Segmentation", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(f"[pipeline] Frame {frame_idx}/{total} ({pct:.0f}%)")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"[pipeline] Done — processed {frame_idx} frames.")

    if save_json and output_dir is not None:
        json_path = output_dir / "features.json"
        with open(json_path, "w") as f:
            json.dump(all_features, f, indent=2, cls=_NumpyEncoder)
        print(f"[pipeline] Features saved → {json_path}")

    return all_features


# ──────────────────────────────────────────────────────────────────────────────
# Folder processing
# ──────────────────────────────────────────────────────────────────────────────

def process_folder(
    input_folder: str,
    output_folder: str,
    visualise: bool = False,
    save_video: bool = True,
    save_json: bool = True,
    debug: bool = False,
):
    """
    Process all .mp4 files found in input_folder.
    Each video gets its own subdirectory inside output_folder named after the video stem.

    Parameters
    ----------
    input_folder  : path to folder containing .mp4 files
    output_folder : path where results will be written
    visualise     : show live preview window while processing
    save_video    : save annotated video to each video's output subdirectory
    save_json     : save per-frame feature JSON to each video's output subdirectory
    debug         : print per-frame diagnostics
    """
    input_dir  = Path(input_folder)
    output_dir = Path(output_folder)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in: {input_dir}")

    print(f"[folder] Found {len(videos)} video(s) in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, video_path in enumerate(videos, 1):
        print(f"\n[folder] ── Video {i}/{len(videos)}: {video_path.name} ──")
        video_out_dir = output_dir / video_path.stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        process_video(
            source=str(video_path),
            output_dir=video_out_dir,
            visualise=visualise,
            save_video=save_video,
            save_json=save_json,
            debug=debug,
        )

    print(f"\n[folder] All done. Results in: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — edit the paths below and run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_FOLDER  = "../Data/GRID_Dataset/s1"   # folder containing .mp4 files
    OUTPUT_FOLDER = "../Data/GRID_Dataset/s1_json"         # results written here

    process_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        visualise=False,   # set True to watch each video as it processes
        save_video=True,   # saves annotated .mp4 per video
        save_json=True,    # saves features.json per video
        debug=False,       # set True for per-frame area diagnostics
    )