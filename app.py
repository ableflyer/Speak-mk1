import os
import sys
import time
import signal
import atexit
import torch
import librosa
import tempfile
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from scipy.spatial.distance import cdist
from scipy.io.wavfile import write as wav_write
import mediapipe as mp
import cv2

# ─── paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
IMAGE_PATH = os.path.join(BASE_DIR, "images", "Gemini_Generated_Image_vvw7ulvvw7ulvvw7.png")

# ─── wav2vec2 model (loaded once) ────────────────────────────────────────────
print("🧠 Loading Wav2Vec2 model – hang tight …")
_w2v_name = "facebook/wav2vec2-large-xlsr-53"
_feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(_w2v_name)
_w2v_model = Wav2Vec2Model.from_pretrained(_w2v_name)
_w2v_model.eval()
print("✅ Model loaded!")

# ─── gold‑standard reference for /r/ ─────────────────────────────────────────
GOLD_AUDIO = os.path.join(BASE_DIR, "correct.wav")

# ─── MediaPipe landmark constants ────────────────────────────────────────────
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     185, 40, 39, 37, 0, 267, 269, 270, 409]
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                     191, 80, 81, 82, 13, 312, 311, 310, 415]
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_LIP_OUTER = 0
LOWER_LIP_OUTER = 17

# ─── persistent user state ────────────────────────────────────────────────────
user_state = {"xp": 0, "streak": 0, "attempts": 0}

# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO  FEATURES  (from wav2vec2test.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_speech_features(audio_path: str) -> np.ndarray:
    speech, _ = librosa.load(audio_path, sr=16000)
    speech, _ = librosa.effects.trim(speech, top_db=20)
    inp = _feat_ext(speech, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        out = _w2v_model(inp)
    feats = out.last_hidden_state.squeeze(0).numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1
    feats = feats / norms
    d1 = np.diff(feats, axis=0)
    d2 = np.diff(d1, axis=0)
    d1 = np.vstack([d1, d1[-1:]])
    d2 = np.vstack([d2, d2[-1:], d2[-1:]])
    return np.concatenate([feats, d1 * 2.0, d2 * 4.0], axis=1)


def compare_speech(feat_gold: np.ndarray, feat_user: np.ndarray):
    cost = cdist(feat_gold, feat_user, metric="cosine")
    n, m = cost.shape
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        cands = [(dtw[i - 1, j - 1], i - 1, j - 1),
                 (dtw[i - 1, j], i - 1, j),
                 (dtw[i, j - 1], i, j - 1)]
        _, i, j = min(cands, key=lambda x: x[0])
    path.reverse()
    dists = np.array([cost[i, j] for i, j in path])
    avg = np.mean(dists)
    p90 = np.percentile(dists, 90)
    blended = avg * 0.5 + p90 * 0.5
    score = max(0, min(100, (1 - blended / 0.10) * 100))
    return score, dists, path


# ═══════════════════════════════════════════════════════════════════════════════
#  MOUTH  SHAPE  ANALYSIS  (MediaPipe)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_mouth_metrics(landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    upper_i = pt(UPPER_LIP_TOP)
    lower_i = pt(LOWER_LIP_BOTTOM)
    left_c = pt(LEFT_CORNER)
    right_c = pt(RIGHT_CORNER)
    upper_o = pt(UPPER_LIP_OUTER)
    lower_o = pt(LOWER_LIP_OUTER)
    mw = np.linalg.norm(right_c - left_c)
    ih = np.linalg.norm(lower_i - upper_i)
    oh = np.linalg.norm(lower_o - upper_o)
    ar = mw / max(ih, 1e-6)
    openness = ih / max(oh, 1e-6)
    thickness = (oh - ih) / max(mw, 1e-6)
    return {"aspect_ratio": ar, "openness": openness, "mouth_width": mw,
            "inner_height": ih, "lip_thickness": thickness}


def analyse_video_frame(frame_bgr):
    """Run MediaPipe on a single BGR frame and return mouth metrics or None."""
    if frame_bgr is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with FaceLandmarker.create_from_options(options) as lm:
        result = lm.detect(mp_img)
    if result.face_landmarks:
        return _extract_mouth_metrics(result.face_landmarks[0], w, h)
    return None


def mouth_diagnostic(metrics):
    """
    Given mouth metrics from a video frame, diagnose /r/ vs /w/ confusion
    and return (is_r_shape: bool, tip_text: str).
    For /r/: lips slightly rounded, moderate opening.
    For /w/: lips very rounded & pushed out, very small opening.
    """
    if metrics is None:
        return None, ""
    ar = metrics["aspect_ratio"]
    openness = metrics["openness"]
    # /w/ tends to have very high aspect‑ratio (wide relative to tiny opening)
    # and lower openness  (lips more closed/pursed)
    if ar > 8.0 and openness < 0.35:
        return False, (
            "😯 It looks like your lips are too round & pushed out – "
            "that makes a **\"w\"** sound!\n\n"
            "**Try this:** Pull your lips back just a tiny bit (like a small smile) "
            "and curl the tip of your tongue UP toward the roof of your mouth. "
            "Think of a pirate saying \"Rrrrr!\" 🏴‍☠️"
        )
    return True, (
        "👍 Great lip shape! Keep your tongue curled up "
        "toward the roof of your mouth."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_DUO_GREEN = "#58CC02"
_DUO_RED = "#FF4B4B"
_DUO_GOLD = "#FFC800"
_DUO_BLUE = "#1CB0F6"
_DUO_BG = "#131F24"
_DUO_CARD = "#1B2B33"
_DUO_TEXT = "#FFFFFF"


def _make_heatmap(audio_score):
    """
    Return a matplotlib Figure with a target (green circle) and user dot (red→green).
    The dot moves closer to the centre as score increases.
    """
    fig, ax = plt.subplots(figsize=(4, 4), facecolor=_DUO_BG)
    ax.set_facecolor(_DUO_BG)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Target glow rings
    for r, a in [(0.55, 0.08), (0.45, 0.12), (0.35, 0.18)]:
        c = Circle((0, 0), r, color=_DUO_GREEN, alpha=a)
        ax.add_patch(c)
    target = Circle((0, 0), 0.25, linewidth=2.5,
                     edgecolor=_DUO_GREEN, facecolor="none")
    ax.add_patch(target)
    ax.text(0, 0, "🎯", ha="center", va="center", fontsize=22)

    # User dot position – farther away = worse score
    dist = 1.0 - (audio_score / 100.0)  # 0 = centre, 1 = edge
    angle = np.random.uniform(0, 2 * np.pi)
    ux, uy = dist * np.cos(angle), dist * np.sin(angle)
    dot_color = _DUO_GREEN if audio_score >= 70 else (_DUO_GOLD if audio_score >= 40 else _DUO_RED)
    ax.plot(ux, uy, "o", color=dot_color, markersize=16, markeredgecolor="white", markeredgewidth=2)
    ax.annotate("YOU", (ux, uy), textcoords="offset points", xytext=(12, 12),
                fontsize=10, fontweight="bold", color=dot_color,
                arrowprops=dict(arrowstyle="->", color=dot_color, lw=1.5))

    ax.set_title("Comparison Map", color=_DUO_TEXT, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _make_score_card(audio_score, xp, streak, is_r_shape):
    """Return a matplotlib Figure that looks like a Duolingo result card."""
    fig, ax = plt.subplots(figsize=(5, 3.4), facecolor=_DUO_BG)
    ax.set_facecolor(_DUO_BG)
    ax.axis("off")

    # clamp display
    display_score = int(round(audio_score))
    xp_gain = max(5, int(audio_score / 5))

    # Color
    if display_score >= 80:
        main_col = _DUO_GREEN
        emoji = "🌟"
        phrase = "Amazing!"
    elif display_score >= 60:
        main_col = _DUO_BLUE
        emoji = "👍"
        phrase = "Good job!"
    elif display_score >= 40:
        main_col = _DUO_GOLD
        emoji = "💪"
        phrase = "Keep trying!"
    else:
        main_col = _DUO_RED
        emoji = "🔄"
        phrase = "Let's try again!"

    ax.text(0.5, 0.88, f"{emoji}  {phrase}", transform=ax.transAxes,
            ha="center", va="center", fontsize=22, fontweight="bold", color=main_col)

    ax.text(0.5, 0.62, f"{display_score}% Match", transform=ax.transAxes,
            ha="center", va="center", fontsize=30, fontweight="bold", color=_DUO_TEXT)

    ax.text(0.5, 0.42, f"+{xp_gain} XP", transform=ax.transAxes,
            ha="center", va="center", fontsize=18, fontweight="bold", color=_DUO_GOLD)

    streak_text = f"🔥 Streak: {streak}" if streak > 0 else ""
    ax.text(0.5, 0.24, streak_text, transform=ax.transAxes,
            ha="center", va="center", fontsize=14, color="#FF9600")

    # mouth feedback one-liner
    if is_r_shape is True:
        mouth_line = "👄 Mouth shape: ✅ looks good!"
    elif is_r_shape is False:
        mouth_line = "👄 Mouth shape: ⚠️ too round – see tips below"
    else:
        mouth_line = "👄 Mouth shape: (no video captured)"
    ax.text(0.5, 0.08, mouth_line, transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#AABBCC")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE  EXERCISE  LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def run_exercise(audio_filepath, webcam_frame):
    """
    Called when the user clicks 'Check My Sound!'.
    audio_filepath – path to recorded .wav from gr.Audio
    webcam_frame   – numpy BGR frame from gr.Image(source="webcam") or None
    Returns: (heatmap_fig, score_card_fig, tip_html)
    """
    if audio_filepath is None:
        return None, None, _card("⚠️ No audio!", "Please record yourself saying <b>\"Right\"</b> first.", "warning")

    # ── 1. Audio analysis ────────────────────────────────────────────────────
    try:
        feat_user = get_speech_features(audio_filepath)
    except Exception as e:
        return None, None, _card("Audio Error", f"Could not process audio: {e}", "error")

    if os.path.exists(GOLD_AUDIO):
        feat_gold = get_speech_features(GOLD_AUDIO)
    else:
        # No gold file yet → use self‑comparison (score will be ~100 as placeholder)
        feat_gold = feat_user

    audio_score, dists, path = compare_speech(feat_gold, feat_user)

    # ── 2. Mouth analysis (if webcam frame provided) ─────────────────────────
    metrics = None
    if webcam_frame is not None:
        # Gradio webcam gives RGB numpy; convert to BGR for cv2/MP
        if isinstance(webcam_frame, np.ndarray):
            bgr = cv2.cvtColor(webcam_frame, cv2.COLOR_RGB2BGR)
            metrics = analyse_video_frame(bgr)

    is_r, mouth_tip = mouth_diagnostic(metrics)

    # ── 3. Update state ──────────────────────────────────────────────────────
    user_state["attempts"] += 1
    xp_gain = max(5, int(audio_score / 5))
    user_state["xp"] += xp_gain
    if audio_score >= 60:
        user_state["streak"] += 1
    else:
        user_state["streak"] = 0

    # ── 4. Build visuals ─────────────────────────────────────────────────────
    heatmap = _make_heatmap(audio_score)
    scorecard = _make_score_card(audio_score, user_state["xp"], user_state["streak"], is_r)

    # ── 5. Build HTML tips ───────────────────────────────────────────────────
    tip_html = ""
    if is_r is False:
        tip_html = _card(
            "🏴‍☠️ Pirate Tip!",
            mouth_tip.replace("\n", "<br>"),
            "tip"
        )
    elif is_r is True:
        tip_html = _card("✅ Mouth Shape", mouth_tip, "success")
    else:
        tip_html = _card("📷 No face detected",
                         "Try turning on your camera so I can check your mouth shape!",
                         "info")

    # Extra coaching if audio score is low
    if audio_score < 50:
        tip_html += _card(
            "🗣️ Sound Tip",
            "It sounds like you might be saying <b>\"Wight\"</b> instead of <b>\"Right\"</b>.<br><br>"
            "Here's a trick: <b>Growl like a tiger first</b> – "
            "<i>\"Grrrrr…\"</i> – then add <i>\"ight\"</i>.<br>"
            "So: <b>Grrr → Right!</b> 🐯",
            "tip"
        )

    return heatmap, scorecard, tip_html


def _card(title, body, kind="info"):
    colors = {
        "info": ("#1CB0F6", "#0D2B3E"),
        "tip": ("#FFC800", "#2B2200"),
        "success": ("#58CC02", "#0D2B00"),
        "warning": ("#FF9600", "#2B1A00"),
        "error": ("#FF4B4B", "#2B0000"),
    }
    accent, bg = colors.get(kind, colors["info"])
    return (
        f'<div style="background:{bg};border-left:4px solid {accent};'
        f'border-radius:12px;padding:16px 20px;margin:8px 0;'
        f'font-family:\'Nunito\',sans-serif;">'
        f'<div style="color:{accent};font-weight:800;font-size:1.1em;margin-bottom:6px;">{title}</div>'
        f'<div style="color:#E0E0E0;line-height:1.5;">{body}</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO  UI  –  Duolingo‑style theme
# ═══════════════════════════════════════════════════════════════════════════════

_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;800;900&display=swap');

/* ─── global ─── */
body, .gradio-container {
    background: #131F24 !important;
    font-family: 'Nunito', sans-serif !important;
    color: #FFFFFF !important;
}

/* ─── header banner ─── */
#banner {
    text-align: center;
    padding: 12px 0 4px 0;
}
#banner h1 {
    color: #58CC02;
    font-size: 2.4em;
    font-weight: 900;
    margin: 0;
    letter-spacing: -0.5px;
}
#banner p {
    color: #8899AA;
    font-size: 1.05em;
    margin: 2px 0 0 0;
}

/* ─── XP bar ─── */
#xp-bar {
    text-align: center;
    padding: 6px 0;
    font-size: 1.05em;
    color: #FFC800;
    font-weight: 800;
}

/* ─── cards / panels ─── */
.gr-panel, .gr-box, .gr-form, .gr-input, .gr-padded {
    background: #1B2B33 !important;
    border: 1px solid #2A3F4D !important;
    border-radius: 16px !important;
    color: #FFFFFF !important;
}

/* ─── instruction card ─── */
#instruction-card {
    background: #1B2B33;
    border: 2px solid #58CC02;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
}
#instruction-card h2 {
    color: #58CC02;
    font-weight: 900;
    font-size: 1.5em;
    margin: 0 0 4px 0;
}
#instruction-card h3 {
    color: #FFC800;
    font-weight: 800;
    font-size: 1.9em;
    margin: 6px 0;
}
#instruction-card p {
    color: #AABBCC;
    font-size: 1em;
}

/* ─── big record button ─── */
#record-btn button, #check-btn {
    background: #58CC02 !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.15em !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 12px 32px !important;
    box-shadow: 0 4px 0 #46A302 !important;
    transition: transform 0.1s !important;
}
#record-btn button:active, #check-btn:active {
    transform: translateY(3px) !important;
    box-shadow: 0 1px 0 #46A302 !important;
}

/* ─── results area ─── */
#results-col {
    background: #1B2B33;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #2A3F4D;
}

/* labels & text */
label, .gr-check-radio label, .label-wrap span {
    color: #AABBCC !important;
    font-weight: 700 !important;
}

/* Plots transparent bg */
.gr-plot {
    background: transparent !important;
}

/* ─── streak animation ─── */
@keyframes streakPulse {
    0%   { transform: scale(1); }
    50%  { transform: scale(1.15); }
    100% { transform: scale(1); }
}
.streak-anim {
    animation: streakPulse 0.6s ease-in-out 3;
    display: inline-block;
}

/* ─── syncing badge ─── */
#syncing-badge {
    text-align: center;
    padding: 10px;
}
"""

_INSTRUCTION_HTML = f"""
<div id="instruction-card">
    <h2>Lesson 1 &nbsp;·&nbsp; The /R/ Sound</h2>
    <h3>Say: &nbsp;"Right" &nbsp;👉</h3>
    <p>Look at the picture, then <b>record</b> yourself saying the word clearly!</p>
</div>
"""


def _xp_html():
    xp = user_state["xp"]
    streak = user_state["streak"]
    streak_span = (
        f'<span class="streak-anim">🔥 Streak {streak}</span>'
        if streak > 1 else (f"🔥 Streak {streak}" if streak == 1 else "")
    )
    return (
        f'<div id="xp-bar">⭐ {xp} XP &nbsp;&nbsp;&nbsp; {streak_span}</div>'
    )


def _syncing_html(visible=False):
    if not visible:
        return ""
    return (
        '<div id="syncing-badge">'
        '<span style="color:#1CB0F6;font-weight:800;font-size:1.1em;">'
        '🔄 Syncing audio + video …</span></div>'
    )


def on_check(audio, webcam_snapshot):
    """Wrapper that feeds Gradio inputs → run_exercise → Gradio outputs."""
    heatmap, scorecard, tips = run_exercise(audio, webcam_snapshot)
    return (
        heatmap,          # heatmap plot
        scorecard,        # score card plot
        tips,             # tips HTML
        _xp_html(),       # updated XP bar
    )


# ─── build the interface ──────────────────────────────────────────────────────

with gr.Blocks(title="SpeakQuest – Speech Therapy") as demo:

    # ── Banner ────────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="banner">'
        '<h1>🗣️ SpeakQuest</h1>'
        '<p>Your speech adventure starts here!</p>'
        '</div>'
    )
    xp_bar = gr.HTML(value=_xp_html(), elem_id="xp-bar-container")

    # ── Main two‑column layout ────────────────────────────────────────────────
    with gr.Row(equal_height=False):
        # LEFT COLUMN – Instruction + capture
        with gr.Column(scale=1, min_width=380):
            gr.HTML(_INSTRUCTION_HTML)
            gr.Image(
                value=IMAGE_PATH if os.path.exists(IMAGE_PATH) else None,
                label="Visual Cue",
                type="filepath",
                interactive=False,
                height=260,
            )

            gr.Markdown("### 🎤 Record yourself saying **\"Right\"**")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Your Recording",
                elem_id="record-btn",
            )

            gr.Markdown("### 📷 Webcam snapshot *(optional – helps check mouth shape)*")
            webcam_input = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Webcam",
                height=220,
            )

            check_btn = gr.Button("✨ Check My Sound!", elem_id="check-btn", size="lg")

        # RIGHT COLUMN – Results
        with gr.Column(scale=1, min_width=380, elem_id="results-col"):
            gr.Markdown("## 📊 Results")
            syncing_box = gr.HTML(value="", elem_id="syncing-area")
            heatmap_plot = gr.Plot(label="Comparison Map")
            score_plot = gr.Plot(label="Score")
            tips_html = gr.HTML(value=_card(
                "👋 Ready?",
                "Record yourself saying <b>\"Right\"</b> and click <b>Check My Sound!</b>",
                "info",
            ))

    # ── Event wiring ──────────────────────────────────────────────────────────
    check_btn.click(
        fn=lambda: _syncing_html(True),
        inputs=None,
        outputs=syncing_box,
    ).then(
        fn=on_check,
        inputs=[audio_input, webcam_input],
        outputs=[heatmap_plot, score_plot, tips_html, xp_bar],
    ).then(
        fn=lambda: _syncing_html(False),
        inputs=None,
        outputs=syncing_box,
    )

# ─── launch ───────────────────────────────────────────────────────────────────
def _cleanup():
    """Release all resources on shutdown."""
    print("\n🧹 Cleaning up …")
    plt.close("all")
    cv2.destroyAllWindows()
    try:
        demo.close()          # tells Gradio server to stop → browser drops mic/cam
    except Exception:
        pass
    print("✅ All resources released. Goodbye!")


atexit.register(_cleanup)


def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    _cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

if __name__ == "__main__":
    demo.launch(share=False, css=_CUSTOM_CSS, theme=gr.themes.Base())
