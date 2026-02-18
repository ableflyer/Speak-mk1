import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import sounddevice as sd
from scipy.io.wavfile import write
import cv2
import mediapipe as mp
import time
import os

# 1. Load the "Ear" (Pre-trained on 53 languages for high phonetic accuracy)
model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

def get_speech_features(audio_file):
    # 2. Load audio (Wav2Vec2 REQUIRES 16kHz)
    speech, sr = librosa.load(audio_file, sr=16000)
    
    # 2.5 Trim silence from beginning and end so we only analyze actual speech
    speech_trimmed, _ = librosa.effects.trim(speech, top_db=20)
    
    # 3. Pre-process (Convert to tensors)
    input_values = feature_extractor(speech_trimmed, return_tensors="pt", sampling_rate=16000).input_values
    
    # 4. Extract Features
    with torch.no_grad():
        outputs = model(input_values)
    
    # This is the "Acoustic Fingerprint" [Batch, Time, Hidden_Size]
    features = outputs.last_hidden_state.squeeze(0).numpy()
    
    # 5. Normalize features per frame so we compare phonetic shape, not volume/speaker
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid divide by zero
    features = features / norms
    
    # 6. Compute delta features (how the sound CHANGES between frames)
    # This is critical: "s" -> "n" transition is very different from "th" -> "n" transition
    # Delta features capture these transitions, which is where mispronunciations show up
    deltas = np.diff(features, axis=0)  # First derivative (velocity of change)
    delta_deltas = np.diff(deltas, axis=0)  # Second derivative (acceleration of change)
    
    # Pad deltas to match original length
    deltas = np.vstack([deltas, deltas[-1:]])
    delta_deltas = np.vstack([delta_deltas, delta_deltas[-1:], delta_deltas[-1:]])
    
    # 7. Concatenate: original features + deltas + delta-deltas
    # This triples the feature size but makes phonetic differences MUCH more visible
    combined = np.concatenate([features, deltas * 2.0, delta_deltas * 4.0], axis=1)
    
    return combined

# ============================================================
# MOUTH TRACKING WITH MEDIAPIPE TASKS API
# ============================================================

# Path to the face landmarker model (download from MediaPipe)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# MediaPipe Tasks API types
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe mouth landmark indices (lips outer + inner)
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
                     185, 40, 39, 37, 0, 267, 269, 270, 409]
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                     191, 80, 81, 82, 13, 312, 311, 310, 415]

# Key measurement landmarks
UPPER_LIP_TOP = 13       # Top of upper lip (inner)
LOWER_LIP_BOTTOM = 14    # Bottom of lower lip (inner)
LEFT_CORNER = 61         # Left corner of mouth
RIGHT_CORNER = 291       # Right corner of mouth
UPPER_LIP_OUTER = 0      # Top of upper lip (outer)
LOWER_LIP_OUTER = 17     # Bottom of lower lip (outer)

def extract_mouth_metrics(landmarks, frame_w, frame_h):
    """
    Extract meaningful mouth shape metrics from face landmarks.
    landmarks: list of NormalizedLandmark from MediaPipe Tasks API
    """
    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])
    
    upper_inner = get_point(UPPER_LIP_TOP)
    lower_inner = get_point(LOWER_LIP_BOTTOM)
    left_corner = get_point(LEFT_CORNER)
    right_corner = get_point(RIGHT_CORNER)
    upper_outer = get_point(UPPER_LIP_OUTER)
    lower_outer = get_point(LOWER_LIP_OUTER)
    
    mouth_width = np.linalg.norm(right_corner - left_corner)
    inner_height = np.linalg.norm(lower_inner - upper_inner)
    outer_height = np.linalg.norm(lower_outer - upper_outer)
    aspect_ratio = mouth_width / max(inner_height, 1e-6)
    openness = inner_height / max(outer_height, 1e-6)
    lip_thickness = (outer_height - inner_height) / max(mouth_width, 1e-6)
    
    mouth_center = (upper_inner + lower_inner) / 2
    inner_points = np.array([get_point(i) for i in INNER_LIP_INDICES])
    inner_shape = (inner_points - mouth_center) / max(mouth_width, 1e-6)
    shape_signature = inner_shape.flatten()
    
    return {
        'mouth_width': mouth_width,
        'inner_height': inner_height,
        'outer_height': outer_height,
        'aspect_ratio': aspect_ratio,
        'openness': openness,
        'lip_thickness': lip_thickness,
        'shape_signature': shape_signature,
    }

def record_with_face_tracking(duration=3, filename="recorded_audio.wav"):
    """
    Record audio AND track mouth movements simultaneously using
    MediaPipe Tasks API (FaceLandmarker) in LIVE_STREAM mode.
    """
    sample_rate = 16000
    mouth_data = []
    latest_result = [None]  # Use list to allow mutation in callback
    
    # --- Callback for live stream results ---
    def on_result(result, output_image, timestamp_ms):
        latest_result[0] = result
    
    # --- Check model file exists ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print("   Falling back to audio-only mode.")
        return record_from_microphone(duration, filename), None
    
    # --- Create FaceLandmarker with LIVE_STREAM mode ---
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        result_callback=on_result,
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera! Falling back to audio-only mode.")
        return record_from_microphone(duration, filename), None
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("🎤📷 Initializing microphone + camera...")
    
    # Start audio recording (with warmup)
    warmup_duration = 1.0
    total_samples = int((duration + warmup_duration) * sample_rate)
    
    audio = sd.rec(total_samples,
                   samplerate=sample_rate,
                   channels=1,
                   dtype='float32')
    
    time.sleep(warmup_duration)
    
    print(f"🔴 SPEAK NOW! Recording for {duration} seconds...")
    print("   (Look at the camera so your mouth is visible)")
    
    start_time = time.time()
    frame_timestamp_ms = 0
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            
            # Convert frame to MediaPipe Image and send for async detection
            frame_timestamp_ms = int(elapsed * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # Process the latest result (from callback)
            if latest_result[0] is not None and latest_result[0].face_landmarks:
                landmarks = latest_result[0].face_landmarks[0]  # First face
                
                # Extract mouth metrics
                metrics = extract_mouth_metrics(landmarks, frame_w, frame_h)
                metrics['timestamp'] = elapsed
                mouth_data.append(metrics)
                
                # Draw mouth landmarks on frame for visual feedback
                for idx in OUTER_LIP_INDICES + INNER_LIP_INDICES:
                    lm = landmarks[idx]
                    x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Draw mouth opening measurement line
                upper = landmarks[UPPER_LIP_TOP]
                lower = landmarks[LOWER_LIP_BOTTOM]
                cv2.line(frame,
                         (int(upper.x * frame_w), int(upper.y * frame_h)),
                         (int(lower.x * frame_w), int(lower.y * frame_h)),
                         (0, 0, 255), 2)
                
                # Show live metrics on screen
                cv2.putText(frame, f"Opening: {metrics['inner_height']:.1f}px", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Width: {metrics['mouth_width']:.1f}px",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Aspect: {metrics['aspect_ratio']:.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show countdown
            cv2.putText(frame, f"REC {remaining:.1f}s", (frame_w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Mouth Tracking - Press Q to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    sd.wait()
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Recording complete! Captured {len(mouth_data)} mouth frames.")
    
    # Trim warmup from audio
    valid_start_index = int((warmup_duration - 0.5) * sample_rate)
    if valid_start_index < 0:
        valid_start_index = 0
    audio = audio[valid_start_index:]
    
    # Save audio
    audio_int16 = np.int16(audio * 32767)
    write(filename, sample_rate, audio_int16)
    print(f"💾 Audio saved to: {filename}")
    
    return filename, mouth_data

def compare_mouth_shapes(mouth_data_correct, mouth_data_attempt):
    """
    Compare mouth movement patterns between correct and attempt recordings.
    Returns a mouth similarity score and per-frame details.
    """
    if not mouth_data_correct or not mouth_data_attempt:
        return None, None, None
    
    # Extract time series of key metrics
    def get_metric_series(mouth_data):
        aspect_ratios = [m['aspect_ratio'] for m in mouth_data]
        openness = [m['openness'] for m in mouth_data]
        widths = [m['mouth_width'] for m in mouth_data]
        heights = [m['inner_height'] for m in mouth_data]
        thickness = [m['lip_thickness'] for m in mouth_data]
        return np.column_stack([aspect_ratios, openness, widths, heights, thickness])
    
    series_correct = get_metric_series(mouth_data_correct)
    series_attempt = get_metric_series(mouth_data_attempt)
    
    # Normalize each metric column to 0-1 range for fair comparison
    for col in range(series_correct.shape[1]):
        all_vals = np.concatenate([series_correct[:, col], series_attempt[:, col]])
        vmin, vmax = all_vals.min(), all_vals.max()
        rng = vmax - vmin if vmax - vmin > 1e-6 else 1.0
        series_correct[:, col] = (series_correct[:, col] - vmin) / rng
        series_attempt[:, col] = (series_attempt[:, col] - vmin) / rng
    
    # Use DTW to align the two mouth movement sequences
    from scipy.spatial.distance import cdist
    cost = cdist(series_correct, series_attempt, metric='euclidean')
    
    n, m = cost.shape
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i-1, j-1] + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # Backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [(dtw[i-1, j-1], i-1, j-1), (dtw[i-1, j], i-1, j), (dtw[i, j-1], i, j-1)]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()
    
    path_distances = np.array([cost[i, j] for i, j in path])
    avg_dist = np.mean(path_distances)
    
    # Scale: typical euclidean distances on normalized 5D vectors
    mouth_score = max(0, min(100, (1 - avg_dist / 0.5) * 100))
    
    return mouth_score, path_distances, {
        'series_correct': series_correct,
        'series_attempt': series_attempt,
    }

def compare_speech(features_correct, features_attempt):
    """
    Compare two speech feature sets and return a similarity analysis.
    Uses Dynamic Time Warping (DTW) to handle differences in speaking speed.
    AGGRESSIVE mode: amplifies phonetic differences so "thnake" vs "snake" is very different.
    
    Returns:
        similarity_score: 0-100 (100 = perfect match)
        distance_over_time: per-frame distance showing WHERE the speech differs
    """
    from scipy.spatial.distance import cdist
    
    # 1. Compute cost matrix between every frame of correct vs attempt
    cost_matrix = cdist(features_correct, features_attempt, metric='cosine')
    
    # 2. Dynamic Time Warping (DTW) - aligns the two signals even if
    #    they were spoken at different speeds
    n, m = cost_matrix.shape
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost_matrix[i-1, j-1] + min(
                dtw[i-1, j],     # insertion
                dtw[i, j-1],     # deletion
                dtw[i-1, j-1]    # match
            )
    
    # 3. Backtrack to find the optimal alignment path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [
            (dtw[i-1, j-1], i-1, j-1),
            (dtw[i-1, j],   i-1, j),
            (dtw[i, j-1],   i,   j-1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()
    
    # 4. Compute per-frame distance along the aligned path
    path_distances = np.array([cost_matrix[i, j] for i, j in path])
    
    # 5. AGGRESSIVE scoring
    avg_distance = np.mean(path_distances)
    
    # Amplify the worst frames: penalize peaks heavily
    # The 90th percentile captures the "problem sounds" (like the "th" vs "s")
    top_10_pct = np.percentile(path_distances, 90)
    
    # Weighted score: 50% average distance + 50% worst-case distance
    # This ensures a single bad sound (like "th" instead of "s") tanks the score
    blended_distance = avg_distance * 0.5 + top_10_pct * 0.5
    
    # Very tight scale: 
    # 0.02 or less = 100 (nearly identical recordings)
    # 0.05 = ~70 (minor differences)
    # 0.08 = ~40 (clear mispronunciation like thnake/snake)
    # 0.12+ = 0 (completely different)
    similarity_score = max(0, min(100, (1 - blended_distance / 0.10) * 100))
    
    return similarity_score, path_distances, path

def visualize_comparison(features_correct, features_attempt, path_distances, similarity_score, path,
                         mouth_score=None, mouth_distances=None, mouth_details=None):
    """
    Create a visual report showing where the two pronunciations differ.
    """
    has_mouth = mouth_score is not None and mouth_distances is not None
    num_plots = 5 if has_mouth else 4
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots))
    
    # Plot 1: Correct pronunciation features
    ax1 = axes[0]
    ax1.imshow(features_correct.T, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title('✅ Correct Pronunciation (Target)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature Dim')
    
    # Plot 2: Attempt pronunciation features
    ax2 = axes[1]
    ax2.imshow(features_attempt.T, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('🎤 Your Pronunciation (Attempt)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature Dim')
    
    # Plot 3: Audio difference
    ax3 = axes[2]
    colors = plt.cm.RdYlGn_r(path_distances / max(path_distances.max(), 0.01))
    ax3.bar(range(len(path_distances)), path_distances, color=colors, width=1.0)
    ax3.set_title('� Audio Difference (Red = Mismatch)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cosine Distance')
    threshold = np.mean(path_distances) + np.std(path_distances)
    ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label='Problem threshold')
    ax3.legend()
    
    # Plot 4 (optional): Mouth shape difference
    if has_mouth:
        ax_mouth = axes[3]
        m_colors = plt.cm.RdYlGn_r(mouth_distances / max(mouth_distances.max(), 0.01))
        ax_mouth.bar(range(len(mouth_distances)), mouth_distances, color=m_colors, width=1.0)
        ax_mouth.set_title('👄 Mouth Shape Difference (Red = Wrong Shape)', fontsize=12, fontweight='bold')
        ax_mouth.set_ylabel('Shape Distance')
        m_threshold = np.mean(mouth_distances) + np.std(mouth_distances)
        ax_mouth.axhline(y=m_threshold, color='red', linestyle='--', alpha=0.7, label='Problem threshold')
        ax_mouth.legend()
        summary_ax = axes[4]
    else:
        summary_ax = axes[3]
    
    # Summary plot
    summary_ax.axis('off')
    
    # Combined score
    if has_mouth:
        combined_score = similarity_score * 0.6 + mouth_score * 0.4
        score_text = (
            f"🔊 Audio Score: {similarity_score:.1f}/100\n"
            f"👄 Mouth Score: {mouth_score:.1f}/100\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Combined Score: {combined_score:.1f}/100"
        )
    else:
        combined_score = similarity_score
        score_text = f"🎯 Audio Score: {similarity_score:.1f}/100"
    
    # Determine feedback
    if combined_score >= 85:
        grade = "🌟 EXCELLENT!"
        color = 'green'
        feedback = "Your pronunciation is very close to the target!"
    elif combined_score >= 70:
        grade = "👍 GOOD"
        color = 'orange'
        feedback = "Pretty close! Some sounds need a little work."
    elif combined_score >= 50:
        grade = "⚠️ NEEDS WORK"
        color = 'darkorange'
        feedback = "There are noticeable differences. Focus on the red areas above."
    else:
        grade = "🔴 TRY AGAIN"
        color = 'red'
        feedback = "Significant differences detected. Let's practice more!"
    
    # Find audio problem regions
    problem_frames = np.where(path_distances > threshold)[0]
    total_frames = len(path_distances)
    if len(problem_frames) > 0:
        problem_start_pct = (problem_frames[0] / total_frames) * 100
        problem_end_pct = (problem_frames[-1] / total_frames) * 100
        problem_text = f"🔊 Audio problem: ~{problem_start_pct:.0f}%-{problem_end_pct:.0f}% through the word"
    else:
        problem_text = "🔊 No major audio problems detected!"
    
    # Find mouth problem regions
    if has_mouth:
        m_problem_frames = np.where(mouth_distances > m_threshold)[0]
        m_total = len(mouth_distances)
        if len(m_problem_frames) > 0:
            mp_start = (m_problem_frames[0] / m_total) * 100
            mp_end = (m_problem_frames[-1] / m_total) * 100
            problem_text += f"\n👄 Mouth problem: ~{mp_start:.0f}%-{mp_end:.0f}% through the word"
        else:
            problem_text += "\n👄 Mouth shapes look good!"
    
    summary = (
        f"{score_text}\n\n"
        f"Grade: {grade}\n"
        f"{feedback}\n\n"
        f"{problem_text}"
    )
    summary_ax.text(0.5, 0.5, summary, transform=summary_ax.transAxes,
             fontsize=13, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor=color, linewidth=3))
    
    plt.tight_layout()
    plt.savefig('speech_comparison.png', dpi=150)
    print("📈 Comparison saved to: speech_comparison.png")
    plt.show()

def record_from_microphone(duration=7, filename="recorded_audio.wav"):
    """
    Record audio from the microphone and save it as a .wav file.
    
    Args:
        duration: Recording duration in seconds (default: 3)
        filename: Output filename (default: "recorded_audio.wav")
    
    Returns:
        filename: Path to the saved .wav file
    """
    import time
    sample_rate = 16000  # Wav2Vec2 requires 16kHz
    
    print("🎤 Initializing microphone (please wait)...")
    
    # STRATEGY: Start recording silently *before* prompting the user.
    # This solves the "startup lag" where the first word gets cut off.
    warmup_duration = 1.0 
    total_samples = int((duration + warmup_duration) * sample_rate)
    
    # Start recording in background (non-blocking) - this starts the hardware
    print("   Starting stream...")
    audio = sd.rec(total_samples, 
                   samplerate=sample_rate, 
                   channels=1,  # Mono audio
                   dtype='float32')
                   
    # Wait for the hardware to actually spin up (warmup)
    time.sleep(warmup_duration)
    
    # NOW tell the user to speak (recording is already running!)
    print(f"🔴 SPEAK NOW! Recording for {duration} seconds...")
    
    # Show usage progress bar for user feedback
    for i in range(duration, 0, -1):
        print(f"   ⏳ {i}s remaining...", end="\r")
        time.sleep(1)
    
    # Wait for the remaining time (if any)
    sd.wait()
    print("\n✅ Recording complete!")
    
    # Discard the silence from the warmup period which happened before we said "Speak"
    # We keep the last 0.5 seconds of the warmup just to be safe (catch the very start)
    valid_start_index = int((warmup_duration - 0.5) * sample_rate)
    if valid_start_index < 0:
        valid_start_index = 0
        
    audio = audio[valid_start_index:]
    
    # Convert to int16 for .wav file (standard format)
    audio_int16 = np.int16(audio * 32767)
    
    # Save as .wav file
    write(filename, sample_rate, audio_int16)
    print(f"💾 Saved to: {filename}")
    
    return filename

# Example Usage:
# features_correct = get_speech_features("snake_correct.wav")
# features_error = get_speech_features("thnake_error.wav")

# NEW: Record from microphone and analyze!
if __name__ == "__main__":
    word = input("📝 What word are you practicing? (e.g. 'snake'): ").strip()
    duration = int(input("⏱️  How many seconds per recording? (e.g. 3): ").strip())
    
    # --- Step 1: Record the CORRECT pronunciation ---
    print(f"\n{'='*50}")
    print(f"STEP 1: Say '{word}' CORRECTLY")
    print(f"  (Look at the camera so your mouth is visible)")
    print(f"{'='*50}")
    input("Press Enter when ready...")
    correct_file, mouth_correct = record_with_face_tracking(duration=duration, filename="correct.wav")
    
    print(f"\n🧠 Analyzing correct pronunciation...")
    features_correct = get_speech_features(correct_file)
    print(f"   Audio features shape: {features_correct.shape}")
    if mouth_correct:
        print(f"   Mouth frames captured: {len(mouth_correct)}")
    
    # --- Step 2: Record the ATTEMPT pronunciation ---
    print(f"\n{'='*50}")
    print(f"STEP 2: Now say '{word}' the way you normally say it")
    print(f"  (Keep looking at the camera!)")
    print(f"{'='*50}")
    input("Press Enter when ready...")
    attempt_file, mouth_attempt = record_with_face_tracking(duration=duration, filename="attempt.wav")
    
    print(f"\n🧠 Analyzing your attempt...")
    features_attempt = get_speech_features(attempt_file)
    print(f"   Audio features shape: {features_attempt.shape}")
    if mouth_attempt:
        print(f"   Mouth frames captured: {len(mouth_attempt)}")
    
    # --- Step 3: Compare Audio ---
    print(f"\n{'='*50}")
    print("STEP 3: Comparing pronunciations...")
    print(f"{'='*50}")
    
    similarity, distances, path = compare_speech(features_correct, features_attempt)
    
    print(f"\n🔊 Audio Similarity: {similarity:.1f}/100")
    print(f"   Avg cosine distance: {np.mean(distances):.4f}")
    print(f"   Max cosine distance: {np.max(distances):.4f}")
    
    # --- Step 4: Compare Mouth Shapes ---
    mouth_score = None
    mouth_distances = None
    mouth_details = None
    
    if mouth_correct and mouth_attempt:
        print(f"\n👄 Comparing mouth shapes...")
        mouth_score, mouth_distances, mouth_details = compare_mouth_shapes(mouth_correct, mouth_attempt)
        if mouth_score is not None:
            print(f"   Mouth Similarity: {mouth_score:.1f}/100")
            combined = similarity * 0.6 + mouth_score * 0.4
            print(f"\n🎯 COMBINED SCORE: {combined:.1f}/100  (60% audio + 40% mouth)")
        else:
            print("   ⚠️ Could not compare mouth data")
    else:
        print("\n⚠️ No mouth data captured - showing audio-only results")
    
    # --- Step 5: Visualize ---
    visualize_comparison(features_correct, features_attempt, distances, similarity, path,
                         mouth_score, mouth_distances, mouth_details)