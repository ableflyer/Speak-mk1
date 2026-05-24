import librosa
import soundfile as sf
import numpy as np
from scipy.signal import lfilter

def extract_formants(frame, sr):
    # 1. Apply Windowing (Hanning/Hamming) to the frame
    windowed_frame = frame * np.hamming(len(frame))
    
    # 2. Get LPC Coefficients 
    # usually we use the Rule of thumb: order = 2 + (sampling_rate / 1000)
    # but it might find fake noise if we put it at 50 since the max sound that we could get is 24kHz
    lpc_coeffs = librosa.lpc(windowed_frame, order=27)
    
    # 3. Find roots of the polynomial
    roots = np.roots(lpc_coeffs)
    
    # 4. Filter out roots that are in the bottom half of the Z-plane (complex conjugates)
    roots = [r for r in roots if np.imag(r) > 0]
    
    # 5. Convert roots to frequencies (Hz)
    angz = np.arctan2(np.imag(roots), np.real(roots))
    frqs = sorted(angz * (sr / (2 * np.pi)))
    
    # 6. Return the first three formants (F1, F2, F3)
    # Use 0 as a placeholder if fewer than 3 are found
    return (frqs[:3] + [0, 0, 0])[:3]

def extract_fractals(frame):
    # 1. Calculate the total path length (sum of distances between successive points)
    dists = np.sqrt(1 + np.diff(frame)**2)
    L = np.sum(dists)
    
    # 2. Calculate the "Diameter" (max distance from the first point)
    # Using simple Euclidean distance for a 1D signal
    d = np.max(np.abs(frame - frame[0]))
    
    # 3. Number of steps
    n = len(frame) - 1
    
    # 4. The Formula
    if d == 0: return 0 # Avoid log(0)
    return np.log10(n) / (np.log10(n) + np.log10(d / L))

def detect_voice_activity(y, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    return zcr.ravel()

def extract_fast_energy_distribution(y, sr, n_mels=40):
    # FBANKs are the standard for modern "Native Audio" models
    # It provides a rich energy map that Transformers can attend to
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_melspec = librosa.power_to_db(melspec) 
    
    # Return the current frame
    return log_melspec

def extract_subband_derivatives(y, sr, n_mels=40):
    # 1. Calculate Mel Spectrogram (Energy in sub-bands)
    # We use power=1 for energy, power=2 for power
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=1)
    
    # 2. Convert to Decibels (Log scaling)
    # This mimics human hearing and makes derivatives more stable
    log_S = librosa.amplitude_to_db(S)
    
    # 3. Calculate the Derivative (Delta)
    # librosa.feature.delta calculates: S[t] - S[t-1]
    # width=3 is the smallest window for a first-order derivative
    subband_deltas = librosa.feature.delta(log_S, width=3, order=1)
    
    # 4. Half-Wave Rectification (Optional but recommended for burst detection)
    # We only care about sudden increases in energy
    burst_features = np.maximum(0, subband_deltas)
    
    return burst_features

def extract_vot(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'), 
                                                sr=sr)

    return voiced_probs

def main():
    # 1. LOAD & PRE-PROCESS (The Diagram: Pre-emphasis filter)
    y, sr = librosa.load("../Data/beard.m4a", sr=48000)
    y_pre = librosa.effects.preemphasis(y, coef=0.97)

    # 2. EXTRACT CONTINUOUS FEATURES
    zcr = detect_voice_activity(y_pre, 2048, 512)            # VAD (frames,)
    articulation_map = extract_fast_energy_distribution(y_pre, sr).T # Log-Mel (frames, 40)
    v_probs = extract_vot(y_pre, sr)               # Continuous Voicing (frames,)
    
    # 3. EXTRACT SLICE-BASED FEATURES
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    frames = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length)
    
    all_formants = []
    all_fractals = []
    for i in range(frames.shape[1]):
        all_formants.append(extract_formants(frames[:, i], sr))
        all_fractals.append(extract_fractals(frames[:, i]))
    
    all_formants = np.array(all_formants) # (frames, 3)
    all_fractals = np.array(all_fractals) # (frames,)

    # 4. THE CONCATENATION (The "Frame-Level Concatenation" Block)
    # Find the shortest length to prevent index errors
    min_len = min(len(zcr), len(v_probs), all_formants.shape[0], articulation_map.shape[0])
    
    # Stack everything as columns
    expert_features = np.column_stack((
        zcr[:min_len],                # [0] ZCR
        v_probs[:min_len],            # [1] Continuous Voicing (VOT hint)
        all_fractals[:min_len],       # [2] Katz Fractal
        all_formants[:min_len, 0],    # [3] F1
        all_formants[:min_len, 1],    # [4] F2
        all_formants[:min_len, 2],    # [5] F3
        articulation_map[:min_len]    # [6-45] Log-Mel (40 bands)
    ))

    print(f"Expert Feature Matrix Shape: {expert_features.shape}") 
    # This shape (N, 46) is exactly what the Transformer Encoder needs.
    return expert_features

if __name__ == "__main__":
    expert_features = main()



