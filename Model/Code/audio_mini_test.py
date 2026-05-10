
"""
test_audio_encoder.py
=====================
Standalone diagnostic script for SpeakMK1 AudioEncoder.
Tests encoder with real WAV files to verify:
    1. Feature extraction quality (real vs silence)
    2. Layer-by-layer activation statistics
    3. QFormer vs DirectProjection behavior
    4. Checkpoint loading integrity

Usage:
    python test_audio_encoder.py --wav /path/to/audio.wav --ckpt /path/to/encoder.pt
    python test_audio_encoder.py --wav /path/to/audio.wav --ckpt /path/to/encoder.pt --compare-silence
    python test_audio_encoder.py --librispeech /path/to/LibriSpeech/test-clean --ckpt /path/to/encoder.pt --num-samples 10
"""

import argparse
import os
import glob
import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import torchaudio
from dataclasses import dataclass

# ── Local imports ────────────────────────────────────────────────────────
# Adjust these paths based on your project structure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_encoder import AudioEncoder, AudioEncoderConfig


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TestConfig:
    sample_rate: int = 16_000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_frames: int = 512
    
    # Model dims — must match your trained encoder
    d_model: int = 512      # or 1024 if that's what you trained with
    dstate: int = 64
    d_conv: int = 4
    expand: int = 2
    num_uni_mamba: int = 6
    num_bi_mamba_moe: int = 4
    num_experts: int = 4
    attn_heads: int = 8      # 512/8=64 or 1024/16=64
    window_size: int = 64
    cgmlp_expansion: int = 4
    num_queries: int = 64
    num_qformer_layers: int = 2
    llm_dim: int = 512       # or 4096
    training_heads: bool = True
    dropout: float = 0.1


# ════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_wav_to_mel(wav_path: str, cfg: TestConfig, device: torch.device) -> torch.Tensor:
    """Load WAV file and convert to mel spectrogram."""
    y, sr = librosa.load(wav_path, sr=cfg.sample_rate)
    waveform = torch.from_numpy(y).unsqueeze(0)  # (1, samples)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    mel = mel_transform(waveform)
    mel = db_transform(mel).squeeze(0).T  # (T, n_mels)
    
    T = mel.shape[0]
    if T >= cfg.max_frames:
        mel = mel[:cfg.max_frames]
    else:
        mel = F.pad(mel, (0, 0, 0, cfg.max_frames - T))
    
    return mel.unsqueeze(0).to(device)  # (1, max_frames, n_mels)


def create_zero_mel(cfg: TestConfig, device: torch.device) -> torch.Tensor:
    """Create a zero mel for silence comparison."""
    return torch.zeros(1, cfg.max_frames, cfg.n_mels).to(device)


def create_noise_mel(cfg: TestConfig, device: torch.device, scale: float = 1.0) -> torch.Tensor:
    """Create random noise mel for baseline comparison."""
    return torch.randn(1, cfg.max_frames, cfg.n_mels).to(device) * scale


# ════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def analyze_tensor(name: str, tensor: torch.Tensor, show_hist: bool = False):
    """Print comprehensive statistics for a tensor."""
    flat = tensor.detach().cpu().float().flatten()
    
    print(f"\\n{name}")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  mean: {flat.mean():.6f}")
    print(f"  std:  {flat.std():.6f}")
    print(f"  min:  {flat.min():.6f}")
    print(f"  max:  {flat.max():.6f}")
    print(f"  median: {flat.median():.6f}")
    print(f"  abs mean: {flat.abs().mean():.6f}")
    print(f"  % zeros: {(flat == 0).float().mean()*100:.2f}%")
    print(f"  % near-zero (<1e-6): {(flat.abs() < 1e-6).float().mean()*100:.2f}%")
    print(f"  % positive: {(flat > 0).float().mean()*100:.2f}%")
    
    # Check for dead neurons (all same value across batch/time)
    if tensor.dim() >= 3:
        # Check variance across time dimension
        time_var = tensor.var(dim=1).mean().item()
        print(f"  mean time-variance: {time_var:.6f}")
        
        # Check variance across feature dimension
        feat_var = tensor.var(dim=-1).mean().item()
        print(f"  mean feature-variance: {feat_var:.6f}")
    
    if show_hist:
        # Print rough histogram
        hist, bins = torch.histogram(flat, bins=10)
        print(f"  histogram:")
        for i in range(len(hist)):
            print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}): {hist[i].item()} ({hist[i].item()/len(flat)*100:.1f}%)")
    
    # Check for NaN/Inf
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0:
        print(f"  ⚠️  NaN detected: {nan_count} values")
    if inf_count > 0:
        print(f"  ⚠️  Inf detected: {inf_count} values")


def compare_features(feat_a: torch.Tensor, feat_b: torch.Tensor, name_a: str = "A", name_b: str = "B"):
    """Compare two feature tensors comprehensively."""
    print(f"\\n{'='*60}")
    print(f"COMPARISON: {name_a} vs {name_b}")
    print(f"{'='*60}")
    
    # Basic stats
    mean_a = feat_a.mean().item()
    mean_b = feat_b.mean().item()
    std_a = feat_a.std().item()
    std_b = feat_b.std().item()
    
    print(f"\\nMean: {mean_a:.6f} vs {mean_b:.6f} (diff: {abs(mean_a-mean_b):.6f})")
    print(f"Std:  {std_a:.6f} vs {std_b:.6f} (diff: {abs(std_a-std_b):.6f})")
    
    # Cosine similarity (mean pooled across time)
    a_pooled = feat_a.mean(dim=1)  # (B, D)
    b_pooled = feat_b.mean(dim=1)
    cos_sim = F.cosine_similarity(a_pooled, b_pooled, dim=-1)
    print(f"\\nCosine similarity (mean-pooled): {cos_sim.item():.6f}")
    
    # Per-frame cosine similarity
    B, T, D = feat_a.shape
    frame_sims = []
    for t in range(T):
        sim = F.cosine_similarity(feat_a[:, t, :], feat_b[:, t, :], dim=-1)
        frame_sims.append(sim.item())
    
    print(f"Per-frame cosine similarity:")
    print(f"  mean: {np.mean(frame_sims):.6f}")
    print(f"  std:  {np.std(frame_sims):.6f}")
    print(f"  min:  {np.min(frame_sims):.6f}")
    print(f"  max:  {np.max(frame_sims):.6f}")
    
    # L2 distance
    l2_dist = torch.norm(feat_a - feat_b, p=2, dim=-1).mean().item()
    print(f"\\nMean L2 distance per frame: {l2_dist:.6f}")
    
    # Relative difference
    rel_diff = (feat_a - feat_b).abs() / (feat_a.abs() + feat_b.abs() + 1e-8)
    print(f"Mean relative difference: {rel_diff.mean().item():.6f}")
    
    # Are they statistically the same?
    if cos_sim.item() > 0.99 and l2_dist < 0.1:
        print(f"\\n🔴 VERDICT: Features are NEARLY IDENTICAL — encoder is NOT extracting meaningful features!")
    elif cos_sim.item() > 0.9:
        print(f"\\n🟡 WARNING: Very high similarity — encoder may be weak or untrained")
    elif cos_sim.item() < 0.5:
        print(f"\\n🟢 GOOD: Features are distinct — encoder is differentiating inputs")
    else:
        print(f"\\n🟡 MODERATE: Some differentiation but could be better")
    
    return cos_sim.item(), l2_dist


# ════════════════════════════════════════════════════════════════════════════
# LAYER-BY-LAYER DIAGNOSTIC
# ════════════════════════════════════════════════════════════════════════════

def layer_by_layer_diagnostic(model: AudioEncoder, mel: torch.Tensor, name: str = "real"):
    """Run encoder step by step and print stats at each layer."""
    print(f"\\n{'='*60}")
    print(f"LAYER-BY-LAYER DIAGNOSTIC ({name})")
    print(f"{'='*60}")
    
    model.eval()
    with torch.no_grad():
        # Stem
        x = model.stem(mel)
        analyze_tensor("After Stem", x)
        
        # Uni-Mamba layers
        for i, layer in enumerate(model.uni_mamba_layers):
            x = layer(x)
            analyze_tensor(f"After Uni-Mamba {i+1}/{len(model.uni_mamba_layers)}", x)
        
        # RMS-ATT
        pad_mask = model._make_key_padding_mask(mel)
        x = model.rms_att(x, key_padding_mask=pad_mask)
        analyze_tensor("After RMS-ATT", x)
        
        # Bi-Mamba MoE layers
        for i, layer in enumerate(model.bi_mamba_moe_layers):
            x = layer(x)
            analyze_tensor(f"After Bi-Mamba MoE {i+1}/{len(model.bi_mamba_moe_layers)}", x)
        
        # Phonological layer
        x, _ = model.phonological(x)
        analyze_tensor("After Phonological Layer", x)
        
        # QFormer (if applicable)
        if hasattr(model, 'qformer'):
            q_out = model.qformer(x, memory_key_padding_mask=pad_mask)
            analyze_tensor("After QFormer", q_out)
    
    return x


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT VERIFICATION
# ════════════════════════════════════════════════════════════════════════════

def verify_checkpoint(ckpt_path: str, model: AudioEncoder):
    """Verify that checkpoint weights loaded correctly."""
    print(f"\\n{'='*60}")
    print(f"CHECKPOINT VERIFICATION")
    print(f"{'='*60}")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return False
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Check checkpoint structure
    print(f"\\nCheckpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'flat tensor'}")
    
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state = ckpt['model']
            print("Found 'model' key in checkpoint")
        elif 'audio_encoder' in ckpt:
            state = ckpt['audio_encoder']
            print("Found 'audio_encoder' key in checkpoint")
        else:
            state = ckpt
            print("Using checkpoint as state_dict directly")
    else:
        state = ckpt
    
    # Count parameters
    ckpt_params = sum(v.numel() for v in state.values())
    model_params = sum(p.numel() for p in model.parameters())
    
    print(f"\\nCheckpoint params: {ckpt_params:,}")
    print(f"Model params: {model_params:,}")
    print(f"Match: {'✅' if ckpt_params == model_params else '❌'}")
    
    # Check for missing/unexpected keys
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    
    if missing:
        print(f"\\n⚠️  Missing keys ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")
    
    if unexpected:
        print(f"\\n⚠️  Unexpected keys ({len(unexpected)}):")
        for k in sorted(unexpected)[:10]:
            print(f"    {k}")
        if len(unexpected) > 10:
            print(f"    ... and {len(unexpected)-10} more")
    
    # Check weight statistics
    print(f"\\nCheckpoint weight statistics:")
    total_mean = []
    total_std = []
    for k, v in state.items():
        if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            total_mean.append(v.float().mean().item())
            total_std.append(v.float().std().item())
    
    print(f"  Mean of all weights: {np.mean(total_mean):.6f}")
    print(f"  Std of all weights: {np.mean(total_std):.6f}")
    
    # Check if weights look random (std ≈ 0.02 for xavier init)
    if np.mean(total_std) < 0.001:
        print(f"  ⚠️  Weights look very small — might be zeroed or corrupted")
    elif np.mean(total_std) > 1.0:
        print(f"  ⚠️  Weights look very large — might be untrained or exploded")
    else:
        print(f"  ✅ Weight magnitudes look reasonable")
    
    return len(missing) == 0 and len(unexpected) == 0


# ════════════════════════════════════════════════════════════════════════════
# LIBRISPEECH BATCH TEST
# ════════════════════════════════════════════════════════════════════════════

def test_librispeech_batch(model: AudioEncoder, data_root: str, cfg: TestConfig, 
                           device: torch.device, num_samples: int = 10):
    """Test encoder on multiple LibriSpeech samples."""
    print(f"\\n{'='*60}")
    print(f"LIBRISPEECH BATCH TEST ({num_samples} samples)")
    print(f"{'='*60}")
    
    wav_files = sorted(glob.glob(os.path.join(data_root, "**/*.wav"), recursive=True))
    if not wav_files:
        print(f"❌ No WAV files found in {data_root}")
        return
    
    wav_files = wav_files[:num_samples]
    print(f"Found {len(wav_files)} samples")
    
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for i, wav_path in enumerate(wav_files):
            mel = load_wav_to_mel(wav_path, cfg, device)
            feats = model.encode_features(mel)
            features_list.append(feats)
            
            if i < 3:  # Print details for first 3
                print(f"\\nSample {i+1}: {os.path.basename(wav_path)}")
                analyze_tensor(f"  Features", feats)
    
    # Compare all pairs
    print(f"\\n{'='*60}")
    print(f"PAIRWISE COMPARISONS")
    print(f"{'='*60}")
    
    similarities = []
    for i in range(len(features_list)):
        for j in range(i+1, len(features_list)):
            sim = F.cosine_similarity(features_list[i].mean(1), features_list[j].mean(1)).item()
            similarities.append(sim)
            if i < 3 and j < 4:  # Print first few
                print(f"Sample {i+1} vs {j+1}: cos_sim = {sim:.6f}")
    
    print(f"\\nAll pairwise similarities:")
    print(f"  mean: {np.mean(similarities):.6f}")
    print(f"  std:  {np.std(similarities):.6f}")
    print(f"  min:  {np.min(similarities):.6f}")
    print(f"  max:  {np.max(similarities):.6f}")
    
    # Compare with silence
    zero_mel = create_zero_mel(cfg, device)
    zero_feats = model.encode_features(zero_mel)
    
    zero_sims = []
    for i, feats in enumerate(features_list):
        sim = F.cosine_similarity(feats.mean(1), zero_feats.mean(1)).item()
        zero_sims.append(sim)
    
    print(f"\\nSimilarity to silence:")
    print(f"  mean: {np.mean(zero_sims):.6f}")
    print(f"  std:  {np.std(zero_sims):.6f}")
    print(f"  min:  {np.min(zero_sims):.6f}")
    print(f"  max:  {np.max(zero_sims):.6f}")
    
    if np.mean(zero_sims) > 0.9:
        print(f"\\n🔴 CRITICAL: Real audio is nearly identical to silence!")
        print(f"   The encoder is NOT learning meaningful features.")
    elif np.mean(zero_sims) > 0.5:
        print(f"\\n🟡 WARNING: Real audio is somewhat similar to silence")
    else:
        print(f"\\n🟢 GOOD: Real audio is distinct from silence")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Audio Encoder Diagnostic")
    parser.add_argument("--wav", type=str, help="Path to a WAV file to test")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to encoder checkpoint")
    parser.add_argument("--librispeech", type=str, help="Path to LibriSpeech test-clean for batch test")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples for batch test")
    parser.add_argument("--compare-silence", action="store_true", help="Compare real audio vs silence")
    parser.add_argument("--layer-diagnostic", action="store_true", help="Run layer-by-layer diagnostic")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--llm-dim", type=int, default=512, help="LLM dimension")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build config
    cfg = TestConfig()
    cfg.d_model = args.d_model
    cfg.llm_dim = args.llm_dim
    
    # Build model
    print(f"\\nBuilding AudioEncoder (d_model={cfg.d_model}, llm_dim={cfg.llm_dim})...")
    encoder_cfg = AudioEncoderConfig(
        d_model=cfg.d_model,
        dstate=cfg.dstate,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        num_uni_mamba=cfg.num_uni_mamba,
        num_bi_mamba_moe=cfg.num_bi_mamba_moe,
        num_experts=cfg.num_experts,
        attn_heads=cfg.attn_heads,
        window_size=cfg.window_size,
        cgmlp_expansion=cfg.cgmlp_expansion,
        num_queries=cfg.num_queries,
        num_qformer_layers=cfg.num_qformer_layers,
        llm_dim=cfg.llm_dim,
        training_heads=cfg.training_heads,
        dropout=cfg.dropout,
    )
    
    model = AudioEncoder(encoder_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.1f}M")
    
    # Load checkpoint
    print(f"\\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state = ckpt['model']
        elif 'audio_encoder' in ckpt:
            state = ckpt['audio_encoder']
        else:
            state = ckpt
    else:
        state = ckpt
    
    # Strip qformer if present (since we're testing encode_features)
    state = {k: v for k, v in state.items() if not k.startswith("qformer")}
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
        for k in sorted(missing)[:5]:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for k in sorted(unexpected)[:5]:
            print(f"  {k}")
    
    model.eval()
    
    # Verify checkpoint
    verify_checkpoint(args.ckpt, model)
    
    # Single file test
    if args.wav:
        if not os.path.exists(args.wav):
            print(f"❌ WAV file not found: {args.wav}")
            return
        
        print(f"\\n{'='*60}")
        print(f"SINGLE FILE TEST: {args.wav}")
        print(f"{'='*60}")
        
        mel = load_wav_to_mel(args.wav, cfg, device)
        print(f"\\nMel spectrogram:")
        analyze_tensor("Mel", mel, show_hist=True)
        
        # Layer-by-layer
        if args.layer_diagnostic:
            layer_by_layer_diagnostic(model, mel, "real")
        
        # Encode features
        with torch.no_grad():
            features = model.encode_features(mel)
        
        print(f"\\nFinal encoded features:")
        analyze_tensor("Features", features, show_hist=True)
        
        # Compare with silence
        if args.compare_silence:
            zero_mel = create_zero_mel(cfg, device)
            with torch.no_grad():
                zero_features = model.encode_features(zero_mel)
            
            print(f"\\nSilence features:")
            analyze_tensor("Zero Features", zero_features)
            
            compare_features(features, zero_features, "Real Audio", "Silence")
            
            # Also compare with noise
            noise_mel = create_noise_mel(cfg, device, scale=mel.std().item())
            with torch.no_grad():
                noise_features = model.encode_features(noise_mel)
            
            print(f"\\nNoise features:")
            analyze_tensor("Noise Features", noise_features)
            
            compare_features(features, noise_features, "Real Audio", "Random Noise")
            compare_features(zero_features, noise_features, "Silence", "Random Noise")
    
    # Batch test on LibriSpeech
    if args.librispeech:
        test_librispeech_batch(model, args.librispeech, cfg, device, args.num_samples)
    
    print(f"\\n{'='*60}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
