"""
audio_tester.py
================
Quick qualitative + quantitative tester for SpeakMK1's AudioEncoder against
real UltraSuite utterances.

For each sampled utterance it:
  1. Runs the model (no labels -> pure inference, logits only)
  2. Greedy-decodes the CTC head into a phone sequence
  3. Prints predicted vs. reference phones + per-utterance PER
  4. Prints frame-level voicing/manner/place accuracy for that utterance
  5. (optional) saves a mel spectrogram PNG and the raw audio as a .wav
     so you can actually listen to what was fed in

USAGE
-----
    python audio_tester.py \
        --checkpoint ../Model_files/Audio_encoder_v1.5/audio_encoder_best_per.pt \
        --manifest manifest.csv \
        --corpora upx uxssd \
        --split val \
        --num-samples 5 \
        --export-audio \
        --output-dir ./tester_out

NOTE ON ASSUMPTIONS
--------------------
Verified against the real ultrasuite_dataset.py: ManifestRow is an
attribute-access object (row.wav_path, row.utt_id, etc.), collate_fn's
frame_lengths correctly uses item["valid_frames"], and mel is (T, n_mels)
matching AudioEncoder's stem input. Still unverified: ultrasuite_manifest.py
(load_manifest / speaker_independent_split signatures), ctc_decode.py, and
metrics.py -- these are only inferred from how train.py calls them. Any
remaining "# ASSUMPTION" markers below refer to those three files.
"""

import argparse
import os
import random
from dataclasses import dataclass

import torch
import numpy as np

from audio_encoder import AudioEncoder, AudioEncoderConfig
from ultrasuite_manifest import load_manifest, speaker_independent_split
from ultrasuite_dataset import UltraSuiteDataset, collate_fn as ultrasuite_collate_fn
from ctc_decode import greedy_ctc_decode
from sampa_phone_table import PHONE2IDX, PAD_TOKEN

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


# ─────────────────────────────────────────────────────────────────────────
# Must match the config the checkpoint was actually trained with, or
# load_state_dict will fail on shape mismatch. Mirrors SmallConfig in
# train.py -- keep these two in sync by hand.
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class SmallConfig(AudioEncoderConfig):
    d_model: int = 512
    dstate: int = 32
    num_uni_mamba: int = 6
    num_bi_mamba_moe: int = 4
    num_experts: int = 2
    attn_heads: int = 8
    num_queries: int = 64
    dropout: float = 0.15
    training_heads: bool = True  # must stay True -- inference needs the logits


IDX2PHONE = {v: k for k, v in PHONE2IDX.items()}


def ids_to_phones(ids):
    return [IDX2PHONE.get(i, f"<UNK:{i}>") for i in ids]


def edit_distance(ref, hyp):
    """Plain Levenshtein distance over phone-id sequences (word-error-rate style)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def per_utterance_per(ref_ids, hyp_ids):
    if len(ref_ids) == 0:
        return 0.0 if len(hyp_ids) == 0 else 1.0
    return edit_distance(ref_ids, hyp_ids) / len(ref_ids)


def frame_acc(pred_ids, ref_ids, ignore_id=-100):
    correct = total = 0
    for p, r in zip(pred_ids, ref_ids):
        if r == ignore_id:
            continue
        total += 1
        if p == r:
            correct += 1
    return correct, total


def load_model(checkpoint_path: str, device: torch.device) -> AudioEncoder:
    cfg = SmallConfig()
    model = AudioEncoder(cfg).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # ASSUMPTION: checkpoint is a raw state_dict, matching train.py's
    # torch.save(model.state_dict(), ...). If you switch to saving a dict
    # with {"model": ..., "optimizer": ..., "epoch": ...}, unwrap it here.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def gather_samples(manifest_path, corpora, split, num_samples, seed):
    # ASSUMPTION: same filtering/split logic as train.py -- filtered on
    # phone_textgrid_path presence inside load_manifest, not the `usable`
    # flag (per your own memory note on why `usable` is unreliable).
    rows = load_manifest(manifest_path, corpora=corpora)
    splits = speaker_independent_split(rows, val_frac=0.1, test_frac=0.1, seed=seed)
    pool = splits[split]
    rng = random.Random(seed)
    n = min(num_samples, len(pool))
    return rng.sample(pool, n)


def save_mel_plot(mel: np.ndarray, title: str, out_path: str):
    if not HAS_MPL:
        print(f"  [skip plot: matplotlib not installed] {out_path}")
        return
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.T, aspect="auto", origin="lower", cmap="magma")
    plt.title(title, fontsize=9)
    plt.xlabel("frame")
    plt.ylabel("mel bin")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def export_audio(row, out_path: str, sr: int, audio_path_col: str):
    """
    Writes the utterance's raw audio to disk so you can actually listen.

    row is a ManifestRow (attribute access, e.g. row.wav_path) -- confirmed
    against ultrasuite_dataset.py's UltraSuiteDataset._load_audio(row.wav_path).
    Default audio_path_col="wav_path" matches the real field name.
    """
    if not HAS_SF:
        print("  [skip audio export: soundfile not installed]")
        return
    wav_path = getattr(row, audio_path_col, None)
    if not wav_path or not os.path.exists(wav_path):
        print(f"  [skip audio export: no valid '{audio_path_col}' on this manifest row]")
        return
    import librosa
    y, _ = librosa.load(wav_path, sr=sr)
    sf.write(out_path, y, sr)


def main():
    ap = argparse.ArgumentParser(description="Test AudioEncoder on real UltraSuite samples")
    ap.add_argument("--checkpoint", required=True, help="Path to a saved AudioEncoder state_dict (.pt)")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv")
    ap.add_argument("--corpora", nargs="+", default=["upx", "uxssd"])
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="./tester_out")
    ap.add_argument("--export-audio", action="store_true", help="Also dump raw audio + mel plot per sample")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--audio-path-col", default="wav_path",
                     help="ManifestRow attribute holding the raw audio path (confirmed against ultrasuite_dataset.py)")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    print(f"Sampling {args.num_samples} '{args.split}' utterances from {args.manifest} ...")
    sample_rows = gather_samples(args.manifest, args.corpora, args.split, args.num_samples, args.seed)

    dataset = UltraSuiteDataset(sample_rows, max_frames=1024)
    loader_batch = ultrasuite_collate_fn([dataset[i] for i in range(len(dataset))])

    mel = loader_batch["mel"].to(device)
    frame_lengths = loader_batch["frame_lengths"].to(device)
    phone_seq = loader_batch["phone_seq"]
    phone_seq_lengths = loader_batch["phone_seq_lengths"]
    voicing_ref = loader_batch["voicing"]
    manner_ref = loader_batch["manner"]
    place_ref = loader_batch["place"]

    with torch.no_grad():
        outputs = model(mel, labels=None, attention_mask=None)

    ctc_logits = outputs["ctc_logits"]          # (B, T, V)
    voicing_logits = outputs["voicing_logits"]
    manner_logits = outputs["manner_logits"]
    place_logits = outputs["place_logits"]

    hyps = greedy_ctc_decode(ctc_logits, frame_lengths, blank_id=PHONE2IDX[PAD_TOKEN])

    voicing_pred = voicing_logits.argmax(dim=-1).cpu()
    manner_pred = manner_logits.argmax(dim=-1).cpu()
    place_pred = place_logits.argmax(dim=-1).cpu()

    per_scores = []
    print("\n" + "=" * 78)
    for i, row in enumerate(sample_rows):
        ref_ids = phone_seq[i, :phone_seq_lengths[i]].tolist()
        hyp_ids = hyps[i]

        per = per_utterance_per(ref_ids, hyp_ids)
        per_scores.append(per)

        v_c, v_t = frame_acc(voicing_pred[i].tolist(), voicing_ref[i].tolist())
        m_c, m_t = frame_acc(manner_pred[i].tolist(), manner_ref[i].tolist())
        p_c, p_t = frame_acc(place_pred[i].tolist(), place_ref[i].tolist())

        utt_id = getattr(row, "utt_id", f"sample_{i}")
        print(f"[{i}] utt_id={utt_id}  PER={per:.3f}")
        print(f"     ref phones : {' '.join(ids_to_phones(ref_ids))}")
        print(f"     hyp phones : {' '.join(ids_to_phones(hyp_ids))}")
        print(f"     voicing acc: {v_c}/{v_t} = {v_c/v_t if v_t else 0:.3f}   "
              f"manner acc: {m_c}/{m_t} = {m_c/m_t if m_t else 0:.3f}   "
              f"place acc: {p_c}/{p_t} = {p_c/p_t if p_t else 0:.3f}")

        if args.export_audio:
            mel_np = mel[i, :frame_lengths[i]].cpu().numpy()
            plot_path = os.path.join(args.output_dir, f"{utt_id}_mel.png")
            wav_path = os.path.join(args.output_dir, f"{utt_id}.wav")
            save_mel_plot(mel_np, f"{utt_id} | PER={per:.3f}", plot_path)
            export_audio(row, wav_path, args.sr, args.audio_path_col)
            print(f"     saved: {plot_path}" + (f", {wav_path}" if HAS_SF else ""))

        print("-" * 78)

    avg_per = sum(per_scores) / len(per_scores) if per_scores else 0.0
    print(f"\nSampled-set average PER: {avg_per:.4f} over {len(per_scores)} utterances")
    print("=" * 78)


if __name__ == "__main__":
    main()