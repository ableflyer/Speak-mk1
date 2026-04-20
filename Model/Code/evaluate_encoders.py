"""
evaluate_encoders.py
====================
Tests all 20 epoch checkpoints (epoch_0 to epoch_19) and ranks them
by phonological head accuracy, loss, and token quality.

Usage:
    python3 evaluate_encoders.py

Output:
    - Per-epoch results printed to console
    - Full ranking saved to encoder_evaluation_results.txt
    - Loss curve saved to loss_curve.png
"""

import os
import glob
import torch
import torch.nn.functional as F
import librosa
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import DataLoader

from audio_encoder import AudioEncoder, AudioEncoderConfig
from librispeechMFADataset import LibriSpeechMFADataset, collate_fn


# ── Config — must match training config exactly ──────────────────────────────

@dataclass
class SmallConfig(AudioEncoderConfig):
    d_model: int = 256
    dstate: int = 16
    num_uni_mamba: int = 3
    num_bi_mamba_moe: int = 2
    num_experts: int = 2
    attn_heads: int = 8
    num_queries: int = 32
    llm_dim: int = 2048
    dropout: float = 0.0        # always 0 during eval
    training_heads: bool = True


# ── Paths — update these to match your setup ─────────────────────────────────

CHECKPOINT_DIR   = "../Model_files"
LIBRISPEECH_TEST = "../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/test-clean"
MFA_TEST_ROOT    = "/home/ableflyer/librispeech_aligned_test"
RESULTS_FILE     = "encoder_evaluation_results.txt"
NUM_EPOCHS       = 20
MAX_BATCHES      = 50    # limit eval batches per checkpoint for speed
                         # set to None to evaluate on full test set


# ── Helper: load a mel from a wav file ───────────────────────────────────────

def wav_to_mel(wav_path: str, max_frames: int = 1024) -> torch.Tensor:
    y, sr = librosa.load(wav_path, sr=16_000)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=80, hop_length=160, n_fft=400
    )
    mel = librosa.power_to_db(mel).T                        # (T, 80)
    mel = torch.from_numpy(mel).float()
    if mel.shape[0] < max_frames:
        mel = F.pad(mel, (0, 0, 0, max_frames - mel.shape[0]))
    else:
        mel = mel[:max_frames]
    return mel.unsqueeze(0)                                  # (1, T, 80)


# ── Helper: run encoder forward manually to get head logits ──────────────────

@torch.no_grad()
def get_head_predictions(model: AudioEncoder, mel: torch.Tensor, device):
    """Run the full encoder pipeline and return per-head logits."""
    mel = mel.to(device)
    x = model.stem(mel)
    for layer in model.uni_mamba_layers:
        x = layer(x)
    pad_mask = model._make_key_padding_mask(mel)
    x = model.rms_att(x, key_padding_mask=pad_mask)
    for layer in model.bi_mamba_moe_layers:
        x = layer(x)
    x, _ = model.phonological(x)
    return x, model.phonological.heads

def infer_config_from_checkpoint(state_dict: dict) -> AudioEncoderConfig:
    from dataclasses import dataclass

    @dataclass
    class InferredConfig(AudioEncoderConfig):
        pass

    cfg = InferredConfig()

    # d_model
    cfg.d_model = state_dict["stem.0.weight"].shape[0]

    # dstate
    ssm_in_dim = state_dict["uni_mamba_layers.0.in_proj.weight"].shape[0]
    cfg.dstate  = (ssm_in_dim - cfg.d_model) // 2

    # num_uni_mamba
    cfg.num_uni_mamba = sum(
        1 for k in state_dict
        if k.startswith("uni_mamba_layers.") and k.endswith(".norm.weight")
    )

    # num_bi_mamba_moe
    cfg.num_bi_mamba_moe = sum(
        1 for k in state_dict
        if k.startswith("bi_mamba_moe_layers.") and k.endswith(".merge_norm.weight")
    )

    # num_experts — read directly from expert weight shape
    cfg.num_experts = state_dict[
        "bi_mamba_moe_layers.0.bi_mamba_moe.expert_W1"
    ].shape[0]

    # num_queries and llm_dim
    cfg.num_queries = state_dict["qformer.queries"].shape[1]
    cfg.llm_dim     = state_dict["qformer.out_proj.weight"].shape[0]

    # attn_heads — infer from q_proj output dim / d_model
    # q_proj is (d_model, d_model) so nheads = d_model // headdim
    # headdim is fixed at 64 in LocalWindowAttention default
    cfg.attn_heads = cfg.d_model // 64

    cfg.dropout        = 0.0
    cfg.training_heads = True

    return cfg


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_checkpoint(checkpoint_path: str, test_loader, device) -> dict:
    """
    Evaluate a single checkpoint.
    Returns a dict with accuracy per head, avg loss, and token stats.
    """
    state = torch.load(checkpoint_path, map_location=device)
    
    # Infer config from checkpoint instead of hardcoding
    cfg   = infer_config_from_checkpoint(state)
    model = AudioEncoder(cfg).to(device)
    model.load_state_dict(state)
    model.eval()

    print(f"  Inferred config: d_model={cfg.d_model}, "
          f"uni={cfg.num_uni_mamba}, bi={cfg.num_bi_mamba_moe}, "
          f"queries={cfg.num_queries}, llm_dim={cfg.llm_dim}")

    head_names   = ['voicing', 'manner', 'place', 'correctness']
    correct      = {k: 0 for k in head_names}
    total        = {k: 0 for k in head_names}
    total_loss   = 0.0
    num_batches  = 0
    token_norms  = []

    with torch.no_grad():
        for batch_idx, (mel, labels) in enumerate(test_loader):
            if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
                break

            mel    = mel.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            # Forward pass for loss
            try:
                llm_tokens, losses = model(mel, labels=labels, use_extractor=False)
            except Exception as e:
                print(f"  Forward pass error: {e}")
                continue

            if losses:
                batch_loss = sum(losses.values()).item()
                if not np.isnan(batch_loss):
                    total_loss += batch_loss
                    num_batches += 1

            # Token quality — collect L2 norms of output tokens
            token_norms.append(llm_tokens.norm(dim=-1).mean().item())

            # Per-head accuracy
            feat, heads = get_head_predictions(model, mel, device)
            for name in head_names:
                if name not in heads:
                    continue
                logits = heads[name](feat)              # (B, T, C)
                preds  = logits.argmax(dim=-1)          # (B, T)
                tgt    = labels[name]                   # (B, T)
                mask   = tgt != -100
                if mask.sum() == 0:
                    continue
                correct[name] += (preds[mask] == tgt[mask]).sum().item()
                total[name]   += mask.sum().item()

    # Compute accuracies
    accuracies = {}
    for name in head_names:
        if total[name] > 0:
            accuracies[name] = correct[name] / total[name] * 100
        else:
            accuracies[name] = 0.0

    avg_loss       = total_loss / max(num_batches, 1)
    avg_token_norm = float(np.mean(token_norms)) if token_norms else 0.0
    overall_acc    = np.mean(list(accuracies.values()))

    return {
        "accuracies":     accuracies,
        "avg_loss":       avg_loss,
        "overall_acc":    overall_acc,
        "avg_token_norm": avg_token_norm,
        "num_batches":    num_batches,
    }


# ── Similarity test between two wav files ────────────────────────────────────

@torch.no_grad()
def token_similarity_test(model: AudioEncoder, device) -> dict:
    """
    Quick qualitative test:
    Find two wav files in the test set and compare token cosine similarity.
    Similar utterances should score higher than random pairs.
    """
    # Find a few wav files to test with
    import random
    wav_files = glob.glob(
        os.path.join(LIBRISPEECH_TEST, "**/*.wav"), recursive=True
    )
    if len(wav_files) < 3:
        return {"similarity_same_speaker": None, "similarity_diff_speaker": None}

    # Group by speaker
    speaker_files = {}
    for f in wav_files[:200]:    # limit search
        spk = os.path.basename(f).split("-")[0]
        speaker_files.setdefault(spk, []).append(f)

    speakers = [s for s, fs in speaker_files.items() if len(fs) >= 2]
    if len(speakers) < 2:
        return {"similarity_same_speaker": None, "similarity_diff_speaker": None}

    model.eval()

    # Same speaker pair
    spk_a = speakers[0]
    mel_a1 = wav_to_mel(speaker_files[spk_a][0]).to(device)
    mel_a2 = wav_to_mel(speaker_files[spk_a][1]).to(device)
    tok_a1 = model.encode(mel_a1).flatten()
    tok_a2 = model.encode(mel_a2).flatten()
    sim_same = F.cosine_similarity(tok_a1, tok_a2, dim=0).item()

    # Different speaker pair
    spk_b = speakers[1]
    mel_b1 = wav_to_mel(speaker_files[spk_b][0]).to(device)
    tok_b1 = model.encode(mel_b1).flatten()
    sim_diff = F.cosine_similarity(tok_a1, tok_b1, dim=0).item()

    return {
        "similarity_same_speaker": sim_same,
        "similarity_diff_speaker": sim_diff,
        "similarity_gap": sim_same - sim_diff,   # higher = better separation
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Evaluating {NUM_EPOCHS} checkpoints...\n")

    # Build test loader
    print("Loading test dataset...")
    try:
        test_set = LibriSpeechMFADataset(
            librispeech_root=LIBRISPEECH_TEST,
            mfa_output_root=MFA_TEST_ROOT,
            max_frames=1024,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        use_test_loader = True
        print(f"Test set: {len(test_set)} utterances\n")
    except Exception as e:
        print(f"Warning: Could not load test set ({e})")
        print("Skipping accuracy/loss evaluation, running similarity tests only.\n")
        test_loader    = None
        use_test_loader = False

    results = {}

    for epoch in range(NUM_EPOCHS):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"audio_encoder_epoch_{epoch}.pt")

        if not os.path.exists(ckpt_path):
            print(f"Epoch {epoch:02d}: checkpoint not found, skipping.")
            continue

        print(f"Epoch {epoch:02d}: evaluating {ckpt_path} ...")

        if use_test_loader:
            result = evaluate_checkpoint(ckpt_path, test_loader, device)
        else:
            result = {
                "accuracies":     {k: 0.0 for k in ['voicing','manner','place','correctness']},
                "avg_loss":       float("nan"),
                "overall_acc":    0.0,
                "avg_token_norm": 0.0,
                "num_batches":    0,
            }

        if result is None:
            continue

        # Similarity test
        cfg   = SmallConfig()
        model = AudioEncoder(cfg).to(device)
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            sim   = token_similarity_test(model, device)
            result["similarity"] = sim
        except Exception:
            result["similarity"] = {}

        results[epoch] = result

        # Print summary for this epoch
        acc = result["accuracies"]
        print(f"  Loss          : {result['avg_loss']:.4f}")
        print(f"  Overall Acc   : {result['overall_acc']:.1f}%")
        print(f"  Voicing Acc   : {acc['voicing']:.1f}%")
        print(f"  Manner Acc    : {acc['manner']:.1f}%")
        print(f"  Place Acc     : {acc['place']:.1f}%")
        print(f"  Correctness   : {acc['correctness']:.1f}%")
        sim = result.get("similarity", {})
        if sim.get("similarity_same_speaker") is not None:
            print(f"  Same-spk sim  : {sim['similarity_same_speaker']:.3f}")
            print(f"  Diff-spk sim  : {sim['similarity_diff_speaker']:.3f}")
            print(f"  Sim gap       : {sim['similarity_gap']:.3f}")
        print()

    if not results:
        print("No checkpoints found. Check CHECKPOINT_DIR path.")
        return

    # ── Ranking ──────────────────────────────────────────────────────────────

    # Score = overall accuracy (primary) + similarity gap bonus (secondary)
    def score(r):
        acc_score = r["overall_acc"]
        sim_gap   = r.get("similarity", {}).get("similarity_gap", 0.0) or 0.0
        loss_pen  = r["avg_loss"] if not np.isnan(r["avg_loss"]) else 999
        return acc_score + sim_gap * 10 - loss_pen * 0.1

    ranked = sorted(results.items(), key=lambda x: score(x[1]), reverse=True)

    print("\n" + "="*60)
    print("FINAL RANKING (best to worst)")
    print("="*60)
    for rank, (epoch, r) in enumerate(ranked, 1):
        print(f"  #{rank:2d}  Epoch {epoch:02d}  |  "
              f"Overall Acc: {r['overall_acc']:.1f}%  |  "
              f"Loss: {r['avg_loss']:.4f}  |  "
              f"Score: {score(r):.3f}")

    best_epoch, best_result = ranked[0]
    print(f"\n✓ BEST CHECKPOINT: audio_encoder_epoch_{best_epoch}.pt")
    print(f"  Overall accuracy : {best_result['overall_acc']:.1f}%")
    print(f"  Voicing          : {best_result['accuracies']['voicing']:.1f}%")
    print(f"  Manner           : {best_result['accuracies']['manner']:.1f}%")
    print(f"  Place            : {best_result['accuracies']['place']:.1f}%")
    print(f"  Correctness      : {best_result['accuracies']['correctness']:.1f}%")

    # ── Save results to file ──────────────────────────────────────────────────

    with open(RESULTS_FILE, "w") as f:
        f.write("ENCODER EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write("RANKING (best to worst)\n")
        f.write("-"*60 + "\n")
        for rank, (epoch, r) in enumerate(ranked, 1):
            f.write(f"#{rank:2d}  Epoch {epoch:02d}  |  "
                    f"Overall Acc: {r['overall_acc']:.1f}%  |  "
                    f"Loss: {r['avg_loss']:.4f}\n")
            for name, acc in r["accuracies"].items():
                f.write(f"       {name:>12s}: {acc:.1f}%\n")
            sim = r.get("similarity", {})
            if sim.get("similarity_gap") is not None:
                f.write(f"       sim_gap     : {sim['similarity_gap']:.3f}\n")
            f.write("\n")

        f.write(f"\nBEST CHECKPOINT: audio_encoder_epoch_{best_epoch}.pt\n")

    print(f"\nFull results saved to {RESULTS_FILE}")

    # ── Loss curve plot ───────────────────────────────────────────────────────

    epochs_list = sorted(results.keys())
    losses      = [results[e]["avg_loss"] for e in epochs_list]
    overall_acc = [results[e]["overall_acc"] for e in epochs_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_list, losses, marker='o', color='crimson', linewidth=2)
    ax1.set_title("Avg Loss per Epoch (test set)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(best_epoch, color='green', linestyle='--', label=f"Best (epoch {best_epoch})")
    ax1.legend()

    ax2.plot(epochs_list, overall_acc, marker='o', color='steelblue', linewidth=2)
    ax2.set_title("Overall Head Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(best_epoch, color='green', linestyle='--', label=f"Best (epoch {best_epoch})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Loss curve saved to loss_curve.png")


if __name__ == "__main__":
    main()