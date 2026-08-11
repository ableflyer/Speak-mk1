import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from audio_encoder import AudioEncoder, AudioEncoderConfig, aggregate_losses
from dataclasses import dataclass

from ultrasuite_manifest import load_manifest, speaker_independent_split
from ultrasuite_dataset import UltraSuiteDataset, collate_fn as ultrasuite_collate_fn
from ctc_decode import greedy_ctc_decode
from metrics import corpus_per, frame_accuracy
from sampa_phone_table import PHONE2IDX, PAD_TOKEN


# --- Config for RTX 4060 Laptop ---
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
    training_heads: bool = True


def reinit_qformer(audio_encoder):
    for name, module in audio_encoder.qformer.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    # Reinit learned queries specifically
    if hasattr(audio_encoder.qformer, 'queries'):
        nn.init.trunc_normal_(audio_encoder.qformer.queries, std=0.02)
    print("Q-Former reinitialised.")


def build_labels_dict(batch: dict, device: torch.device) -> dict:
    """
    Adapts collate_fn's flat batch dict to whatever key names AudioEncoder
    expects in `labels`. I don't have audio_encoder.py's source, so these
    key names are an ASSUMPTION mirroring the aggregate_losses weight keys
    ('ctc', 'voicing', 'manner', 'place') from your own script -- if your
    model's forward() expects different key names internally, this is the
    one place to fix it.

    'correctness' is deliberately NOT included: your loader doesn't
    produce correctness labels (that axis isn't built yet -- needs the
    reference/target phone tier + lexicon, which you said not to worry
    about right now). Passing weight=0 in aggregate_losses is not enough
    if the model still expects a 'correctness' key in labels and errors
    on its absence -- if that happens, this is the spot to add a dummy
    all-ignore tensor rather than real supervision.
    """
    return {
        "ctc": batch["phone_seq"].to(device),
        "ctc_lengths": batch["phone_seq_lengths"].to(device),
        "input_lengths": batch["frame_lengths"].to(device),
        "voicing": batch["voicing"].to(device),
        "manner": batch["manner"].to(device),
        "place": batch["place"].to(device),
    }


@torch.no_grad()
def evaluate(model, val_loader, device) -> dict:
    """
    Runs the axes we discussed SEPARATELY:
      - PER (phone identity / "tokenization") via greedy CTC decode
      - frame-level accuracy for voicing / manner / place (secondary,
        alignment-sensitive -- reported per-feature, never blended together
        or with PER)
    """
    model.eval()
    all_hyps, all_refs = [], []
    voicing_correct = voicing_total = 0
    manner_correct = manner_total = 0
    place_correct = place_total = 0

    for batch in val_loader:
        mel = batch["mel"].to(device)
        labels = build_labels_dict(batch, device)

        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(mel, labels=labels)

        # ASSUMPTION: model returns a dict of logits with these keys.
        # Adjust to match audio_encoder.py's actual output structure.
        ctc_logits = outputs["ctc_logits"]          # (B, T, V)
        voicing_logits = outputs["voicing_logits"]  # (B, T, C_voicing)
        manner_logits = outputs["manner_logits"]
        place_logits = outputs["place_logits"]

        hyps = greedy_ctc_decode(ctc_logits, labels["input_lengths"], blank_id=PHONE2IDX[PAD_TOKEN])
        refs = [batch["phone_seq"][i, :batch["phone_seq_lengths"][i]].tolist()
                for i in range(len(hyps))]
        all_hyps.extend(hyps)
        all_refs.extend(refs)

        for pred_logits, ref, correct_total in (
            (voicing_logits, labels["voicing"], "voicing"),
            (manner_logits, labels["manner"], "manner"),
            (place_logits, labels["place"], "place"),
        ):
            pred_ids = pred_logits.argmax(dim=-1)
            for b in range(pred_ids.shape[0]):
                stats = frame_accuracy(
                    pred_ids[b].tolist(), ref[b].tolist(), ignore_id=-100,
                )
                if correct_total == "voicing":
                    voicing_correct += stats["correct"]; voicing_total += stats["total"]
                elif correct_total == "manner":
                    manner_correct += stats["correct"]; manner_total += stats["total"]
                else:
                    place_correct += stats["correct"]; place_total += stats["total"]

    per_result = corpus_per(all_hyps, all_refs)
    model.train()
    return {
        "per": per_result["per"],
        "per_breakdown": {k: v for k, v in per_result.items() if k != "per_utterance"},
        "voicing_acc": voicing_correct / voicing_total if voicing_total else 0.0,
        "manner_acc": manner_correct / manner_total if manner_total else 0.0,
        "place_acc": place_correct / place_total if place_total else 0.0,
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data -- UltraSuite UPX + UXSSD, filtered on phone TextGrid presence
    #    (not the manifest's `usable` flag alone -- see ultrasuite_manifest.py)
    #    and split speaker-independently so no child leaks across splits.
    rows = load_manifest(
        "manifest.csv",  # TODO: point at your real path
        corpora=("upx", "uxssd"),
    )
    splits = speaker_independent_split(rows, val_frac=0.1, test_frac=0.1, seed=42)

    train_set = UltraSuiteDataset(splits["train"], max_frames=1024)
    val_set = UltraSuiteDataset(splits["val"], max_frames=1024)

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        collate_fn=ultrasuite_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        collate_fn=ultrasuite_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Model
    cfg = SmallConfig()
    model = AudioEncoder(cfg).to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Parameters: {params:.2f}M")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = GradScaler()

    # 4. Training loop
    epochs = 20
    accumulation_steps = 4  # effective batch = 8 * 4 = 32
    best_val_per = float("inf")

    model.train()

    for epoch in range(epochs):
        total_loss_epoch = 0.0

        for i, batch in enumerate(train_loader):
            mel = batch["mel"].to(device)
            labels = build_labels_dict(batch, device)
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(mel, labels=labels)
                losses = outputs["losses"]

            # 'correctness' intentionally dropped -- no ground truth for it
            # yet (see build_labels_dict docstring). Re-add once the
            # reference/target phone tier + lexicon work is done.
            loss = aggregate_losses(losses, weights={
                'ctc':     1.0,
                'voicing': 0.5,
                'manner':  0.5,
                'place':   0.5,
            })
            loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                # unscale BEFORE clipping, and clip BEFORE stepping --
                # clipping after scaler.step() (as in the original script)
                # has no effect, since the step already happened with
                # unclipped gradients.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss_epoch += loss.item() * accumulation_steps
            print(f"Epoch {epoch} | Step {i} | Loss: {loss.item()*accumulation_steps:.4f}")

        scheduler.step()
        avg_train_loss = total_loss_epoch / len(train_loader)
        print(f"--- Epoch {epoch} Complete | Avg Train Loss: {avg_train_loss:.4f} ---")

        # --- Validation: PER for phone identity, frame-acc for the rest,
        #     reported separately, never combined into one number.
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"--- Epoch {epoch} Val | PER: {val_metrics['per']:.4f} "
            f"(S={val_metrics['per_breakdown']['substitutions']} "
            f"I={val_metrics['per_breakdown']['insertions']} "
            f"D={val_metrics['per_breakdown']['deletions']} "
            f"/ {val_metrics['per_breakdown']['total_ref_phones']} ref phones) "
            f"| voicing_acc: {val_metrics['voicing_acc']:.4f} "
            f"| manner_acc: {val_metrics['manner_acc']:.4f} "
            f"| place_acc: {val_metrics['place_acc']:.4f} ---"
        )

        torch.save(model.state_dict(), f"../Model_files/Audio_encoder_v1.5/audio_encoder_epoch_{epoch}.pt")

        if val_metrics["per"] < best_val_per:
            best_val_per = val_metrics["per"]
            torch.save(model.state_dict(), f"../Model_files/Audio_encoder_v1.5/audio_encoder_best_per.pt")
            print(f"--- New best val PER: {best_val_per:.4f} -- saved checkpoint ---")


if __name__ == "__main__":
    train()