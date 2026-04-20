import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from audio_encoder import AudioEncoder, AudioEncoderConfig, aggregate_losses
from librispeechMFADataset import LibriSpeechMFADataset, collate_fn
from dataclasses import dataclass


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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data — point these at your LibriSpeech root and MFA output root
    train_set = LibriSpeechMFADataset(
        librispeech_root="../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100",
        mfa_output_root="/home/ableflyer/librispeech_aligned",   # wherever mfa align wrote TextGrids
        max_frames=1024,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
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

    model.train()

    for epoch in range(epochs):
        total_loss_epoch = 0

        for i, (mel, labels) in enumerate(train_loader):
            mel    = mel.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, losses = model(mel, labels=labels, use_extractor=False)

                loss = aggregate_losses(losses, weights={
                    'ctc':         1.0,
                    'voicing':     0.5,
                    'manner':      0.5,
                    'place':       0.5,
                    'correctness': 0.1,
                })
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # add this
                scaler.update()
                optimizer.zero_grad()

                total_loss_epoch += loss.item() * accumulation_steps
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item()*accumulation_steps:.4f}")

        scheduler.step()
        print(f"--- Epoch {epoch} Complete | Avg Loss: {total_loss_epoch / len(train_loader):.4f} ---")

        torch.save(model.state_dict(), f"../Model_files/audio_encoder_epoch_{epoch}.pt")


if __name__ == "__main__":
    train()