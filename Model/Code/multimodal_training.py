"""
train_multimodal.py
===================
Trains SpeakMK1Multimodal end-to-end from mel spectrograms.
No precomputed cache needed — audio encoder is trained jointly.

Two-phase training:
    Phase 1: freeze LLM backbone, train encoder + proj + cross_attn
    Phase 2: unfreeze all, joint fine-tune at lower lr

Data expected: MFA-converted LibriSpeech with per-utterance .wav + .txt
    <root>/<speaker>/<chapter>/
        19-198-0000.wav
        19-198-0000.txt   ← single-line transcript (no ID prefix)
        ...
"""

import os
import glob
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer
import librosa
import torchaudio

from audio_trainer import SmallConfig
from speak_mk1_llm import SpeakMK1LLMConfig
from speak_mk1_combined import SpeakMK1Multimodal


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

class MultimodalTrainConfig:

    # ── Paths ─────────────────────────────────────────────────────────────
    data_root       = "../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100"
    audio_ckpt      = "../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt"
    llm_ckpt        = "../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt"
    save_dir        = "../Model_files/multimodal_training"

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer_name  = "EleutherAI/gpt-neox-20b"
    # GPT-NeoX eos=0, so we pick a safe unused ID for padding
    # to avoid zero-tensor ambiguity. 1 is never a real token.
    pad_token_id    = 1
    max_text_len    = 256     # keep short to fit VRAM with audio tokens

    # ── Audio ─────────────────────────────────────────────────────────────
    sample_rate     = 16_000
    n_mels          = 80
    n_fft           = 400
    hop_length      = 160
    max_frames      = 512

    # ── Model dims — must match checkpoints ───────────────────────────────
    audio_d_model   = 512
    llm_d_model     = 512
    vocab_size      = 50283   # always override, default is wrong
    num_experts     = 4       # always override, default is wrong

    # ── Phase 1 training (backbone frozen) ────────────────────────────────
    phase1_epochs       = 2
    phase1_lr           = 3e-4
    phase1_warmup_steps = 200

    # ── Phase 2 training (full joint fine-tune) ────────────────────────────
    phase2_epochs       = 1
    phase2_lr           = 3e-5   # much lower — backbone is sensitive
    phase2_warmup_steps = 50

    # ── Common ────────────────────────────────────────────────────────────
    batch_size         = 2      # RTX 4060 8GB — increase if VRAM allows
    accumulation_steps = 16     # effective batch = 32
    weight_decay       = 0.01
    max_grad_norm      = 1.0
    log_every          = 50
    save_every_epoch   = True
    num_workers        = 2
    seed               = 42


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class LibriSpeechMultimodalDataset(Dataset):
    """
    Loads (mel, token_ids) pairs directly from MFA-converted LibriSpeech.

    Directory structure expected:
        <root>/<speaker>/<chapter>/<file_id>.wav
        <root>/<speaker>/<chapter>/<file_id>.txt  ← one line, no ID prefix

    Returns
    -------
    mel       : (max_frames, n_mels)  — log-mel spectrogram, padded/truncated
    token_ids : (max_text_len,)       — GPT-NeoX token IDs, padded with pad_id
    """

    def __init__(self, cfg: MultimodalTrainConfig):
        self.cfg = cfg

        # ── Tokeniser ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        # Use a safe non-zero pad ID to distinguish from real tokens
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.pad_id = cfg.pad_token_id

        # ── Mel transform ──────────────────────────────────────────────────
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = cfg.sample_rate,
            n_fft       = cfg.n_fft,
            hop_length  = cfg.hop_length,
            n_mels      = cfg.n_mels,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # ── Index all wav/txt pairs ────────────────────────────────────────
        self.samples = self._build_index()
        print(f"LibriSpeechMultimodalDataset: {len(self.samples)} samples")
        if self.samples:
            print(f"  Example: {self.samples[0][0]}")

    def _build_index(self):
        """
        Walks the data root and pairs each .wav with its sibling .txt.
        Skips utterances with no transcript file.
        """
        wav_files = glob.glob(
            os.path.join(self.cfg.data_root, "**/*.wav"), recursive=True
        )
        samples = []
        for wav_path in sorted(wav_files):
            txt_path = wav_path.replace(".wav", ".txt")
            if not os.path.exists(txt_path):
                continue
            samples.append((wav_path, txt_path))
        return samples

    def _load_mel(self, wav_path: str) -> torch.Tensor:
        """
        Loads wav, computes log-mel spectrogram, pads/truncates to max_frames.

        Returns
        -------
        mel : (max_frames, n_mels)
        """
        y, _ = librosa.load(wav_path, sr=self.cfg.sample_rate)
        waveform = torch.from_numpy(y).unsqueeze(0)       # (1, samples)
        mel = self.mel_transform(waveform)                 # (1, n_mels, T)
        mel = self.db_transform(mel).squeeze(0).T          # (T, n_mels)

        T = mel.shape[0]
        if T >= self.cfg.max_frames:
            mel = mel[:self.cfg.max_frames]
        else:
            mel = F.pad(mel, (0, 0, 0, self.cfg.max_frames - T))
        return mel   # (max_frames, n_mels)

    def _load_token_ids(self, txt_path: str) -> torch.Tensor:
        """
        Reads single-line transcript and tokenises with GPT-NeoX tokenizer.
        Returns all-pad tensor if file is empty or missing.

        Returns
        -------
        token_ids : (max_text_len,)
        """
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            return torch.full(
                (self.cfg.max_text_len,), self.pad_id, dtype=torch.long
            )

        ids = self.tokenizer(
            text,
            max_length      = self.cfg.max_text_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )["input_ids"].squeeze(0)   # (max_text_len,)

        # Replace tokenizer's own pad with our safe pad_id
        ids[ids == self.tokenizer.pad_token_id] = self.pad_id
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, txt_path = self.samples[idx]
        mel       = self._load_mel(wav_path)
        token_ids = self._load_token_ids(txt_path)
        return mel, token_ids


def collate_fn(batch):
    """Stack mel and token_ids into batched tensors."""
    mels      = torch.stack([b[0] for b in batch])   # (B, max_frames, n_mels)
    token_ids = torch.stack([b[1] for b in batch])   # (B, max_text_len)
    return mels, token_ids


# ════════════════════════════════════════════════════════════════════════════
# MODEL SETUP
# ════════════════════════════════════════════════════════════════════════════

def build_model(cfg: MultimodalTrainConfig, device: torch.device, freeze_backbone: bool) -> SpeakMK1Multimodal:
    """
    Constructs SpeakMK1Multimodal, loads pretrained encoder and LLM weights,
    and applies the freeze policy for the current training phase.
    """
    audio_cfg = SmallConfig(d_model=cfg.audio_d_model, llm_dim=cfg.llm_d_model)

    llm_cfg = SpeakMK1LLMConfig(
        vocab_size      = cfg.vocab_size,    # 50283 — always override
        d_model         = cfg.llm_d_model,
        d_state         = 64,
        num_blocks      = 6,
        nheads_ssm      = 8,
        nheads_attn     = 8,
        top_k_audio     = 32,
        num_experts     = cfg.num_experts,   # 4 — always override
        top_k_experts   = 2,
        dropout         = 0.0,
        aux_loss_weight = 1e-2,
    )

    model = SpeakMK1Multimodal(
        audio_cfg       = audio_cfg,
        llm_cfg         = llm_cfg,
        freeze_backbone = freeze_backbone,
    ).to(device)

    print("\nLoading pretrained weights...")
    model.load_audio_encoder(cfg.audio_ckpt, device)
    model.load_llm(cfg.llm_ckpt, device)

    trainable = model.trainable_param_count() / 1e6
    total     = model.total_param_count()     / 1e6
    print(f"\n  Trainable : {trainable:.3f}M / {total:.3f}M total")
    print(f"  Backbone frozen : {freeze_backbone}\n")

    return model


# ════════════════════════════════════════════════════════════════════════════
# OPTIMISER + SCHEDULER
# ════════════════════════════════════════════════════════════════════════════

def build_optimizer_and_scheduler(
    model:        SpeakMK1Multimodal,
    cfg:          MultimodalTrainConfig,
    lr:           float,
    warmup_steps: int,
    total_steps:  int,
) -> tuple:
    """
    AdamW on trainable params only.
    SequentialLR: linear warmup → cosine decay.
    """
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr           = lr,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.95),
    )
    warmup = LinearLR(
        optimizer,
        start_factor = 0.01,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max = max(total_steps - warmup_steps, 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_steps],
    )
    return optimizer, scheduler


# ════════════════════════════════════════════════════════════════════════════
# SINGLE EPOCH
# ════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:       SpeakMK1Multimodal,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    scaler:      GradScaler,
    cfg:         MultimodalTrainConfig,
    device:      torch.device,
    epoch:       int,
    global_step: int,
    total_steps: int,
    phase:       int,
) -> tuple:
    """
    Runs one full epoch of training.

    Returns
    -------
    global_step : updated step count
    avg_lm_loss : mean LM loss over optimizer steps this epoch
    avg_aux_loss: mean aux loss over optimizer steps this epoch
    """
    model.train()
    optimizer.zero_grad()

    epoch_lm  = 0.0
    epoch_aux = 0.0
    opt_steps = 0
    nan_count = 0
    t0        = time.time()

    for step, (mel, token_ids) in enumerate(loader):
        mel       = mel.to(device)
        token_ids = token_ids.to(device)

        # ── Build labels: mask padding with -100 ──────────────────────────
        labels = token_ids.clone()
        labels[labels == cfg.pad_token_id] = -100

        # Sanity check on very first step of training
        if epoch == 0 and step == 0 and phase == 1:
            valid_frac = (labels != -100).float().mean().item()
            print(f"  [Sanity] token min={token_ids.min().item()} "
                  f"max={token_ids.max().item()} "
                  f"valid_label_frac={valid_frac:.3f}")
            if valid_frac < 0.05:
                raise RuntimeError(
                    "Almost all labels masked — transcript loading is broken. "
                    f"pad_token_id={cfg.pad_token_id}"
                )

        # ── Forward pass ──────────────────────────────────────────────────
        with autocast(device.type, dtype=torch.bfloat16):
            _, lm_loss, aux_loss = model(
                mel       = mel,
                input_ids = token_ids,
                labels    = labels,
            )

        # ── NaN guard ─────────────────────────────────────────────────────
        if torch.isnan(lm_loss):
            nan_count += 1
            if nan_count <= 5:
                print(f"  [NaN] step={step} — skipping")
            optimizer.zero_grad()
            continue
        nan_count = 0

        # ── Backward ──────────────────────────────────────────────────────
        total_loss = (lm_loss + aux_loss) / cfg.accumulation_steps
        scaler.scale(total_loss).backward()

        # ── Optimizer step ────────────────────────────────────────────────
        if (step + 1) % cfg.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_lm  += lm_loss.item()
            epoch_aux += aux_loss.item()
            opt_steps += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"Phase {phase} | Ep {epoch+1} | "
                    f"Step {global_step}/{total_steps} | "
                    f"lm={lm_loss.item():.4f} | "
                    f"aux={aux_loss.item():.4f} | "
                    f"gnorm={grad_norm:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.1f}s"
                )
                t0 = time.time()

    avg_lm  = epoch_lm  / max(opt_steps, 1)
    avg_aux = epoch_aux / max(opt_steps, 1)
    return global_step, avg_lm, avg_aux


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE / LOAD
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     SpeakMK1Multimodal,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    avg_lm:    float,
    phase:     int,
    cfg:       MultimodalTrainConfig,
    tag:       str = "",
):
    """
    Saves full model state + optimizer + scheduler.
    Also saves component state dicts separately for easy extraction.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    name = f"phase{phase}_epoch{epoch+1}{tag}.pt"
    path = os.path.join(cfg.save_dir, name)
    torch.save({
        # Full model
        "model"          : model.state_dict(),
        # Components separately — useful for loading into original scripts
        "audio_encoder"  : model.audio_encoder.state_dict(),
        "audio_proj"     : model.audio_proj.state_dict(),
        "llm"            : model.llm.state_dict(),
        # Training state
        "optimizer"      : optimizer.state_dict(),
        "scheduler"      : scheduler.state_dict(),
        "epoch"          : epoch + 1,
        "phase"          : phase,
        "avg_lm_loss"    : avg_lm,
    }, path)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train():
    cfg    = MultimodalTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("=" * 65)
    print("SpeakMK1 — Multimodal Joint Training")
    print(f"  device     : {device}")
    print(f"  data_root  : {cfg.data_root}")
    print(f"  batch_size : {cfg.batch_size}  (accum={cfg.accumulation_steps})")
    print("=" * 65)

    # ── Dataset & Loader ──────────────────────────────────────────────────
    print("\nIndexing dataset...")
    dataset = LibriSpeechMultimodalDataset(cfg)
    loader  = DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = cfg.num_workers,
        pin_memory  = True,
        drop_last   = True,   # keeps batch size consistent
    )
    steps_per_epoch = len(loader) // cfg.accumulation_steps
    print(f"  {len(dataset)} samples | {len(loader)} loader steps | "
          f"{steps_per_epoch} optimizer steps/epoch")

    # ── GradScaler for bf16 ───────────────────────────────────────────────
    scaler = GradScaler()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1 — frozen backbone, train encoder + proj + cross_attn
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PHASE 1 — Backbone frozen")
    print("  Training: audio_encoder, audio_proj, cross_attn weights")
    print("=" * 65)

    model = build_model(cfg, device, freeze_backbone=True)

    total_p1 = steps_per_epoch * cfg.phase1_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, cfg.phase1_lr, cfg.phase1_warmup_steps, total_p1
    )

    global_step = 0
    best_lm     = float("inf")

    for epoch in range(cfg.phase1_epochs):
        torch.cuda.empty_cache()
        global_step, avg_lm, avg_aux = run_epoch(
            model, loader, optimizer, scheduler, scaler,
            cfg, device, epoch, global_step, total_p1, phase=1,
        )
        print(f"\n{'='*65}")
        print(f"Phase 1 | Epoch {epoch+1}/{cfg.phase1_epochs} | "
              f"avg_lm={avg_lm:.4f} | avg_aux={avg_aux:.4f}")
        print(f"{'='*65}\n")

        if cfg.save_every_epoch:
            save_checkpoint(model, optimizer, scheduler,
                            epoch, avg_lm, phase=1, cfg=cfg)

        if avg_lm < best_lm:
            best_lm = avg_lm
            save_checkpoint(model, optimizer, scheduler,
                            epoch, avg_lm, phase=1, cfg=cfg, tag="_best")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2 — unfreeze all, joint fine-tune
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PHASE 2 — Full joint fine-tuning")
    print("  Training: everything (lr lowered to avoid catastrophic forgetting)")
    print("=" * 65)

    # Unfreeze backbone — keep the same model, just change requires_grad
    model.set_freeze_backbone(freeze=False)

    trainable = model.trainable_param_count() / 1e6
    print(f"  Trainable params now: {trainable:.3f}M\n")

    total_p2 = steps_per_epoch * cfg.phase2_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, cfg.phase2_lr, cfg.phase2_warmup_steps, total_p2
    )

    global_step = 0
    best_lm_p2  = float("inf")

    for epoch in range(cfg.phase2_epochs):
        torch.cuda.empty_cache()
        global_step, avg_lm, avg_aux = run_epoch(
            model, loader, optimizer, scheduler, scaler,
            cfg, device, epoch, global_step, total_p2, phase=2,
        )
        print(f"\n{'='*65}")
        print(f"Phase 2 | Epoch {epoch+1}/{cfg.phase2_epochs} | "
              f"avg_lm={avg_lm:.4f} | avg_aux={avg_aux:.4f}")
        print(f"{'='*65}\n")

        if cfg.save_every_epoch:
            save_checkpoint(model, optimizer, scheduler,
                            epoch, avg_lm, phase=2, cfg=cfg)

        if avg_lm < best_lm_p2:
            best_lm_p2 = avg_lm
            save_checkpoint(model, optimizer, scheduler,
                            epoch, avg_lm, phase=2, cfg=cfg, tag="_best")

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(cfg.save_dir, "speakmk1_multimodal_final.pt")
    torch.save({
        "model"        : model.state_dict(),
        "audio_encoder": model.audio_encoder.state_dict(),
        "audio_proj"   : model.audio_proj.state_dict(),
        "llm"          : model.llm.state_dict(),
    }, final_path)
    print(f"\nFinal model saved → {final_path}")
    print(f"Phase 1 best lm: {best_lm:.4f}")
    print(f"Phase 2 best lm: {best_lm_p2:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    train()