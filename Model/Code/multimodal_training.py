"""
This is not used, instead we trained a projection layer at train_proj.py
train_multimodal.py
===================
Trains SpeakMK1Multimodal end-to-end from mel spectrograms.
No precomputed cache needed — audio encoder is trained jointly.

Two-phase training:
    Phase 1: freeze LLM backbone, train encoder + proj + cross_attn
    Phase 2: unfreeze all, joint fine-tune at lower lr

Dual-stream data:
    Stream A — Acoustic (LibriSpeech):
        Input : (mel, transcript_token_ids)
        Loss  : LM loss on all transcript tokens
        Learns: audio → text grounding

    Stream B — Clinical (SLP JSONL):
        Input : (zero_mel, dialogue_token_ids)
        Loss  : LM loss on <|slp|> response tokens ONLY  (child + think masked)
        Learns: clinical response generation
        Note  : mel is zeroed — cross-attention learns to be inactive here.
                When real child audio is available later, swap zero_mel for
                a real spectrogram to get full audio-conditioned SLP output.

Data expected:
    LibriSpeech: <root>/<speaker>/<chapter>/
        19-198-0000.wav
        19-198-0000.txt   ← single-line transcript (no ID prefix)

    SLP JSONL: one JSON object per line, same schema as train_speakmk1_llm.py
        {"metadata": {...}, "dialogue": [{"child_input": ..., "model_thought": ..., "slp_response": ...}, ...]}
"""

import os
import glob
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer
import librosa
import torchaudio

from audio_trainer import SmallConfig
from speak_mk1_llm import SpeakMK1LLMConfig
from speak_mk1_combined import SpeakMK1Multimodal


# ════════════════════════════════════════════════════════════════════════════
# SPECIAL TOKENS  (must match train_speakmk1_llm.py exactly)
# ════════════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = {
    "system_start": "<|system|>",
    "child_start":  "<|child|>",
    "think_start":  "<|think|>",
    "slp_start":    "<|slp|>",
    "turn_end":     "<|endturn|>",
    "seq_end":      "<|endseq|>",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are SpeakMK1, a warm and clinically expert AI speech-language pathologist for children. "
    "The child is {age} years old and presents with {disorder} ({specific_error}). "
    "Clinical goal: {clinical_goal}. "
    "Primary strategy: {strategy}. "
    "Always analyze errors clinically before responding. Be encouraging, patient, and use scaffolding."
)


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

class MultimodalTrainConfig:

    # ── Paths ─────────────────────────────────────────────────────────────
    data_root       = "../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100"
    slp_jsonl_path  = "../Data/SLP/slp_stage3_train_clean.jsonl"
    audio_ckpt      = "../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_2.pt"
    llm_ckpt        = "../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt"
    save_dir        = "../Model_files/multimodal_training_v2.1"

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer_name  = "EleutherAI/gpt-neox-20b"
    pad_token_id    = 1
    max_text_len    = 512     # increased from 256 — SLP dialogues are longer
    max_slp_len     = 1024    # cap for full SLP dialogue sequences

    # ── Audio ─────────────────────────────────────────────────────────────
    sample_rate     = 16_000
    n_mels          = 80
    n_fft           = 400
    hop_length      = 160
    max_frames      = 512

    # ── Model dims — must match checkpoints ───────────────────────────────
    audio_d_model   = 512
    llm_d_model     = 512
    vocab_size      = 50283
    num_experts     = 4

    # ── Dual-stream mix ratio ──────────────────────────────────────────────
    # Fraction of each batch drawn from SLP stream vs LibriSpeech stream.
    slp_mix_ratio   = 0.3

    # ── Phase 1 training (backbone frozen) ────────────────────────────────
    phase1_epochs       = 2
    phase1_lr           = 2e-4
    phase1_warmup_steps = 100

    # ── Phase 2 training (full joint fine-tune) ────────────────────────────
    phase2_epochs       = 2
    phase2_lr           = 3e-5
    phase2_warmup_steps = 50

    # ── Common ────────────────────────────────────────────────────────────
    batch_size         = 2
    accumulation_steps = 16
    weight_decay       = 0.01
    max_grad_norm      = 1.0
    log_every          = 50
    save_every_epoch   = True
    num_workers        = 2
    seed               = 42
    gate_loss_weight   = 1e-2


# ════════════════════════════════════════════════════════════════════════════
# TOKENIZER HELPER
# ════════════════════════════════════════════════════════════════════════════

def get_tokenizer(cfg: MultimodalTrainConfig) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.add_special_tokens({
        "additional_special_tokens": list(SPECIAL_TOKENS.values()) + ["<|ctc_blank|>"]
    })
    return tok


# ════════════════════════════════════════════════════════════════════════════
# STREAM A — ACOUSTIC DATASET  (LibriSpeech wav + transcript)
# ════════════════════════════════════════════════════════════════════════════

class LibriSpeechMultimodalDataset(Dataset):
    """
    Returns (mel, token_ids, labels) triples.

    mel       : (max_frames, n_mels)  — log-mel spectrogram, padded/truncated
    token_ids : (max_text_len,)       — GPT-NeoX token IDs, padded
    labels    : (max_text_len,)       — same as token_ids (-100 at pad positions)
                                        LM loss on ALL transcript tokens.
    stream    : str = "acoustic"      — tag so collate can distinguish streams
    """

    def __init__(self, cfg: MultimodalTrainConfig, tokenizer: AutoTokenizer):
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.pad_id    = cfg.pad_token_id
        self.audio_dropout_prob = 0.15
        ctc_blank_id = None

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = cfg.sample_rate,
            n_fft       = cfg.n_fft,
            hop_length  = cfg.hop_length,
            n_mels      = cfg.n_mels,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.samples = self._build_index()
        print(f"LibriSpeechMultimodalDataset : {len(self.samples):,} samples")

    def _build_index(self):
        wav_files = glob.glob(
            os.path.join(self.cfg.data_root, "**/*.wav"), recursive=True
        )
        samples = []
        for wav_path in sorted(wav_files):
            txt_path = wav_path.replace(".wav", ".txt")
            if os.path.exists(txt_path):
                samples.append((wav_path, txt_path))
        return samples

    def _load_mel(self, wav_path: str):
        # print("LOAD_MEL CALLED")
        y, _ = librosa.load(wav_path, sr=self.cfg.sample_rate)
        waveform = torch.from_numpy(y).unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel).squeeze(0).T  # (T, n_mels)
        T = mel.shape[0]
        real_frames = min(T, self.cfg.max_frames)
        # print(f"T={T} real_frames={real_frames}")
        if T >= self.cfg.max_frames:
            mel = mel[:self.cfg.max_frames]
        else:
            mel = F.pad(mel, (0, 0, 0, self.cfg.max_frames - T))
        return mel, real_frames

    def _load_token_ids(self, txt_path: str):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if " " in text:
            text = text.split(" ", 1)[1]

        L = self.cfg.max_text_len

        if not text:
            ids    = torch.full((L,), self.pad_id, dtype=torch.long)
            labels = torch.full((L,), -100,         dtype=torch.long)
            return ids, labels

        ids = self.tokenizer(
            text,
            max_length     = L,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )["input_ids"].squeeze(0)

        ids[ids == self.tokenizer.pad_token_id] = self.pad_id

        labels = ids.clone()
        labels[labels == self.pad_id] = -100
        return ids, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, txt_path = self.samples[idx]
        mel, real_frames = self._load_mel(wav_path)
        if np.random.rand() < self.audio_dropout_prob:
            mel.zero_()
        token_ids, labels = self._load_token_ids(txt_path)
        # NEW: raw transcript tokens for CTC (no special tokens, no padding to max_text_len)
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        
        # CTC labels: just the transcript token IDs, no padding (handled in collate)
        transcript_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
        transcript_ids = torch.tensor(transcript_ids, dtype=torch.long)
        
        return mel, token_ids, labels, transcript_ids, real_frames, "acoustic"


# ════════════════════════════════════════════════════════════════════════════
# STREAM B — CLINICAL DATASET  (SLP JSONL dialogues)
# ════════════════════════════════════════════════════════════════════════════

class SLPMultimodalDataset(Dataset):
    """
    Converts SLP JSONL dialogues into (mel, token_ids, labels) triples
    where mel is an all-zero tensor (no audio available for SLP data).

    Token layout per sample (mirrors train_speakmk1_llm.py):
        <|system|> ... <|endturn|>
        <|child|>  ... <|endturn|>     ← masked in labels (-100)
        <|think|>  ... <|endturn|>     ← masked in labels (-100)
        <|slp|>    ... <|endturn|>     ← TRAINABLE in labels
        ... (repeated per dialogue turn)
        <|endseq|>

    The model is forced to produce correct SLP responses regardless of
    what the (zero) mel says. Combined with Stream A, the audio encoder
    learns to inform transcription while the LLM backbone learns both
    transcription (Stream A) and clinical response (Stream B).
    """

    def __init__(self, cfg: MultimodalTrainConfig, tokenizer: AutoTokenizer):
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.samples   = self._load_jsonl(cfg.slp_jsonl_path)
        self.zero_mel  = torch.zeros(cfg.max_frames, cfg.n_mels)
        print(f"SLPMultimodalDataset        : {len(self.samples):,} dialogues")

    def _load_jsonl(self, path: str) -> list:
        if not os.path.exists(path):
            print(f"  WARNING: SLP JSONL not found at {path} — SLP stream disabled.")
            return []
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return samples

    def _format_dialogue(self, entry: dict):
        """
        Tokenise one SLP dialogue entry.
        Returns (token_ids, labels) as Python lists of ints.
        Labels are -100 everywhere except on <|slp|> response tokens.
        """
        tok  = self.tokenizer
        meta = entry.get("metadata", {})

        system_text = (
            SPECIAL_TOKENS["system_start"] + " " +
            SYSTEM_PROMPT_TEMPLATE.format(
                age            = meta.get("age",            "unknown"),
                disorder       = meta.get("disorder",       "speech disorder"),
                specific_error = meta.get("specific_error", ""),
                clinical_goal  = meta.get("clinical_goal",  ""),
                strategy       = meta.get("primary_strategy", ""),
            ) + " " + SPECIAL_TOKENS["turn_end"]
        )

        all_ids    = []
        all_labels = []

        # System block — masked
        sys_ids = tok.encode(system_text, add_special_tokens=False)
        all_ids    += sys_ids
        all_labels += [-100] * len(sys_ids)

        for turn in entry.get("dialogue", []):
            child_ids = tok.encode(
                SPECIAL_TOKENS["child_start"] + " " +
                turn.get("child_input", "") + " " +
                SPECIAL_TOKENS["turn_end"],
                add_special_tokens=False,
            )
            think_ids = tok.encode(
                SPECIAL_TOKENS["think_start"] + " " +
                turn.get("model_thought", "") + " " +
                SPECIAL_TOKENS["turn_end"],
                add_special_tokens=False,
            )
            slp_ids = tok.encode(
                SPECIAL_TOKENS["slp_start"] + " " +
                turn.get("slp_response", "") + " " +
                SPECIAL_TOKENS["turn_end"],
                add_special_tokens=False,
            )

            # Child + think — masked
            all_ids    += child_ids + think_ids
            all_labels += [-100] * (len(child_ids) + len(think_ids))

            # SLP response — TRAINABLE
            all_ids    += slp_ids
            all_labels += slp_ids   # loss on response tokens

        eos_ids = tok.encode(SPECIAL_TOKENS["seq_end"], add_special_tokens=False)
        all_ids    += eos_ids
        all_labels += eos_ids

        # Truncate to max_slp_len
        L = self.cfg.max_slp_len
        all_ids    = all_ids[:L]
        all_labels = all_labels[:L]

        # Pad to max_slp_len
        pad_len = L - len(all_ids)
        all_ids    += [self.cfg.pad_token_id] * pad_len
        all_labels += [-100]                  * pad_len

        return (
            torch.tensor(all_ids,    dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        token_ids, labels = self._format_dialogue(entry)
        return self.zero_mel.clone(), token_ids, labels, None, 0, "clinical"


# ════════════════════════════════════════════════════════════════════════════
# COLLATE — handles variable token lengths between streams
# ════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    mels = torch.stack([b[0] for b in batch])

    max_len = max(b[1].shape[0] for b in batch)
    pad_id = 1
    token_ids = torch.stack([
        F.pad(b[1], (0, max_len - b[1].shape[0]), value=pad_id)
        for b in batch
    ])
    labels = torch.stack([
        F.pad(b[2], (0, max_len - b[2].shape[0]), value=-100)
        for b in batch
    ])
    
    # NEW: collate transcript_ids for CTC (variable length, pad with -100)
    valid_transcripts = [b[3] for b in batch if b[3] is not None]
    max_trans_len = max((t.shape[0] for t in valid_transcripts), default=1)
    transcript_ids_list = []
    for b in batch:
        if b[3] is not None:
            padded = F.pad(b[3], (0, max_trans_len - b[3].shape[0]), value=-100)
        else:
            padded = torch.full((max_trans_len,), -100, dtype=torch.long)
        transcript_ids_list.append(padded)
    transcript_ids = torch.stack(transcript_ids_list)
    real_frames = torch.tensor([b[4] for b in batch], dtype=torch.long)
    streams = [b[5] for b in batch]
    return mels, token_ids, labels, transcript_ids, real_frames, streams


# ════════════════════════════════════════════════════════════════════════════
# DUAL-STREAM DATALOADER
# ════════════════════════════════════════════════════════════════════════════

def build_dual_stream_loader(cfg: MultimodalTrainConfig, tokenizer: AutoTokenizer) -> DataLoader:
    """
    Builds a single DataLoader over both streams using WeightedRandomSampler
    so that ~slp_mix_ratio fraction of each batch is clinical data.

    If SLP JSONL is missing, falls back to acoustic-only.
    """
    acoustic_ds = LibriSpeechMultimodalDataset(cfg, tokenizer)
    slp_ds      = SLPMultimodalDataset(cfg, tokenizer)
    
    
    
    if len(slp_ds) == 0:
        print("  No SLP data found — training on acoustic stream only.")
        combined = acoustic_ds
        sampler  = None
    else:
        combined = ConcatDataset([acoustic_ds, slp_ds])
        
        # Build per-sample weights to honour slp_mix_ratio
        n_acoustic = len(acoustic_ds)
        n_slp      = len(slp_ds)
        # Weight per acoustic sample
        w_acoustic = (1.0 - cfg.slp_mix_ratio) / n_acoustic
        # Weight per SLP sample
        w_slp      = cfg.slp_mix_ratio          / n_slp

        sample_weights = (
            [w_acoustic] * n_acoustic +
            [w_slp]      * n_slp
        )
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(combined),
            replacement = True,
        )
        print(f"  Dual-stream sampler: {n_acoustic:,} acoustic | "
              f"{n_slp:,} clinical | mix_ratio={cfg.slp_mix_ratio}")

    return DataLoader(
        combined,
        batch_size  = cfg.batch_size,
        sampler     = sampler,
        shuffle     = (sampler is None),
        collate_fn  = collate_fn,
        num_workers = cfg.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )


# ════════════════════════════════════════════════════════════════════════════
# MODEL SETUP
# ════════════════════════════════════════════════════════════════════════════

def build_model(cfg: MultimodalTrainConfig, device: torch.device, freeze_backbone: bool, tokenizer: AutoTokenizer) -> SpeakMK1Multimodal:
    audio_cfg = SmallConfig(d_model=cfg.audio_d_model, llm_dim=cfg.llm_d_model)

    llm_cfg = SpeakMK1LLMConfig(
        vocab_size      = cfg.vocab_size,
        d_model         = cfg.llm_d_model,
        d_state         = 64,
        num_blocks      = 6,
        nheads_ssm      = 8,
        nheads_attn     = 8,
        top_k_audio     = 32,
        num_experts     = cfg.num_experts,
        top_k_experts   = 2,
        dropout         = 0.0,
        aux_loss_weight = 1e-2,
    )

    model = SpeakMK1Multimodal(
        audio_cfg       = audio_cfg,
        llm_cfg         = llm_cfg,
        freeze_backbone = freeze_backbone,
        endturn_id      = tokenizer.convert_tokens_to_ids("<|endturn|>"),
    ).to(device)

    print("\nLoading pretrained weights...")
    model.load_audio_encoder(cfg.audio_ckpt, device)
    model.load_llm(cfg.llm_ckpt, device)
    
    # tokenizer = get_tokenizer(cfg=cfg)   # need tokenizer to check vocab size
    
    actual_vocab = len(tokenizer)
    if actual_vocab > cfg.vocab_size:
        old_emb = model.llm.embedding
        new_emb = nn.Embedding(actual_vocab, old_emb.embedding_dim)
        new_emb.weight.data[:cfg.vocab_size] = old_emb.weight.data
        nn.init.normal_(new_emb.weight.data[cfg.vocab_size:], std=0.02)
        model.llm.embedding = new_emb.to(device) 

        old_head = model.llm.lm_head
        new_head = nn.Linear(old_head.in_features, actual_vocab, bias=False)
        new_head.weight.data[:cfg.vocab_size] = old_head.weight.data
        nn.init.normal_(new_head.weight.data[cfg.vocab_size:], std=0.02)
        model.llm.lm_head = new_head.to(device)
        print(f"  Resized embedding+lmhead: {cfg.vocab_size} → {actual_vocab}")

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
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr           = lr,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.95),
    )
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
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
    Runs one full epoch.  Logs per-stream loss separately for interpretability.

    Returns
    -------
    global_step, avg_lm_loss, avg_aux_loss
    """
    model.train()
    optimizer.zero_grad()

    epoch_lm      = 0.0
    epoch_aux     = 0.0
    epoch_lm_ac   = 0.0   # acoustic-stream LM loss
    epoch_lm_cl   = 0.0   # clinical-stream LM loss
    n_ac          = 0
    n_cl          = 0
    opt_steps     = 0
    nan_count     = 0
    t0            = time.time()

    for step, (mel, token_ids, labels, transcript_ids, real_frames, streams) in enumerate(loader):
        mel       = mel.to(device)
        token_ids = token_ids.to(device)
        labels    = labels.to(device)
        transcript_ids = transcript_ids.to(device)
        real_frames = real_frames.to(device)
        
        is_acoustic = torch.tensor([s == "acoustic" for s in streams], device=device)
        transcript_labels = transcript_ids.clone() if is_acoustic.any() else None
        if transcript_labels is not None:
            transcript_labels[~is_acoustic] = -100

        # ── Sanity check on very first step ───────────────────────────────
        if epoch == 0 and step == 0 and phase == 1:
            mask = (mel.sum(dim=-1) == 0)
            lengths = (~mask).sum(dim=1)
            print(f"input_lengths: {real_frames.tolist()}")
            print(f"padding_mask mean: {(mel.sum(dim=-1) == 0).float().mean().item():.3f}")
            valid_frac = (labels != -100).float().mean().item()
            print(f"  [Sanity] token min={token_ids.min().item()} "
                  f"max={token_ids.max().item()} "
                  f"valid_label_frac={valid_frac:.3f}")
            if valid_frac < 0.02:
                raise RuntimeError(
                    "Almost all labels masked. Check SLP formatting and pad_token_id."
                )

        # ── Forward ───────────────────────────────────────────────────────
        with autocast(device.type, dtype=torch.bfloat16):
            _, lm_loss, aux_loss = model(
                mel=mel,
                input_ids=token_ids,
                labels=labels,
            )

        # ── NaN guard ─────────────────────────────────────────────────────
        if torch.isnan(lm_loss):
            nan_count += 1
            if nan_count <= 5:
                print(f"  [NaN] step={step} streams={set(streams)} — skipping")
            optimizer.zero_grad()
            continue
        nan_count = 0

        # ── Track per-stream loss for logging ─────────────────────────────
        if "acoustic" in streams:
            epoch_lm_ac += lm_loss.item()
            n_ac += 1
        if "clinical" in streams:
            epoch_lm_cl += lm_loss.item()
            n_cl += 1

        # ── Backward ──────────────────────────────────────────────────────
        total_loss = lm_loss + aux_loss
        if phase == 2:
            gate_loss = sum(
                torch.exp(-block.cross_attn.gate.abs())
                for block in model.llm.blocks
                if hasattr(block, 'cross_attn')
            )
            total_loss = total_loss + cfg.gate_loss_weight * gate_loss
        total_loss = total_loss / cfg.accumulation_steps
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
                elapsed   = time.time() - t0
                avg_lm_ac = epoch_lm_ac / max(n_ac, 1)
                avg_lm_cl = epoch_lm_cl / max(n_cl, 1)
                gate_vals = [
                    block.cross_attn.gate.item()
                    for block in model.llm.blocks
                    if hasattr(block, 'cross_attn')
                ]
                gate_str = f"gate={sum(gate_vals)/len(gate_vals):.4f} | " if gate_vals else ""
                print(
                    f"Phase {phase} | Ep {epoch+1} | "
                    f"Step {global_step}/{total_steps} | "
                    f"lm={lm_loss.item():.4f} | "
                    f"lm_ac={avg_lm_ac:.4f} | "
                    f"lm_cl={avg_lm_cl:.4f} | "
                    f"aux={aux_loss.item():.4f} | "
                    f"{gate_str}"
                    f"gnorm={grad_norm:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.1f}s"
                )
                t0 = time.time()

    avg_lm  = epoch_lm  / max(opt_steps, 1)
    avg_aux = epoch_aux / max(opt_steps, 1)
    return global_step, avg_lm, avg_aux


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     SpeakMK1Multimodal,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    epoch:     int,
    avg_lm:    float,
    phase:     int,
    cfg:       MultimodalTrainConfig,
    tag:       str = "",
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    name = f"phase{phase}_epoch{epoch+1}{tag}.pt"
    path = os.path.join(cfg.save_dir, name)
    torch.save({
        "model":         model.state_dict(),   # already contains encoder/proj/llm/audio_to_text
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "scaler":        scaler.state_dict(),
        "epoch":         epoch + 1,
        "phase":         phase,
        "avg_lm_loss":   avg_lm,
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
    print("SpeakMK1 — Multimodal Joint Training (Dual-Stream)")
    print(f"  device       : {device}")
    print(f"  data_root    : {cfg.data_root}")
    print(f"  slp_jsonl    : {cfg.slp_jsonl_path}")
    print(f"  batch_size   : {cfg.batch_size}  (accum={cfg.accumulation_steps})")
    print(f"  slp_mix_ratio: {cfg.slp_mix_ratio}")
    print("=" * 65)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer(cfg)
    cfg.ctc_blank_id = tokenizer.convert_tokens_to_ids("<|ctc_blank|>")
    assert cfg.ctc_blank_id < len(tokenizer), \
        f"blank {cfg.ctc_blank_id} >= vocab {len(tokenizer)}"
    assert cfg.ctc_blank_id != tokenizer.unk_token_id, \
        "CTC blank token resolved to unk_token_id — special token not added properly"
    assert cfg.ctc_blank_id != tokenizer.pad_token_id, \
        "CTC blank token resolved to pad_token_id — special token not added properly"
    print(f"  CTC blank token id: {cfg.ctc_blank_id}")

    # ── GradScaler ────────────────────────────────────────────────────────
    scaler = GradScaler()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1 — frozen backbone
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PHASE 1 — Backbone frozen")
    print("  Training: audio_encoder, audio_proj, cross_attn weights")
    print("  Streams : acoustic (transcription) + clinical (SLP response)")
    print("=" * 65)
    cfg.slp_mix_ratio = 0.0  # ensure no clinical stream in phase 1 (backbone frozen)
    
    # ── Dataset & Loader ──────────────────────────────────────────────────
    print("\nIndexing datasets...")
    loader          = build_dual_stream_loader(cfg, tokenizer)
    steps_per_epoch = len(loader) // cfg.accumulation_steps
    print(f"\n  {len(loader)} loader steps | "
          f"{steps_per_epoch} optimizer steps/epoch")

    model = build_model(cfg, device, freeze_backbone=True, tokenizer=tokenizer)

    total_p1 = steps_per_epoch * cfg.phase1_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, cfg.phase1_lr, cfg.phase1_warmup_steps, total_p1
    )

    global_step = 0
    best_lm     = float("inf")

    for epoch in range(cfg.phase1_epochs):
        global_step, avg_lm, avg_aux = run_epoch(
            model, loader, optimizer, scheduler, scaler,
            cfg, device, epoch, global_step, total_p1, phase=1,
        )
        print(f"\n{'='*65}")
        print(f"Phase 1 | Epoch {epoch+1}/{cfg.phase1_epochs} | "
              f"avg_lm={avg_lm:.4f} | avg_aux={avg_aux:.4f}")
        print(f"{'='*65}\n")

        if cfg.save_every_epoch:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, avg_lm, phase=1, cfg=cfg)
        if avg_lm < best_lm:
            best_lm = avg_lm
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, avg_lm, phase=1, cfg=cfg, tag="_best")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2 — full joint fine-tune
    # ════════════════════════════════════════════════════════════════════════
    if cfg.phase2_epochs > 0:
        print("\n" + "=" * 65)
        print("PHASE 2 — Full joint fine-tuning")
        print("  Training: everything (lowered lr to avoid catastrophic forgetting)")
        print("=" * 65)

        cfg.slp_mix_ratio = 0.3  # restore original mix ratio for phase 2
        
        # ── Dataset & Loader ──────────────────────────────────────────────────
        print("\nIndexing datasets...")
        loader          = build_dual_stream_loader(cfg, tokenizer)
        steps_per_epoch = len(loader) // cfg.accumulation_steps
        print(f"\n  {len(loader)} loader steps | "
            f"{steps_per_epoch} optimizer steps/epoch")

        model.set_freeze_backbone(freeze=False)
        trainable = model.trainable_param_count() / 1e6
        print(f"  Trainable params now: {trainable:.3f}M\n")

        total_p2 = steps_per_epoch * cfg.phase2_epochs
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, cfg, cfg.phase2_lr, cfg.phase2_warmup_steps, total_p2
        )

        global_step  = 0
        best_lm_p2   = float("inf")

        for epoch in range(cfg.phase2_epochs):
            global_step, avg_lm, avg_aux = run_epoch(
                model, loader, optimizer, scheduler, scaler,
                cfg, device, epoch, global_step, total_p2, phase=2,
            )
            print(f"\n{'='*65}")
            print(f"Phase 2 | Epoch {epoch+1}/{cfg.phase2_epochs} | "
                f"avg_lm={avg_lm:.4f} | avg_aux={avg_aux:.4f}")
            print(f"{'='*65}\n")

            if cfg.save_every_epoch:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, avg_lm, phase=2, cfg=cfg)
            if avg_lm < best_lm_p2:
                best_lm_p2 = avg_lm
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, avg_lm, phase=2, cfg=cfg, tag="_best")
                
        print("\nGate values after training:")
        for i, block in enumerate(model.llm.blocks):
            if hasattr(block, 'cross_attn'):
                print(f"  block {i}: {block.cross_attn.gate.item():.6f} "
                    f"(tanh={torch.tanh(block.cross_attn.gate).item():.6f})")

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(cfg.save_dir, "speakmk1_multimodal_final.pt")
    torch.save({
        "model":         model.state_dict(),
    }, final_path)
    print(f"\nFinal model saved → {final_path}")
    print(f"Phase 1 best lm : {best_lm:.4f}")
    if cfg.phase2_epochs > 0:
        print(f"Phase 2 best lm : {best_lm_p2:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    train()