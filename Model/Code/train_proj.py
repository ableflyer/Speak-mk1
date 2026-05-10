"""
train_audio_proj.py  (RTX 4060 8GB)
====================================
Trains ONLY the audio_proj bridge layer.
Requires cache from precompute_audio_feats.py (GPT-NeoX token IDs).
"""

import os
import time
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from UniMamba import RMSNorm


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

class ProjTrainConfig:
    # ── Paths ─────────────────────────────────────────────────────────────
    # Point to the cache produced by precompute_audio_feats.py
    cache_dir   = "../Data/audio_feats_cache_neox"
    llm_ckpt    = "../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt"
    save_dir    = "../Model_files/audio_proj_training"

    # ── Dims — must match your LLM ────────────────────────────────────────
    audio_d_model = 512
    llm_d_model   = 512

    # ── Tokeniser ─────────────────────────────────────────────────────────
    # GPT-NeoX pad token ID — eos token is used as pad
    # gpt-neox-20b eos_token_id = 0
    # Verify with: AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b").eos_token_id
    pad_token_id  = 0

    # ── Training ──────────────────────────────────────────────────────────
    epochs             = 3
    # batch_size=None because each cached file IS already a batch
    accumulation_steps = 8     # effective batch = 8 files × 4 samples = 32
    lr                 = 3e-4
    weight_decay       = 0.01
    warmup_steps       = 200
    max_grad_norm      = 1.0
    log_every          = 50
    save_every_epoch   = True

    # ── Misc ──────────────────────────────────────────────────────────────
    num_workers = 2
    seed        = 42


# ════════════════════════════════════════════════════════════════════════════
# PROJECTION
# ════════════════════════════════════════════════════════════════════════════

class DirectAudioProjection(nn.Module):
    """(B, T, audio_d_model) → (B, T, llm_d_model)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim,      out_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim,     bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


def init_projection(in_dim: int, out_dim: int, device: torch.device) -> DirectAudioProjection:
    proj = DirectAudioProjection(in_dim, out_dim).to(device)
    for m in proj.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    print(f"  audio_proj initialised: {in_dim} → {out_dim}")
    return proj


# ════════════════════════════════════════════════════════════════════════════
# CACHED DATASET
# ════════════════════════════════════════════════════════════════════════════

class CachedFeatDataset(Dataset):
    """
    Each file is a pre-batched dict with keys:
        feats              : (B, T_audio, 512)
        token_ids          : (B, max_text_len)   <- GPT-NeoX IDs
        audio_padding_mask : (B, T_audio)
    """
    def __init__(self, cache_dir: str):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "batch_*.pt")))
        if not self.files:
            raise RuntimeError(
                f"No cached files found in {cache_dir}.\n"
                f"Run precompute_audio_feats.py first."
            )
        # Validate first file has correct keys
        sample = torch.load(self.files[0], map_location="cpu")
        assert "token_ids" in sample, (
            "Cache uses old 'byte_ids' key — regenerate with precompute_audio_feats.py"
        )
        print(f"  Cache validated : {len(self.files)} batches")
        print(f"  token_ids shape : {sample['token_ids'].shape}")
        print(f"  token_ids range : [{sample['token_ids'].min()}, {sample['token_ids'].max()}]")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        return d["feats"], d["token_ids"], d["audio_padding_mask"]


# ════════════════════════════════════════════════════════════════════════════
# LLM  (frozen, forward-pass only)
# ════════════════════════════════════════════════════════════════════════════

def load_llm(cfg: ProjTrainConfig, device: torch.device) -> SpeakMK1LLM:
    model_cfg = SpeakMK1LLMConfig(
        vocab_size      = 50283,
        d_model         = cfg.llm_d_model,
        d_state         = 64,
        num_blocks      = 6,
        nheads_ssm      = 8,
        nheads_attn     = 8,
        top_k_audio     = 32,
        num_experts     = 4,
        top_k_experts   = 2,
        dropout         = 0.0,
        aux_loss_weight = 1e-2,
    )
    model = SpeakMK1LLM(model_cfg).to(device)
    ckpt  = torch.load(cfg.llm_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    print("  LLM loaded.")

    # Freeze all LLM weights — requires_grad=False means no gradient
    # storage, but the forward pass still builds a compute graph
    # through audio_out so gradients flow back into audio_proj.
    for param in model.parameters():
        param.requires_grad = False
        
    for block in model.blocks:
        if hasattr(block.cross_attn, 'gate'):
            block.cross_attn.gate.data.fill_(1.0)  # tanh(1.0) = 0.76, open enough
            block.cross_attn.gate.requires_grad = True  # let it tune during proj training
    
    
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train():
    cfg    = ProjTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("=" * 60)
    print("SpeakMK1 — audio_proj bridge training")
    print(f"  device     : {device}")
    print(f"  cache_dir  : {cfg.cache_dir}")
    print(f"  pad_token  : {cfg.pad_token_id}")
    print("=" * 60)

    # ── Models ────────────────────────────────────────────────────────────
    print("\nLoading models...")
    llm        = load_llm(cfg, device)
    audio_proj = init_projection(cfg.audio_d_model, cfg.llm_d_model, device)
    print(f"  Trainable params : "
          f"{sum(p.numel() for p in audio_proj.parameters())/1e6:.3f}M\n")
    
    for i, block in enumerate(llm.blocks):
        if hasattr(block.cross_attn, 'gate'):
            g = block.cross_attn.gate.item()
            print(f"  block[{i}] gate={g:.4f}  tanh={torch.tanh(torch.tensor(g)):.4f}")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = CachedFeatDataset(cfg.cache_dir)
    loader  = DataLoader(
        dataset,
        batch_size  = None,    # each file is already a full batch
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    total_opt_steps = (len(loader) // cfg.accumulation_steps) * cfg.epochs
    print(f"Cached batches : {len(dataset)}")
    print(f"Est. samples   : {len(dataset) * 4}")  # 4 = BATCH_SIZE from precompute
    sample = torch.load(dataset.files[0], map_location="cpu")
    print(f"Samples per file : {sample['feats'].shape[0]}")
    print(f"Total samples    : {len(dataset) * sample['feats'].shape[0]}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        audio_proj.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )
    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=cfg.warmup_steps)
    cosine    = CosineAnnealingLR(optimizer,
                                  T_max=max(total_opt_steps - cfg.warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.warmup_steps])

    # ── Loop ──────────────────────────────────────────────────────────────
    print("Starting training...\n")
    global_step = 0
    best_loss   = float("inf")
    nan_count   = 0

    for epoch in range(cfg.epochs):
        torch.cuda.empty_cache()
        audio_proj.train()
        epoch_lm_loss  = 0.0
        epoch_aux_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, (feats, token_ids, audio_padding_mask) in enumerate(loader):
            feats              = feats.to(device)
            token_ids          = token_ids.to(device)
            audio_padding_mask = audio_padding_mask.to(device)

            with autocast(device.type, dtype=torch.bfloat16):
                # ── Project audio features ────────────────────────────
                audio_out = audio_proj(feats)   # (B, T_audio, llm_d_model)
                # print(f"audio_out.requires_grad: {audio_out.requires_grad}")
                # print(f"audio_out.grad_fn: {audio_out.grad_fn}")
                # ── Build labels ──────────────────────────────────────
                # Mask pad positions with -100 so CE loss ignores them.
                # Use the actual GPT-NeoX pad token ID, not hardcoded 0.
                if epoch == 0 and step == 0:
                    audio_out.retain_grad()
                    
                labels = token_ids.clone()
                labels[labels == cfg.pad_token_id] = -100

                # Sanity check on first step only
                if epoch == 0 and step == 0:
                    valid_frac = (labels != -100).float().mean().item()
                    print(f"  Sanity check — token_ids: "
                          f"min={token_ids.min().item()} "
                          f"max={token_ids.max().item()} "
                          f"valid_label_frac={valid_frac:.3f}")
                    if valid_frac < 0.1:
                        raise RuntimeError(
                            "Almost all labels are masked — wrong pad_token_id "
                            f"(cfg.pad_token_id={cfg.pad_token_id}) or corrupt cache.\n"
                            "Check: AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b').eos_token_id"
                        )
                    audio_out_test = audio_proj(feats)
                    audio_out_test.retain_grad()
                    
                    # run just the cross attention of first block
                    text_hidden = llm.embed_proj(llm.embedding(token_ids))  # or however you get initial hidden states
                    ca_out = llm.blocks[0].cross_attn(
                        text_hidden=text_hidden,
                        audio_out=audio_out_test,
                        audio_padding_mask=audio_padding_mask,
                    )
                    ca_out.sum().backward()
                    print(f"cross_attn grad on audio_out: {audio_out_test.grad}")
                # ── LLM forward (frozen weights, live compute graph) ──
                # audio_out.requires_grad=True because audio_proj is trained.
                # LLM weights have requires_grad=False but the forward pass
                # still builds a graph through audio_out, so gradients
                # flow back into audio_proj correctly.
                
                logits, lm_loss, aux_loss = llm(
                    input_ids          = token_ids,
                    labels             = labels,
                    audio_out          = audio_out,
                    audio_padding_mask = audio_padding_mask,
                )

                if torch.isnan(lm_loss):
                    nan_count += 1
                    if nan_count <= 5:
                        print(f"  [NaN step={step}] "
                              f"audio_out mean={audio_out.mean().item():.4f} "
                              f"abs_max={audio_out.abs().max().item():.4f}")
                    optimizer.zero_grad()
                    continue

                nan_count  = 0
                total_loss = (lm_loss + aux_loss) / cfg.accumulation_steps

            total_loss.backward()
            if epoch == 0 and step == 0:
                print(f"lm_loss.grad_fn: {lm_loss.grad_fn}")
                print(f"audio_out.grad: {audio_out.grad}")

            if (step + 1) % cfg.accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    audio_proj.parameters(), cfg.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                epoch_lm_loss  += lm_loss.item()
                epoch_aux_loss += aux_loss.item()
                global_step    += 1

                if global_step % cfg.log_every == 0:
                    elapsed = time.time() - t0
                    print(
                        f"Epoch {epoch+1}/{cfg.epochs} | "
                        f"Step {global_step}/{total_opt_steps} | "
                        f"lm={lm_loss.item():.4f} | "
                        f"aux={aux_loss.item():.4f} | "
                        f"grad_norm={grad_norm:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"{elapsed:.1f}s"
                    )
                    t0 = time.time()

        # ── Epoch end ─────────────────────────────────────────────────
        n_opt   = max(len(loader) // cfg.accumulation_steps, 1)
        avg_lm  = epoch_lm_loss  / n_opt
        avg_aux = epoch_aux_loss / n_opt
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} complete | avg_lm={avg_lm:.4f} | avg_aux={avg_aux:.4f}")
        print(f"{'='*60}\n")

        if cfg.save_every_epoch:
            path = os.path.join(cfg.save_dir, f"audio_proj_epoch_{epoch+1}.pt")
            torch.save({
                "audio_proj" : audio_proj.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "epoch"      : epoch + 1,
                "avg_lm_loss": avg_lm,
            }, path)
            print(f"  Saved → {path}")

        if avg_lm < best_loss:
            best_loss = avg_lm
            best_path = os.path.join(cfg.save_dir, "audio_proj_best.pt")
            torch.save({"audio_proj": audio_proj.state_dict()}, best_path)
            print(f"  New best → {best_path}  (lm={best_loss:.4f})\n")

    print("Training complete.")
    print(f"Best lm_loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()