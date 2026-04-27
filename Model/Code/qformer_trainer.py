import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Import your models
from audio_encoder import AudioEncoder, AudioEncoderConfig
from librispeech_byte_dataset import LibriSpeechByteDataset, collate_fn
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig, count_parameters
from bitsandbytes.optim import AdamW8bit
from audio_trainer import SmallConfig
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# # ─────────────────────────────────────────────────────────────────────────────
# # 1. PLACEHOLDER: Your Mamba LLM (Replace with your actual Nemotron/Mamba code)
# # ─────────────────────────────────────────────────────────────────────────────
# class ByteMambaLM(nn.Module):
#     """
#     Placeholder for your custom Mamba-based LLM.
#     Assumptions:
#     - vocab_size: 257 (0=pad, 1-256=bytes)
#     - d_model: matches AudioEncoder.llm_dim
#     """
#     def __init__(self, vocab_size=257, d_model=4096):
#         super().__init__()
#         self.embed_tokens = nn.Embedding(vocab_size, d_model)
#         # ... your Mamba layers would go here ...
#         # For this placeholder, we use a simple linear layer to simulate the LM Head
#         self.lm_head = nn.Linear(d_model, vocab_size)
        
#     def forward(self, inputs_embeds=None, labels=None):
#         # 1. Pass through Mamba layers (skipped here for brevity)
#         # hidden_states = self.mamba_layers(inputs_embeds)
#         hidden_states = inputs_embeds # <--- REPLACE THIS WITH YOUR MAMBA LOGIC
        
#         # 2. Compute Logits
#         logits = self.lm_head(hidden_states)
        
#         # 3. Compute Loss
#         loss = None
#         if labels is not None:
#             # Shift logits and labels for next-token prediction
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
            
#             # Flatten
#             loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Ignore padding/audio tokens
#             loss = loss_fct(
#                 shift_logits.view(-1, shift_logits.size(-1)), 
#                 shift_labels.view(-1)
#             )
            
#         return {"loss": loss, "logits": logits}

#     def get_input_embeddings(self):
#         return self.embed_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class TrainConfig:
    # Data
    librispeech_root = "../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100"
    max_frames = 1024
    max_text_len = 512
    
    # Model Dimensions (Must match Phase 1)
    d_model = 512
    llm_dim = 4096
    
    # Training
    batch_size = 8        # Adjust based on GPU VRAM
    epochs = 10
    lr = 1e-5
    accumulation_steps = 4
    
    # Paths
    checkpoint_path = "../Model_files/audio_encoder_epoch_19.pt" # Your Phase 1 weights

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    temperature = nn.Parameter(torch.tensor(0.5, device=device))

    # --- A. Load Dataset ---
    dataset = LibriSpeechByteDataset(
        librispeech_root=cfg.librispeech_root,
        max_frames=cfg.max_frames,
        max_text_len=cfg.max_text_len
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # --- B. Load Models ---
    
    # 1. Audio Encoder
    audio_cfg = SmallConfig(d_model=cfg.d_model, llm_dim=cfg.llm_dim)
    audio_encoder = AudioEncoder(audio_cfg).to(device)
    
    # Load Phase 1 Weights
    print(f"Loading Phase 1 weights from {cfg.checkpoint_path}...")
    audio_encoder.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device), strict=False)

    # 2. LLM
    model_cfg = SpeakMK1LLMConfig(
        vocab_size=50277,
        d_model=512,
        d_state=64,
        num_blocks=6,
        nheads_ssm=8,
        nheads_attn=8,
        top_k_audio=32,
        num_experts=4,
        top_k_experts=2,
        dropout=0.0,
        aux_loss_weight=1e-2,
    )
    llm = SpeakMK1LLM(model_cfg).to(device)
    llm_checkpoint = "../Model_files/checkpoints_v2/stage1/ckpt_final.pt"
    ckpt = torch.load(llm_checkpoint, map_location=device, weights_only=False)
    llm.load_state_dict(ckpt['model'])
    audio_projection = nn.Sequential(
        nn.Linear(cfg.llm_dim, model_cfg.d_model),
        nn.LayerNorm(model_cfg.d_model),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(model_cfg.d_model, model_cfg.d_model)
    ).to(device)
    for layer in audio_projection:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
    print(f"Added projection: {cfg.llm_dim} -> {model_cfg.d_model}")
    
    # --- C. FREEZING STRATEGY ---
    
    # 1. Freeze Audio Encoder Backbone
    print("Freezing Audio Encoder Backbone...")
    for name, param in audio_encoder.named_parameters():
        # Unfreeze Q-Former + last 2 Mamba/MoE backbone blocks
        # Adjust the block indices based on your architecture (num_blocks=6 in your config)
        if 'qformer' in name:
            param.requires_grad = True
            print(f"  > Training Q-Former: {name}")
        elif 'blocks.4' in name or 'blocks.5' in name or 'blocks.3' in name:
            param.requires_grad = True
            print(f"  > Training Backbone Block: {name}")
        else:
            param.requires_grad = False

    # 2. Freeze LLM (Standard BLIP-2 approach)
    # If your LLM is not pre-trained, you might want to unfreeze this later 
    # or train the LLM separately first.
    print("Freezing LLM...")
    for param in llm.parameters():
        param.requires_grad = False
    # --- D. Optimizer ---
    # Only pass parameters that require gradients
    trainable_params = (
        [p for p in audio_encoder.parameters() if p.requires_grad] +
        list(audio_projection.parameters()) +
        [temperature]  # learn the temperature
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=100)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(loader) - 100)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[100])
    
    scaler = GradScaler(device.type)

    # --- E. Training Loop ---
    print("Starting Phase 2: Q-Former Alignment...")
    
    # Set modes
    audio_encoder.train() # Train mode (BatchNorm/Dropout active in Q-Former)
    llm.eval()            # Eval mode (LLM is frozen)
    # for name, param in audio_encoder.named_parameters():
    #     if 'qformer' in name:
    #         print(f"{name} | requires_grad={param.requires_grad}")
    # time.sleep(2) # Just a moment to review the printout before training starts

    for epoch in range(cfg.epochs):
        total_loss = 0
        
        for i, (mel, byte_ids) in enumerate(loader):
            mel = mel.to(device)
            byte_ids = byte_ids.to(device) # (B, Text_Len)

            with autocast(device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    text_embeds = llm.embed_proj(llm.embedding(byte_ids))  # (B, L, d_model)

                audio_out_raw, _ = audio_encoder(mel)
                audio_embeds = audio_projection(audio_out_raw)  # (B, num_queries, d_model)

                # ── ATC loss (contrastive) ──
                audio_norm = F.normalize(audio_embeds.mean(dim=1), dim=-1)  # (B, d_model)
                text_norm = F.normalize(text_embeds.mean(dim=1).detach(), dim=-1)  # (B, d_model)
                sim_a2t = audio_norm @ text_norm.T / temperature.clamp(min=0.01)  # (B, B)
                sim_t2a = sim_a2t.T
                targets = torch.arange(audio_norm.shape[0], device=device)
                loss_atc = (
                    F.cross_entropy(sim_a2t, targets, label_smoothing=0.1) +
                    F.cross_entropy(sim_t2a, targets, label_smoothing=0.1)
                ) / 2

                # ── LM loss (force qformer output useful to llm) ──
                # concat audio queries + text, predict next text token
                with torch.no_grad():
                    num_queries = audio_embeds.shape[1]
                    audio_labels = torch.full(
                        (byte_ids.shape[0], num_queries), -100,
                        dtype=torch.long, device=device
                    )
                    labels = torch.cat([audio_labels, byte_ids], dim=1)

                inputs_embeds = torch.cat([audio_embeds, text_embeds.detach()], dim=1)
                logits, loss_lm, _ = llm(
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                )
                Q = audio_embeds  # (B, num_queries, d)
                Q_norm = F.normalize(Q, dim=-1)
                query_sim = torch.bmm(Q_norm, Q_norm.transpose(1,2))  # (B, Q, Q)
                eye = torch.eye(query_sim.shape[1], device=device).unsqueeze(0)
                loss_div = (query_sim * (1 - eye)).pow(2).mean()
                loss = loss_atc + 0.5 * loss_lm + 0.1 * loss_div
                total_step_loss = loss / cfg.accumulation_steps
                # print(f"sim shape: {sim.shape}")
                # print(f"targets shape: {targets.shape}, max: {targets.max()}, min: {targets.min()}")
                # print(f"sim min: {sim.min():.4f}, max: {sim.max():.4f}, mean: {sim.mean():.4f}")
            # Backward
            scaler.scale(total_step_loss).backward()
            
            # Optimizer Step
            if (i + 1) % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                for name, param in audio_encoder.named_parameters():
                    if 'qformer' in name and param.grad is not None:
                        print(f"GRAD OK | {name} | norm={param.grad.norm().item():.6f}")
                        break
                else:
                    print("WARNING: no qformer gradients found!")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                total_loss += total_step_loss.item() * cfg.accumulation_steps
                print(f"Epoch {epoch} | Step {i} | Loss: {total_step_loss.item() * cfg.accumulation_steps:.4f}")

        avg_loss = total_loss / (len(loader) // cfg.accumulation_steps * cfg.accumulation_steps)
        print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ---")
        
        # Save Checkpoint (Saving only the Q-Former state dict is usually cleaner)
        torch.save({
            'encoder': audio_encoder.state_dict(),
            'projection': audio_projection.state_dict(),
        }, f"../Model_files/qformer_v2/qformer_aligned_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()