import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import your models
from audio_encoder import AudioEncoder, AudioEncoderConfig
from librispeech_byte_dataset import LibriSpeechByteDataset, collate_fn

# ─────────────────────────────────────────────────────────────────────────────
# 1. PLACEHOLDER: Your Mamba LLM (Replace with your actual Nemotron/Mamba code)
# ─────────────────────────────────────────────────────────────────────────────
class ByteMambaLM(nn.Module):
    """
    Placeholder for your custom Mamba-based LLM.
    Assumptions:
    - vocab_size: 257 (0=pad, 1-256=bytes)
    - d_model: matches AudioEncoder.llm_dim
    """
    def __init__(self, vocab_size=257, d_model=4096):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        # ... your Mamba layers would go here ...
        # For this placeholder, we use a simple linear layer to simulate the LM Head
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, inputs_embeds=None, labels=None):
        # 1. Pass through Mamba layers (skipped here for brevity)
        # hidden_states = self.mamba_layers(inputs_embeds)
        hidden_states = inputs_embeds # <--- REPLACE THIS WITH YOUR MAMBA LOGIC
        
        # 2. Compute Logits
        logits = self.lm_head(hidden_states)
        
        # 3. Compute Loss
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Ignore padding/audio tokens
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
        return {"loss": loss, "logits": logits}

    def get_input_embeddings(self):
        return self.embed_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class TrainConfig:
    # Data
    librispeech_root = "/path/to/LibriSpeech/train-clean-100"
    max_frames = 1024
    max_text_len = 512
    
    # Model Dimensions (Must match Phase 1)
    d_model = 1024
    llm_dim = 4096
    
    # Training
    batch_size = 8        # Adjust based on GPU VRAM
    epochs = 10
    lr = 1e-4
    accumulation_steps = 4
    
    # Paths
    checkpoint_path = "../Model_files/audio_encoder_epoch_19.pt" # Your Phase 1 weights

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()

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
    audio_cfg = AudioEncoderConfig(d_model=cfg.d_model, llm_dim=cfg.llm_dim)
    audio_encoder = AudioEncoder(audio_cfg).to(device)
    
    # Load Phase 1 Weights
    print(f"Loading Phase 1 weights from {cfg.checkpoint_path}...")
    audio_encoder.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

    # 2. LLM (Placeholder)
    llm = ByteMambaLM(vocab_size=257, d_model=cfg.llm_dim).to(device)
    
    # --- C. FREEZING STRATEGY ---
    
    # 1. Freeze Audio Encoder Backbone
    print("Freezing Audio Encoder Backbone...")
    for name, param in audio_encoder.named_parameters():
        # Only Q-Former trains. Everything else (Mamba, MoE, Stem) is frozen.
        if 'qformer' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"  > Training Q-Former Layer: {name}")

    # 2. Freeze LLM (Standard BLIP-2 approach)
    # If your LLM is not pre-trained, you might want to unfreeze this later 
    # or train the LLM separately first.
    print("Freezing LLM...")
    for param in llm.parameters():
        param.requires_grad = False

    # --- D. Optimizer ---
    # Only pass parameters that require gradients
    trainable_params = [p for p in audio_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    scaler = GradScaler()

    # --- E. Training Loop ---
    print("Starting Phase 2: Q-Former Alignment...")
    
    # Set modes
    audio_encoder.train() # Train mode (BatchNorm/Dropout active in Q-Former)
    llm.eval()            # Eval mode (LLM is frozen)

    for epoch in range(cfg.epochs):
        total_loss = 0
        
        for i, (mel, byte_ids) in enumerate(loader):
            mel = mel.to(device)
            byte_ids = byte_ids.to(device) # (B, Text_Len)

            with autocast(dtype=torch.bfloat16):
                # 1. Get Audio Embeddings (Soft Prompts)
                # Output shape: (B, num_queries, llm_dim)
                audio_embeds, _ = audio_encoder(mel)
                
                # 2. Get Text Embeddings from Frozen LLM
                # Input: byte_ids, Output: (B, Text_Len, llm_dim)
                # Note: We need embeddings, not logits.
                text_embeds = llm.get_input_embeddings()(byte_ids)
                
                # 3. Concatenate [Audio; Text]
                # This creates a single stream for the Mamba LLM
                inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
                
                # 4. Prepare Labels
                # We want the model to predict the text bytes.
                # We ignore the audio tokens in the loss calculation (set to -100).
                batch_size, num_queries, _ = audio_embeds.shape
                
                # Create a tensor of -100s for the audio part
                audio_labels = torch.full(
                    (batch_size, num_queries), -100, 
                    dtype=torch.long, device=device
                )
                
                # Text labels are the byte_ids themselves
                # (CrossEntropyLoss expects indices, not one-hot)
                text_labels = byte_ids
                
                # Combine labels: [-100, -100, ..., byte_id, byte_id, ...]
                labels = torch.cat([audio_labels, text_labels], dim=1)
                
                # 5. Forward Pass through LLM
                outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs['loss']
                
                # Scale loss for accumulation
                loss = loss / cfg.accumulation_steps

            # Backward
            scaler.scale(loss).backward()
            
            # Optimizer Step
            if (i + 1) % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                total_loss += loss.item() * cfg.accumulation_steps
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item() * cfg.accumulation_steps:.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ---")
        
        # Save Checkpoint (Saving only the Q-Former state dict is usually cleaner)
        torch.save(audio_encoder.state_dict(), f"qformer_aligned_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()