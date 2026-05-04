# qformer_tester.py
import torch
import torch.nn.functional as F
from audio_encoder import AudioEncoder
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from audio_trainer import SmallConfig
from librispeech_byte_dataset import LibriSpeechByteDataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = LibriSpeechByteDataset(
    librispeech_root="../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100",
    max_frames=1024,
    max_text_len=512
)
loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# ── load models ──
audio_cfg = SmallConfig(d_model=512, llm_dim=4096)
audio_encoder = AudioEncoder(audio_cfg).to(device)
audio_ckpt = torch.load("../Model_files/qformer_v3.1/qformer_aligned_epoch_0.pt", map_location=device)
audio_encoder.load_state_dict(audio_ckpt['encoder'])
audio_encoder.eval()
qformer_params = {k:v for k,v in audio_encoder.named_parameters() 
                  if 'qformer' in k}
print(list(qformer_params.keys())[:5])  # Do keys exist?
print(next(iter(qformer_params.values())).mean())

model_cfg = SpeakMK1LLMConfig(
    vocab_size=50283, d_model=512, d_state=64, num_blocks=6,
    nheads_ssm=8, nheads_attn=8, top_k_audio=32,
    num_experts=4, top_k_experts=2, dropout=0.0, aux_loss_weight=1e-2,
)
llm = SpeakMK1LLM(model_cfg).to(device)
ckpt = torch.load("../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt", map_location=device, weights_only=False)
llm.load_state_dict(ckpt['model'])
llm.eval()

audio_projection = nn.Sequential(
    nn.Linear(4096, 512), nn.LayerNorm(512), nn.GELU(),
    nn.Dropout(0.1), nn.Linear(512, 512)
).to(device)
# load projection weights if saved separately
# audio_projection.load_state_dict(torch.load("projection.pt"))
audio_projection.load_state_dict(audio_ckpt['projection'])
print("=== QFORMER TESTER ===\n")
mel, byte_ids = next(iter(loader))
mel = mel.to(device)
byte_ids = byte_ids.to(device)
with torch.no_grad():
    # ── test 1: retrieval accuracy ──
    # make fake batch of 8 different mel inputs
    # B = 8
    # mels = torch.randn(B, 200, 80).to(device)  # different audio
    
    # # fake different text
    # text_ids = torch.randint(0, 50277, (B, 64)).to(device)
    
    audio_out, _ = audio_encoder(mel)
    audio_embeds = audio_projection(audio_out)
    text_embeds = llm.embed_proj(llm.embedding(byte_ids))
    
    audio_norm = F.normalize(audio_embeds.mean(dim=1), dim=-1)
    text_norm = F.normalize(text_embeds.mean(dim=1).detach(), dim=-1)
    print(f"byte_ids shape: {byte_ids.shape}")
    print(f"text_embeds shape: {text_embeds.shape}")
    print(f"audio_norm shape: {audio_norm.shape}")
    print(f"llm embedding weight shape: {llm.embedding.weight.shape}")
    
    sim = audio_norm @ text_norm.T
    retrieval_acc = (sim.argmax(dim=1) == torch.arange(8, device=device)).float().mean()
    
    # ── test 2: collapse detection ──
    off_diag = sim[~torch.eye(8, device=device).bool()]
    
    # ── test 3: embedding diversity ──
    audio_std = audio_embeds.std(dim=0).mean()
    
    print(f"Retrieval accuracy : {retrieval_acc:.3f}  (random = {1/8:.3f})")
    print(f"Off-diag sim mean  : {off_diag.mean():.4f}  (collapsed if > 0.8)")
    print(f"Off-diag sim max   : {off_diag.max():.4f}")
    print(f"Audio embed std    : {audio_std:.4f}  (dead if < 0.01)")
    print()
    
    if retrieval_acc > 0.3:
        print("GOOD: qformer learning alignment!")
    elif retrieval_acc > 0.125:
        print("OK: slightly above random, needs more training")
    else:
        print("BAD: random chance, qformer not learning")
        
    if off_diag.mean() > 0.8:
        print("BAD: collapsed embeddings!")
    else:
        print("GOOD: embeddings diverse!")
    
    print(f"Sim matrix:\n{sim}")
    print(f"Diagonal (correct pairs): {sim.diagonal()}")
    print(f"Audio embeds sample: {audio_norm[0, :5]}")
    print(f"Text embeds sample: {text_norm[0, :5]}")
    