import torch
import librosa
import numpy as np
import sounddevice as sd
from audio_encoder import AudioEncoder, AudioEncoderConfig
from audio_trainer import SmallConfig

device = torch.device("cuda")
SAMPLE_RATE = 16000
DURATION = 5

enc_cfg = SmallConfig()
encoder = AudioEncoder(enc_cfg).to(device)
enc_ckpt = torch.load("../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt", map_location=device, weights_only=False)
encoder.load_state_dict(enc_ckpt, strict=False)
encoder.eval()

# Use a synthetic tone instead of recording so we can control input
audio_np = np.sin(2 * np.pi * 440 * np.linspace(0, DURATION, SAMPLE_RATE * DURATION)).astype(np.float32)

mel_np = librosa.feature.melspectrogram(y=audio_np, sr=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80)
mel_np = librosa.power_to_db(mel_np, ref=np.max)
mel = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    feats = encoder.encode_features(mel)

print(f"feats shape: {feats.shape}")
print(f"feats mean:  {feats.mean().item():.6f}")
print(f"feats std:   {feats.std().item():.6f}")
print(f"feats min:   {feats.min().item():.6f}")
print(f"feats max:   {feats.max().item():.6f}")
print(f"feats has nan: {torch.isnan(feats).any().item()}")
print(f"feats has inf: {torch.isinf(feats).any().item()}")

# Now test with pure zeros (silence) - output should be DIFFERENT from above
mel_zero = torch.zeros_like(mel)
with torch.no_grad():
    feats_zero = encoder.encode_features(mel_zero)

print(f"\nzero input mean: {feats_zero.mean().item():.6f}")
print(f"diff from real:  {(feats - feats_zero).abs().mean().item():.6f}")

from train_proj import DirectAudioProjection
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from transformers import AutoTokenizer

proj = DirectAudioProjection(512, 512).to(device)
proj_ckpt = torch.load("../Model_files/audio_proj_training/audio_proj_best.pt", map_location=device, weights_only=False)
proj.load_state_dict(proj_ckpt["audio_proj"])
proj.eval()

with torch.no_grad():
    audio_out = proj(feats)
    audio_out_zero = proj(feats_zero)

print(f"proj out mean:  {audio_out.mean().item():.6f}")
print(f"proj out std:   {audio_out.std().item():.6f}")
print(f"proj diff real vs zero: {(audio_out - audio_out_zero).abs().mean().item():.6f}")

# Now test if the LLM actually sees the audio
llm_cfg = SpeakMK1LLMConfig(vocab_size=50283, d_model=512, d_state=64,
    num_blocks=6, nheads_ssm=8, nheads_attn=8, top_k_audio=32,
    num_experts=4, top_k_experts=2, dropout=0.0, aux_loss_weight=1e-2)
llm = SpeakMK1LLM(llm_cfg).to(device)
llm_ckpt = torch.load("../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt", map_location=device, weights_only=False)
llm.load_state_dict(llm_ckpt["model"], strict=True)
with torch.no_grad():
    for block in llm.blocks:
        if hasattr(block.cross_attn, 'gate'):
            block.cross_attn.gate.data.fill_(1.0)
llm.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
prompt = "<|system|>You are an SLP assistant.<|child|>"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    logits_with, _, _ = llm(input_ids=input_ids, audio_out=audio_out, audio_padding_mask=None)
    logits_without, _, _ = llm(input_ids=input_ids, audio_out=None, audio_padding_mask=None)
    logits_zero, _, _ = llm(input_ids=input_ids, audio_out=audio_out_zero, audio_padding_mask=None)

print(f"\nlogits with audio - without audio: {(logits_with - logits_without).abs().mean().item():.6f}")
print(f"logits real audio - zero audio:    {(logits_with - logits_zero).abs().mean().item():.6f}")
print(f"\ntop5 with audio:    {logits_with[0,-1].topk(5).indices.tolist()}")
print(f"top5 without audio: {logits_without[0,-1].topk(5).indices.tolist()}")
print(f"top5 zero audio:    {logits_zero[0,-1].topk(5).indices.tolist()}")