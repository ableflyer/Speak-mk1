import torch
import sounddevice as sd
import librosa
import numpy as np
from audio_encoder import AudioEncoder, AudioEncoderConfig, DirectProjection
from audio_trainer import SmallConfig
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from transformers import AutoTokenizer
from kokoro_onnx import Kokoro

device = torch.device("cuda")

# load encoder
enc_cfg = SmallConfig(d_model=512, llm_dim=4096)
encoder = AudioEncoder(enc_cfg).to(device)
enc_ckpt = torch.load("../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt", map_location=device, weights_only=False)
print("encoder keys:", enc_ckpt.keys() if isinstance(enc_ckpt, dict) else type(enc_ckpt))
encoder.load_state_dict(enc_ckpt, strict=False)
encoder.eval()

# load projection
from train_proj import DirectAudioProjection
proj = DirectAudioProjection(512, 512).to(device)
proj_ckpt = torch.load("../Model_files/audio_proj_training/audio_proj_best.pt", map_location=device, weights_only=False)
print("proj keys:", proj_ckpt.keys() if isinstance(proj_ckpt, dict) else type(proj_ckpt))
proj.load_state_dict(proj_ckpt["audio_proj"])
proj.eval()

# load LLM
llm_cfg = SpeakMK1LLMConfig(
        vocab_size=50283,
        d_model=512,
        d_state=64,
        # FIXED: Use num_blocks, not num_outer_blocks/num_inner_repeats
        num_blocks=6,
        nheads_ssm=8,
        nheads_attn=8,
        top_k_audio=32,
        num_experts=4,
        top_k_experts=2,
        dropout=0.0,
        aux_loss_weight=1e-2,
    )
llm = SpeakMK1LLM(llm_cfg).to(device)
kokoro = Kokoro("../Model_files/kokoro-onnx-models/kokoro-v1.0.onnx", "../Model_files/kokoro-onnx-models/voices-v1.0.bin")
llm_ckpt = torch.load("../Model_files/checkpoints_v2.3.1/stage5_audio/ckpt_final.pt", map_location=device, weights_only=False)
print("llm keys:", llm_ckpt.keys() if isinstance(llm_ckpt, dict) else type(llm_ckpt))
llm.load_state_dict(llm_ckpt["model"], strict=True)
with torch.no_grad():
    for block in llm.blocks:
        if hasattr(block.cross_attn, 'gate'):
            block.cross_attn.gate.data.fill_(0.3)
llm.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.add_special_tokens({"additional_special_tokens": [
    "<|system|>", "<|child|>", "<|slp|>", "<|think|>", "<|endturn|>"
]})

def generate(llm, tokenizer, input_ids, audio_out, max_new_tokens=100, temperature=0.7, top_p=0.9):
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = llm(input_ids=generated, audio_out=audio_out, audio_padding_mask=None)
            next_logits = logits[0, -1] / temperature
            
            # top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = -float('inf')
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices[torch.multinomial(probs, 1)]
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=False)

def speak(text):
    # af_heart is warm/friendly, good for SLP context
    # other options: af_bella, af_sarah, am_adam
    samples, sample_rate = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
    sd.play(samples, sample_rate)
    sd.wait()

# run
with torch.no_grad():
    # --- Record ---
    SAMPLE_RATE = 16000
    DURATION = 5  # seconds, adjust as needed

    print("Recording... speak now")
    audio_np = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Done recording")

    audio_np = audio_np.squeeze()  # (samples,)

    # --- Mel spectrogram matching your encoder's expected input ---
    mel_np = librosa.feature.melspectrogram(
        y=audio_np,
        sr=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        fmin=0.0,
        fmax=8000.0,
    )
    mel_np = librosa.power_to_db(mel_np, ref=np.max)  # log scale

    # mel_np is (80, T) — transpose to (T, 80), add batch dim
    mel = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 80)

    print(f"Mel shape: {mel.shape}")  # should be (1, ~300, 80) for 3s
    audio_feats = encoder.encode_features(mel)         # (1, T, 512)
    audio_out = proj(audio_feats)                      # (1, T, 512)
    
    prompt = "<|system|>You are a warm, expert AI speech-language pathologist helping a child with articulation errors. Analyze the error and provide encouraging corrective feedback.<|child|> I wanna pway wif my fwiends.<|slp|>"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    logits, _, _ = llm(
        input_ids=input_ids,
        audio_out=audio_out,
        audio_padding_mask=None,
    )
    output = generate(llm, tokenizer, input_ids, audio_out)
    print(output)
    _, _, after_turn = output.partition("<|slp|>")
    speak(after_turn.strip())