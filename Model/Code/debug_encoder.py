import torch
import numpy as np
import librosa
import sys

# copy your model loading code here
from audio_encoder import AudioEncoder
from audio_trainer import SmallConfig

device = torch.device("cuda")
enc_cfg = SmallConfig(d_model=512, llm_dim=4096)
encoder = AudioEncoder(enc_cfg).to(device)
enc_ckpt = torch.load("../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt", map_location=device, weights_only=False)
encoder.load_state_dict(enc_ckpt, strict=False)
encoder.eval()

# make a dummy mel
silent = np.zeros(16000*5, dtype=np.float32)
mel_np = librosa.feature.melspectrogram(y=silent, sr=16000, n_fft=400, hop_length=160, n_mels=80)
mel_np = librosa.power_to_db(mel_np, ref=np.max)
mel = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(device)
print("mel shape:", mel.shape)

with torch.no_grad():
    x = encoder.stem(mel)
    print("stem:", type(x), x.shape if isinstance(x, torch.Tensor) else [t.shape if isinstance(t, torch.Tensor) else type(t) for t in x])
    x = x[0] if isinstance(x, tuple) else x

    for i, layer in enumerate(encoder.uni_mamba_layers):
        x = layer(x)
        print(f"uni_mamba[{i}]:", type(x), x.shape if isinstance(x, torch.Tensor) else [t.shape if isinstance(t, torch.Tensor) else type(t) for t in x])
        x = x[0] if isinstance(x, tuple) else x

    x = encoder.rms_att(x, key_padding_mask=None)
    print("rms_att:", type(x), x.shape if isinstance(x, torch.Tensor) else [t.shape if isinstance(t, torch.Tensor) else type(t) for t in x])
    x = x[0] if isinstance(x, tuple) else x

    for i, layer in enumerate(encoder.bi_mamba_moe_layers):
        x = layer(x)
        print(f"bi_mamba[{i}]:", type(x), x.shape if isinstance(x, torch.Tensor) else [t.shape if isinstance(t, torch.Tensor) else type(t) for t in x])
        x = x[0] if isinstance(x, tuple) else x

    print("final x shape:", x.shape)
    
    phon_result = encoder.phonological(x)
    print("phonological result:", type(phon_result))
    if isinstance(phon_result, tuple):
        for i, t in enumerate(phon_result):
            print(f"  [{i}]:", type(t), t.shape if isinstance(t, torch.Tensor) else t)
    else:
        print("  shape:", phon_result.shape)
        
    phon_out = encoder.phonological(x)[0]
    print("phon_out:", phon_out.shape)

    for name, head in encoder.phonological.heads.items():
        logits = head(phon_out)
        print(f"head '{name}':", logits.shape)
    print(encoder.rms_att)