import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# 1. Load the "Ear" (Pre-trained on 53 languages for high phonetic accuracy)
model_name = "facebook/wav2vec2-large-xlsr-53"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

def get_speech_features(audio_file):
    # 2. Load audio (Wav2Vec2 REQUIRES 16kHz)
    speech, sr = librosa.load(audio_file, sr=16000)
    
    # 3. Pre-process (Convert to tensors)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    
    # 4. Extract Features
    with torch.no_grad():
        outputs = model(input_values)
    
    # This is the "Acoustic Fingerprint" [Batch, Time, Hidden_Size]
    features = outputs.last_hidden_state.squeeze(0).numpy()
    return features

# Example Usage:
# features_correct = get_speech_features("snake_correct.wav")
# features_error = get_speech_features("thnake_error.wav")