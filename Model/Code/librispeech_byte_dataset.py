import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import librosa

# ── Mel spectrogram config (must match AudioEncoderConfig) ──────────────────
N_MELS      = 80
SAMPLE_RATE = 16_000
HOP_LENGTH  = 160
N_FFT       = 400

class LibriSpeechByteDataset(Dataset):
    """
    Dataset for Phase 2: Q-Former Alignment.
    
    Returns:
        mel:  (max_frames, n_mels) Audio features.
        bytes: (max_text_len,)      Text represented as UTF-8 bytes (0-255).
        
    Args:
        librispeech_root: Path to LibriSpeech (e.g., train-clean-100)
        max_frames:       Max audio frames (1024 ≈ 10s)
        max_text_len:     Max byte sequence length (bytes take up more space than tokens)
    """
    def __init__(
        self,
        librispeech_root: str,
        max_frames: int = 1024,
        max_text_len: int = 512, # Bytes are longer than tokens, so buffer needs to be bigger
    ):
        self.librispeech_root = librispeech_root
        self.max_frames = max_frames
        self.max_text_len = max_text_len

        # Audio Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Index all .wav files
        self.samples = self._build_index()
        print(f"LibriSpeechByteDataset: {len(self.samples)} samples loaded.")

    def _build_index(self):
        # LibriSpeech structure: speaker/chapter/audio.wav
        wav_files = glob.glob(os.path.join(self.librispeech_root, "**/*.wav"), recursive=True)
        
        # We need to map audio files to their transcript files
        # LibriSpeech usually puts trans in speaker/chapter/trans.txt
        samples = []
        
        # Group by chapter to read transcript files efficiently (optional optimization)
        # For simplicity, we'll search for the text on the fly or cache it here:
        
        for wav_path in wav_files:
            # Determine transcript path
            # e.g., .../103/1240/103-1240-0000.wav -> .../103/1240/103-1240.trans.txt
            chapter_dir = os.path.dirname(wav_path)
            wav_filename = os.path.basename(wav_path)
            file_id = wav_filename.replace(".wav", "")
            
            # Standard LibriSpeech transcript naming
            trans_file = glob.glob(os.path.join(chapter_dir, "*.txt"))
            if not trans_file:
                continue # Skip if no transcript found
                
            # We store (wav_path, trans_file_path, file_id)
            samples.append((wav_path, trans_file[0], file_id))
            
        return samples

    def _get_text_bytes(self, trans_file, file_id):
        """Reads specific line from transcript file and converts to bytes."""
        # This is a bit slow to do every time, but robust.
        # In a production setting, pre-process this into a JSON/CSV.
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(file_id):
                    # Remove file ID and strip whitespace
                    text = line.strip().split(' ', 1)[1]
                    
                    # Convert to bytes
                    raw_bytes = text.encode('utf-8')
                    
                    # Shift bytes to range 1-256 (0 is reserved for padding)
                    # byte values are originally 0-255.
                    # We map 0->1, 1->2 ... 255->256. 
                    # Wait, simpler: map 0->1, 1->2... then 255->256.
                    # Then 0 remains our padding token.
                    byte_list = [b + 1 for b in raw_bytes]
                    
                    return torch.tensor(byte_list, dtype=torch.long)
        
        return torch.tensor([], dtype=torch.long) # Empty if not found

    def _load_mel(self, wav_path):
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        waveform = torch.from_numpy(y).unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel).squeeze(0).T # (T, n_mels)
        return mel

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, trans_file, file_id = self.samples[idx]

        # 1. Load Audio
        mel = self._load_mel(wav_path)
        T = mel.shape[0]

        # 2. Load Bytes
        byte_tensor = self._get_text_bytes(trans_file, file_id)

        # 3. Truncate/Pad Audio
        if T >= self.max_frames:
            mel = mel[:self.max_frames]
        else:
            mel = F.pad(mel, (0, 0, 0, self.max_frames - T))

        # 4. Truncate/Pad Text (Bytes)
        txt_len = byte_tensor.shape[0]
        if txt_len >= self.max_text_len:
            byte_tensor = byte_tensor[:self.max_text_len]
        else:
            byte_tensor = F.pad(byte_tensor, (0, self.max_text_len - txt_len), value=0)

        return mel, byte_tensor

def collate_fn(batch):
    """Collate function for the DataLoader."""
    mels = torch.stack([b[0] for b in batch])
    texts = torch.stack([b[1] for b in batch])
    return mels, texts