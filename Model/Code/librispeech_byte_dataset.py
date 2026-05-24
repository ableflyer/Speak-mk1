import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import librosa
from transformers import AutoTokenizer

N_MELS      = 80
SAMPLE_RATE = 16_000
HOP_LENGTH  = 160
N_FFT       = 400

class LibriSpeechByteDataset(Dataset):
    def __init__(
        self,
        librispeech_root: str,
        max_frames:   int = 512,
        max_text_len: int = 512,
        tokenizer_name: str = "EleutherAI/gpt-neox-20b",
    ):
        self.librispeech_root = librispeech_root
        self.max_frames       = max_frames
        self.max_text_len     = max_text_len

        # Tokeniser — same one your LLM was trained with
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.samples = self._build_index()
        print(f"LibriSpeechByteDataset: {len(self.samples)} samples loaded.")
        print(f"  tokeniser : {tokenizer_name}")
        print(f"  pad_id    : {self.pad_id}")

    def _build_index(self):
        wav_files = glob.glob(
            os.path.join(self.librispeech_root, "**/*.wav"), recursive=True
        )
        samples = []
        for wav_path in wav_files:
            file_id  = os.path.basename(wav_path).replace(".wav", "")
            txt_path = wav_path.replace(".wav", ".txt")  # sibling .txt
            if not os.path.exists(txt_path):
                continue
            samples.append((wav_path, txt_path, file_id))
        return samples


    def _get_token_ids(self, trans_file: str, file_id: str) -> torch.Tensor:
        with open(trans_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            return torch.full((self.max_text_len,), self.pad_id, dtype=torch.long)
        ids = self.tokenizer.encode(
            text,
            max_length     = self.max_text_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        ).squeeze(0)
        return ids
        # # transcript not found — return all-pad tensor
        # return torch.full((self.max_text_len,), self.pad_id, dtype=torch.long)

    def _load_mel(self, wav_path: str) -> torch.Tensor:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        waveform = torch.from_numpy(y).unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel).squeeze(0).T   # (T, n_mels)
        return mel

    def get_pad_id(self) -> int:
        return self.pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, trans_file, file_id = self.samples[idx]

        mel       = self._load_mel(wav_path)
        token_ids = self._get_token_ids(trans_file, file_id)

        # Pad/truncate mel
        T = mel.shape[0]
        if T >= self.max_frames:
            mel = mel[:self.max_frames]
        else:
            mel = F.pad(mel, (0, 0, 0, self.max_frames - T))

        return mel, token_ids


def collate_fn(batch):
    mels      = torch.stack([b[0] for b in batch])
    token_ids = torch.stack([b[1] for b in batch])
    return mels, token_ids