import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import tgt  # pip install tgt
import librosa
import numpy as np


# ── Phoneme → label mappings (ARPAbet, matches SharedPhonologicalLayer heads) ──

VOICING_MAP = {
    # 0 = unvoiced, 1 = voiced
    'AA': 1, 'AE': 1, 'AH': 1, 'AO': 1, 'AW': 1, 'AY': 1,
    'EH': 1, 'ER': 1, 'EY': 1, 'IH': 1, 'IY': 1, 'OW': 1,
    'OY': 1, 'UH': 1, 'UW': 1,
    'B':  1, 'D':  1, 'G':  1, 'V':  1, 'DH': 1, 'Z':  1,
    'ZH': 1, 'M':  1, 'N':  1, 'NG': 1, 'L':  1, 'R':  1,
    'W':  1, 'Y':  1, 'JH': 1,
    'P':  0, 'T':  0, 'K':  0, 'F':  0, 'TH': 0, 'S':  0,
    'SH': 0, 'HH': 0, 'CH': 0,
    'SIL': 0, 'SP': 0, 'spn': 0,
}

MANNER_MAP = {
    # 0=stop 1=fricative 2=affricate 3=nasal 4=liquid 5=glide 6=vowel 7=silence
    'P': 0, 'B': 0, 'T': 0, 'D': 0, 'K': 0, 'G': 0,
    'F': 1, 'V': 1, 'TH': 1, 'DH': 1, 'S': 1, 'Z': 1, 'SH': 1, 'ZH': 1, 'HH': 1,
    'CH': 2, 'JH': 2,
    'M': 3, 'N': 3, 'NG': 3,
    'L': 4, 'R': 4,
    'W': 5, 'Y': 5,
    'AA': 6, 'AE': 6, 'AH': 6, 'AO': 6, 'AW': 6, 'AY': 6,
    'EH': 6, 'ER': 6, 'EY': 6, 'IH': 6, 'IY': 6, 'OW': 6,
    'OY': 6, 'UH': 6, 'UW': 6,
    'SIL': 7, 'SP': 7, 'spn': 7,
}

PLACE_MAP = {
    # 0=bilabial 1=labiodental 2=dental 3=alveolar 4=postalveolar 5=velar 6=glottal 7=vowel/other
    'P': 0, 'B': 0, 'M': 0,
    'F': 1, 'V': 1,
    'TH': 2, 'DH': 2,
    'T': 3, 'D': 3, 'S': 3, 'Z': 3, 'N': 3, 'L': 3, 'R': 3,
    'SH': 4, 'ZH': 4, 'CH': 4, 'JH': 4,
    'K': 5, 'G': 5, 'NG': 5,
    'HH': 6,
    'AA': 7, 'AE': 7, 'AH': 7, 'AO': 7, 'AW': 7, 'AY': 7,
    'EH': 7, 'ER': 7, 'EY': 7, 'IH': 7, 'IY': 7, 'OW': 7,
    'OY': 7, 'UH': 7, 'UW': 7, 'W': 7, 'Y': 7,
    'SIL': 7, 'SP': 7, 'spn': 7,
}

# ARPAbet phones that are considered correct articulation
# (everything except silence/noise tokens)
CORRECT_PHONES = {
    'AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW',
    'B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG',
    'P','R','S','SH','T','TH','V','W','Y','Z','ZH',
}


def normalize_phone(phone: str) -> str:
    """Strip stress markers (AH0 → AH, IH1 → IH) and uppercase."""
    return phone.upper().rstrip('012')


def phone_to_labels(phone: str) -> Dict[str, int]:
    p = normalize_phone(phone)
    return {
        'voicing':     VOICING_MAP.get(p, 0),
        'manner':      MANNER_MAP.get(p, 7),
        'place':       PLACE_MAP.get(p, 7),
        'correctness': 1 if p in CORRECT_PHONES else 0,
    }


# ── Mel spectrogram config (must match AudioEncoderConfig) ──────────────────

N_MELS      = 80
SAMPLE_RATE = 16_000
HOP_LENGTH  = 160
N_FFT       = 400


class LibriSpeechMFADataset(Dataset):
    """
    Args:
        librispeech_root : path to LibriSpeech split, e.g. .../train-clean-100
        mfa_output_root  : path to MFA TextGrid output for the same split
        max_frames       : truncate/pad mel to this many frames (default 1024 ≈ 10s)
    """

    def __init__(
        self,
        librispeech_root: str,
        mfa_output_root: str,
        max_frames: int = 1024,
    ):
        self.librispeech_root = librispeech_root
        self.mfa_output_root  = mfa_output_root
        self.max_frames       = max_frames

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Build index: list of (wav_path, textgrid_path)
        self.samples = self._build_index()
        print(f"LibriSpeechMFADataset: {len(self.samples)} utterances")

    def _build_index(self) -> List[Tuple[str, str]]:
        wav_files = glob.glob(
            os.path.join(self.librispeech_root, "**/*.wav"),
            recursive=True,
        )
        samples = []
        missing = 0
        for wav_path in wav_files:
            # Mirror the directory structure in MFA output
            rel = os.path.relpath(wav_path, self.librispeech_root)
            tg_path = os.path.join(
                self.mfa_output_root,
                rel.replace(".wav", ".TextGrid"),
            )
            if os.path.exists(tg_path):
                samples.append((wav_path, tg_path))
            else:
                missing += 1
        if missing:
            print(f"  Warning: {missing} wav files have no TextGrid, skipping.")
        return samples

    def _load_mel(self, wav_path: str) -> torch.Tensor:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)  # auto-resamples to 16kHz
        waveform = torch.from_numpy(y).unsqueeze(0)      # (1, T)
        mel = self.mel_transform(waveform)               # (1, n_mels, T)
        mel = self.db_transform(mel)
        mel = mel.squeeze(0).T                           # (T, n_mels)
        return mel

    def _load_labels(self, tg_path: str, num_frames: int) -> Dict[str, torch.Tensor]:
        tg = tgt.io.read_textgrid(tg_path)

        # MFA names the tier 'phones' (sometimes 'phone')
        tier_name = 'phones' if 'phones' in tg.get_tier_names() else tg.get_tier_names()[0]
        phone_tier = tg.get_tier_by_name(tier_name)

        voicing     = torch.zeros(num_frames, dtype=torch.long)
        manner      = torch.full((num_frames,), 7, dtype=torch.long)
        place       = torch.full((num_frames,), 7, dtype=torch.long)
        correctness = torch.zeros(num_frames, dtype=torch.long)
        # CTC: blank=0, then ARPAbet index 1..N
        ctc         = torch.zeros(num_frames, dtype=torch.long)

        for interval in phone_tier.intervals:
            if not interval.text or interval.text in ('', 'sp', 'SIL', 'SPN', 'spn'):
                continue
            start_frame = int(interval.start_time * SAMPLE_RATE / HOP_LENGTH)
            end_frame   = int(interval.end_time   * SAMPLE_RATE / HOP_LENGTH)
            end_frame   = min(end_frame, num_frames)

            labels = phone_to_labels(interval.text)
            voicing[start_frame:end_frame]     = labels['voicing']
            manner[start_frame:end_frame]      = labels['manner']
            place[start_frame:end_frame]       = labels['place']
            correctness[start_frame:end_frame] = labels['correctness']

        return {
            'voicing':     voicing,
            'manner':      manner,
            'place':       place,
            'correctness': correctness,
            'ctc':         ctc,          # placeholder — use real CTC loss separately
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        wav_path, tg_path = self.samples[idx]

        mel = self._load_mel(wav_path)          # (T, n_mels)
        T   = mel.shape[0]

        labels = self._load_labels(tg_path, T)

        # Truncate or pad to max_frames
        if T >= self.max_frames:
            mel    = mel[:self.max_frames]
            labels = {k: v[:self.max_frames] for k, v in labels.items()}
        else:
            pad = self.max_frames - T
            mel = F.pad(mel, (0, 0, 0, pad))    # pad time dim
            labels = {k: F.pad(v, (0, pad), value=-100) for k, v in labels.items()}
            for name, v in labels.items():
                bad = ((v != -100) & ((v < 0) | (v >= {"voicing":2,"manner":8,"place":8,"correctness":2,"ctc":40}[name]))).sum()
                if bad > 0:
                    print(f"BAD LABELS in {name}: {bad} values, min={v.min()}, max={v.max()}")

        return mel, labels


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    All samples are already padded to max_frames in __getitem__,
    so this is just a simple stack.
    """
    mels   = torch.stack([b[0] for b in batch])
    labels = {
        k: torch.stack([b[1][k] for b in batch])
        for k in batch[0][1].keys()
    }
    return mels, labels

# should return something like:
# Directory structure expected:
#     librispeech_root/
#         103/1240/103-1240-0000.wav
#         103/1240/103-1240-0000.txt
#         ...
#     mfa_output_root/
#         103/1240/103-1240-0000.TextGrid
#         ...