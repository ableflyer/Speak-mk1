"""
ultrasuite_dataset.py

torch.utils.data.Dataset for UltraSuite UPX + UXSSD, producing frame-level
phone/voicing/manner/place targets plus an utterance-level phone-id
sequence, built from your real manifest.csv and the actual 48-symbol
Unisyn-SAMPA phone tier.

Usage:
    from ultrasuite_manifest import load_manifest, speaker_independent_split
    from ultrasuite_dataset import UltraSuiteDataset, collate_fn
    from torch.utils.data import DataLoader

    rows = load_manifest("manifest.csv")
    splits = speaker_independent_split(rows)

    train_ds = UltraSuiteDataset(splits["train"])
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,
                           collate_fn=collate_fn, num_workers=4)

KNOWN GAPS (flagging rather than quietly working around):
  1. No correctness/target-vs-produced comparison. UltraSuite's
     "reference phoneme" set (see phoneme_keys.txt) implies a
     canonical/target phone tier used for disorder scoring
     (substitution/omission/distortion), but I don't have that tier's
     name or your doc/*.lex lexicons to build it correctly. This loader
     only surfaces the produced-phone tier (tier_1).
  2. UPX/UXSSD are licensed CC BY-NC 4.0 -- per your own tracking, this
     is pretraining/research use only until the CSTR permission email
     is resolved. Nothing in this file changes that; it's just the data
     loader.
  3. This does not touch ultrasound imaging channels at all (audio + text
     grids only), matching what your manifest actually indexes.
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from ultrasuite_manifest import ManifestRow
from textgrid_parser import parse_textgrid, get_tier
from frame_alignment import build_frame_targets, build_phone_sequence
from sampa_phone_table import IGNORE_INDEX, PHONE_VOCAB, PAD_TOKEN, PHONE2IDX


class UltraSuiteDataset(Dataset):
    def __init__(
        self,
        rows: List[ManifestRow],
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        max_frames: Optional[int] = 1024,
        phone_tier_name: str = "tier_1",
        drop_silence_from_sequence: bool = True,
        strict_phones: bool = False,
        return_waveform: bool = False,
    ):
        """
        max_frames: if set, pad/truncate every item to this length (simple,
            uniform-shape batching, matching your existing pipeline
            convention). Set to None to get variable-length items back and
            use the provided collate_fn, which pads per-batch instead --
            better if your seqlens vary a lot and you don't want to waste
            compute on padding to a fixed global max.
        return_waveform: if True, __getitem__ also returns the raw
            resampled waveform (mono, 16kHz) alongside the mel features,
            in case SpeakMK1's front-end wants raw audio instead of/as
            well as mel.
        """
        self.rows = rows
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.phone_tier_name = phone_tier_name
        self.drop_silence_from_sequence = drop_silence_from_sequence
        self.strict_phones = strict_phones
        self.return_waveform = return_waveform

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono-mix if needed
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self._resamplers[sr](waveform)
        return waveform  # (1, T)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        waveform = self._load_audio(row.wav_path)
        mel = self.db_transform(self.mel_transform(waveform))  # (1, n_mels, T_frames)
        mel = mel.squeeze(0).transpose(0, 1)  # (T_frames, n_mels)
        num_frames = mel.shape[0]

        tiers = parse_textgrid(row.phone_textgrid_path)
        tier = get_tier(tiers, self.phone_tier_name, fallback_contains="phone")
        if tier is None:
            raise ValueError(
                f"No tier named '{self.phone_tier_name}' (or containing "
                f"'phone') found in {row.phone_textgrid_path}. "
                f"Tiers present: {[t.name for t in tiers]}"
            )

        frame_targets = build_frame_targets(
            tier.intervals, num_frames, self.sample_rate, self.hop_length,
            strict_phones=self.strict_phones,
        )
        phone_seq_ids, phone_seq_symbols = build_phone_sequence(
            tier.intervals, drop_silence=self.drop_silence_from_sequence,
            strict_phones=self.strict_phones,
        )

        item = {
            "mel": mel,                                          # (T, n_mels) float32
            "phone_id": torch.tensor(frame_targets["phone_id"], dtype=torch.long),
            "voicing": torch.tensor(frame_targets["voicing"], dtype=torch.long),
            "manner": torch.tensor(frame_targets["manner"], dtype=torch.long),
            "place": torch.tensor(frame_targets["place"], dtype=torch.long),
            "phone_seq": torch.tensor(phone_seq_ids, dtype=torch.long),  # variable length
            "phone_seq_len": len(phone_seq_ids),
            "num_frames": num_frames,
            # Metadata -- useful for per-corpus / per-session-type eval
            # breakdowns (e.g. tracking therapy progression separately
            # from baseline sessions), not used in the loss itself.
            "corpus": row.corpus,
            "speaker_id": row.speaker_id,
            "session_type": row.session_type,
            "utt_id": row.utt_id,
        }
        if self.return_waveform:
            item["waveform"] = waveform.squeeze(0)  # (T_samples,)

        if self.max_frames is not None:
            item = self._pad_or_truncate(item)

        return item

    def _pad_or_truncate(self, item: Dict) -> Dict:
        T = self.max_frames
        n = item["mel"].shape[0]
        if n >= T:
            item["mel"] = item["mel"][:T]
            for k in ("phone_id", "voicing", "manner", "place"):
                item[k] = item[k][:T]
        else:
            pad_len = T - n
            item["mel"] = F.pad(item["mel"], (0, 0, 0, pad_len))
            item["phone_id"] = F.pad(item["phone_id"], (0, pad_len), value=PHONE2IDX[PAD_TOKEN])
            for k in ("voicing", "manner", "place"):
                item[k] = F.pad(item[k], (0, pad_len), value=IGNORE_INDEX)
        item["valid_frames"] = min(n, T)
        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    For use when UltraSuiteDataset(max_frames=None) -- pads every field to
    the batch's own max length instead of a fixed global one.
    """
    max_frames = max(item["mel"].shape[0] for item in batch)
    max_phone_seq = max(item["phone_seq"].shape[0] for item in batch)

    n_mels = batch[0]["mel"].shape[1]
    B = len(batch)

    mel = torch.zeros(B, max_frames, n_mels)
    phone_id = torch.full((B, max_frames), PHONE2IDX[PAD_TOKEN], dtype=torch.long)
    voicing = torch.full((B, max_frames), IGNORE_INDEX, dtype=torch.long)
    manner = torch.full((B, max_frames), IGNORE_INDEX, dtype=torch.long)
    place = torch.full((B, max_frames), IGNORE_INDEX, dtype=torch.long)
    phone_seq = torch.full((B, max_phone_seq), PHONE2IDX[PAD_TOKEN], dtype=torch.long)
    frame_lengths = torch.zeros(B, dtype=torch.long)
    phone_seq_lengths = torch.zeros(B, dtype=torch.long)

    utt_ids, corpora, speaker_ids, session_types = [], [], [], []

    for i, item in enumerate(batch):
        n = item["mel"].shape[0]
        mel[i, :n] = item["mel"]
        phone_id[i, :n] = item["phone_id"]
        voicing[i, :n] = item["voicing"]
        manner[i, :n] = item["manner"]
        place[i, :n] = item["place"]
        frame_lengths[i] = n

        p = item["phone_seq"].shape[0]
        phone_seq[i, :p] = item["phone_seq"]
        phone_seq_lengths[i] = p

        utt_ids.append(item["utt_id"])
        corpora.append(item["corpus"])
        speaker_ids.append(item["speaker_id"])
        session_types.append(item["session_type"])

    return {
        "mel": mel,
        "phone_id": phone_id,
        "voicing": voicing,
        "manner": manner,
        "place": place,
        "phone_seq": phone_seq,
        "frame_lengths": frame_lengths,
        "phone_seq_lengths": phone_seq_lengths,
        "utt_id": utt_ids,
        "corpus": corpora,
        "speaker_id": speaker_ids,
        "session_type": session_types,
    }