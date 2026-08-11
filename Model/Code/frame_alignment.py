"""
frame_alignment.py

Pure-Python (no torch/torchaudio) logic for projecting a phone-tier
TextGrid onto frame-level targets. Kept separate from the Dataset class so
it can be unit-tested without a working audio stack.
"""

from typing import Dict, List, Tuple

from sampa_phone_table import phone_to_features, IGNORE_INDEX, PHONE2IDX, PAD_TOKEN
from textgrid_parser import Interval


def frame_index(t: float, sample_rate: int, hop_length: int) -> int:
    """t (seconds) -> frame index, matching hop_length-sample frame centers."""
    return int(round(t * sample_rate / hop_length))


def build_frame_targets(
    intervals: List[Interval],
    num_frames: int,
    sample_rate: int,
    hop_length: int,
    strict_phones: bool = False,
) -> Dict[str, List[int]]:
    """
    Project TextGrid intervals onto per-frame integer label arrays.

    Returns a dict with keys: phone_id, voicing, manner, place -- each a
    list[int] of length num_frames. Frames not covered by any interval
    (rounding gaps, or audio padding beyond the last interval) are left at
    IGNORE_INDEX (-100) across ALL FOUR fields, matching the convention a
    downstream cross_entropy(..., ignore_index=-100) head expects. Earlier
    versions of this function defaulted phone_id to the vocab's <pad> id
    (0) instead of -100 -- that meant a classification head would get
    real (wrong) supervision on padded frames instead of having them
    excluded, same class of bug as the padding issue on the other three
    fields. Fixed here so phone_id behaves identically to the others.
    """
    phone_id = [IGNORE_INDEX] * num_frames
    voicing = [IGNORE_INDEX] * num_frames
    manner = [IGNORE_INDEX] * num_frames
    place = [IGNORE_INDEX] * num_frames

    for iv in intervals:
        label = iv.text.strip()
        if not label:
            continue
        start = max(0, min(frame_index(iv.xmin, sample_rate, hop_length), num_frames))
        end = max(0, min(frame_index(iv.xmax, sample_rate, hop_length), num_frames))
        if start >= end:
            continue
        feats = phone_to_features(label, strict=strict_phones)
        for i in range(start, end):
            phone_id[i] = feats["phone_id"]
            voicing[i] = feats["voicing"]
            manner[i] = feats["manner"]
            place[i] = feats["place"]

    return {"phone_id": phone_id, "voicing": voicing, "manner": manner, "place": place}


def build_phone_sequence(
    intervals: List[Interval],
    drop_silence: bool = True,
    strict_phones: bool = False,
) -> Tuple[List[int], List[str]]:
    """
    Build the utterance-level phone-id sequence (for CTC / seq2seq targets),
    independent of frame timing. Returns (phone_ids, raw_symbols).

    drop_silence=True (default) removes 'sil'/'spn' segments from the
    sequence, since they're not phonetic content -- but note this means
    the sequence length will NOT match frame-level non-ignored regions
    1:1 if you also drop silence there; keep this in mind if you're
    aligning CTC targets against frame counts.
    """
    ids: List[int] = []
    symbols: List[str] = []
    for iv in intervals:
        label = iv.text.strip()
        if not label:
            continue
        if drop_silence and label in ("sil", "spn"):
            continue
        feats = phone_to_features(label, strict=strict_phones)
        ids.append(feats["phone_id"])
        symbols.append(label)
    return ids, symbols