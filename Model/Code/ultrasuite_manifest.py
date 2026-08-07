"""
ultrasuite_manifest.py

Loads and filters build_manifest.py's manifest.csv down to usable UPX/UXSSD
rows for phoneme-level training, and produces a speaker-independent
train/val/test split.

Confirmed from your actual manifest.csv:
  columns = corpus, speaker_id, session_type, session_id, prompt_type,
            wav_path, transcription_path, phone_textgrid_path,
            word_textgrid_path, speaker_label_path, slt_label_path, usable

  - corpus in {uxtd, uxssd, upx}; this module only keeps {uxssd, upx}.
  - `usable` does NOT imply a phone TextGrid exists (~1,088 UPX rows and
    ~1,098 UXSSD rows are usable=True with an empty phone_textgrid_path --
    verified against your actual file). We therefore filter on
    phone_textgrid_path being non-empty AND (optionally) existing on disk,
    not on `usable` alone.
  - session_type distinguishes baseline/therapy/maintenance sessions
    (BL1, BL2, Mid, Post, Maint, Therapy_01 ... Therapy_16, etc.) -- kept
    as metadata since it's useful for tracking therapy progression, which
    matters for a clinical tool.
"""

import csv
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class ManifestRow:
    corpus: str
    speaker_id: str
    session_type: str
    session_id: str
    prompt_type: str
    wav_path: str
    phone_textgrid_path: str
    word_textgrid_path: str

    @property
    def speaker_key(self) -> str:
        """Speaker IDs are only unique *within* a corpus (e.g. '01M' exists
        in both uxssd and upx as different children), so group by corpus
        too when splitting to avoid leaking a speaker across splits."""
        return f"{self.corpus}:{self.speaker_id}"

    @property
    def utt_id(self) -> str:
        parts = [self.speaker_id]
        if self.session_type:
            parts.append(self.session_type)
        parts.append(self.session_id + self.prompt_type)
        return "-".join(parts)


def load_manifest(
    manifest_csv: str,
    corpora: Sequence[str] = ("upx", "uxssd"),
    require_files_exist: bool = True,
) -> List[ManifestRow]:
    """
    Read manifest.csv and return only rows that:
      1. belong to one of `corpora`
      2. have a non-empty phone_textgrid_path
      3. (if require_files_exist) both the wav and the phone TextGrid
         actually exist on disk right now

    require_files_exist defaults to True because the manifest was built
    against a specific machine's paths (e.g. /home/sakhr/projects/...) --
    if you're running this on a different machine, set it False and fix
    up paths first, or the dataset will be empty.
    """
    rows: List[ManifestRow] = []
    skipped_no_textgrid = 0
    skipped_missing_file = 0

    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["corpus"] not in corpora:
                continue
            if not r["phone_textgrid_path"]:
                skipped_no_textgrid += 1
                continue
            if require_files_exist and not (
                os.path.isfile(r["wav_path"]) and os.path.isfile(r["phone_textgrid_path"])
            ):
                skipped_missing_file += 1
                continue
            rows.append(ManifestRow(
                corpus=r["corpus"],
                speaker_id=r["speaker_id"],
                session_type=r["session_type"],
                session_id=r["session_id"],
                prompt_type=r["prompt_type"],
                wav_path=r["wav_path"],
                phone_textgrid_path=r["phone_textgrid_path"],
                word_textgrid_path=r["word_textgrid_path"],
            ))

    print(
        f"[ultrasuite_manifest] loaded {len(rows)} usable rows from "
        f"{corpora} | skipped {skipped_no_textgrid} (no phone textgrid), "
        f"{skipped_missing_file} (file missing on disk)"
    )
    return rows


def speaker_independent_split(
    rows: List[ManifestRow],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[ManifestRow]]:
    """
    Split by speaker (grouped per-corpus, see ManifestRow.speaker_key) so no
    child appears in more than one split -- this matters for honest
    generalization numbers, since within-speaker leakage would inflate
    validation accuracy.
    """
    by_speaker: Dict[str, List[ManifestRow]] = {}
    for r in rows:
        by_speaker.setdefault(r.speaker_key, []).append(r)

    speakers = sorted(by_speaker.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    val_speakers = set(speakers[:n_val])
    test_speakers = set(speakers[n_val:n_val + n_test])
    train_speakers = set(speakers[n_val + n_test:])

    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        if r.speaker_key in val_speakers:
            splits["val"].append(r)
        elif r.speaker_key in test_speakers:
            splits["test"].append(r)
        else:
            splits["train"].append(r)

    print(
        f"[ultrasuite_manifest] split by {n} unique speakers -> "
        f"train={len(splits['train'])} utts ({n - n_val - n_test} spk), "
        f"val={len(splits['val'])} utts ({n_val} spk), "
        f"test={len(splits['test'])} utts ({n_test} spk)"
    )
    return splits