#!/usr/bin/env python3
"""
build_manifest.py

Walks the three UltraSuite audio roots (core-uxtd, core-uxssd, core-upx) and the
shared labels root (labels-uxtd-uxssd-upx), joins each wav file to whatever label
files exist for it, and writes a single manifest CSV.

USAGE (single top-level UltraSuite folder containing core-uxtd/core-uxssd/core-upx
and labels-uxtd-uxssd-upx as subfolders -- this is the common case):
    python build_manifest.py --root /path/to/ultrasuite --out manifest.csv

USAGE (explicit paths, if your folders live in different places or are named
differently than the auto-discovery expects):
    python build_manifest.py \
        --core-uxtd  /path/to/core-uxtd \
        --core-uxssd /path/to/core-uxssd \
        --core-upx   /path/to/core-upx \
        --labels     /path/to/labels-uxtd-uxssd-upx \
        --out        manifest.csv

Any explicit --core-*/--labels flag overrides what --root would have auto-discovered,
so you can mix and match (e.g. --root for everything except one relocated folder).

Each row of the output manifest has:
    corpus, speaker_id, session_id, prompt_type, wav_path, transcription_path,
    phone_textgrid_path, word_textgrid_path, speaker_label_path, slt_label_path,
    usable  (True/False -- False for prompt types E/F, which have no phonetic content)

NOTE: This script does NOT assume the exact internal layout of core-uxtd/uxssd/upx
(speaker-subfoldered or flat) -- it just globs for every *.wav under each root and
extracts (speaker_id, session_id, prompt_type) from the filename / parent dir. If your
actual folder layout differs from what's assumed in `parse_stem`, tell Claude the
real layout and this function is the only thing that needs adjusting.
"""

import argparse
import csv
import re
from pathlib import Path

# Prompt-type letters that have usable phonetic content for the audio encoder.
# E = Non-speech (swallow, cough, etc.), F = Other -- both excluded.
USABLE_PROMPT_TYPES = {"A", "B", "C", "D"}

# A speaker ID looks like '01M', '22M', '10F' -- 1-3 digits + M/F.
SPEAKER_RE = re.compile(r"^\d{1,3}[MF]$")

# Label filenames are always fully-qualified. Two conventions seen in the data:
#   3-part (UXSSD/UPX): '01F-BL1-003A'      -> speaker, session_type, session, task
#   2-part (UXTD):       '01M-001B'          -> speaker, session, task (no session_type)
LABEL_STEM_3PART_RE = re.compile(
    r"^(?P<speaker>\d{1,3}[MF])-(?P<session_type>[A-Za-z]+\d*)-(?P<session>\d+)(?P<task>[A-Za-z])$"
)
LABEL_STEM_2PART_RE = re.compile(
    r"^(?P<speaker>\d{1,3}[MF])-(?P<session>\d+)(?P<task>[A-Za-z])$"
)
# Audio filenames are often bare (speaker/session-type come from folder names instead):
#   '001B' -> session='001', task='B'
BARE_STEM_RE = re.compile(r"^(?P<session>\d+)(?P<task>[A-Za-z])$")


def canonical_key(speaker: str, session_type: str, session: str, task: str) -> str:
    """
    Build a single normalized join key from the four identity components.
    session_type is '' for corpora/files that don't have one (e.g. UXTD),
    so a UXTD file and a UXTD label -- neither of which has a session_type --
    still produce matching keys.
    """
    return f"{speaker.upper()}|{(session_type or '').upper()}|{session}|{task.upper()}"


def parse_label_stem(stem: str):
    """
    Parse a fully-qualified label filename stem into (speaker, session_type, session, task).
    Returns None if it doesn't match either known convention.
    """
    m = LABEL_STEM_3PART_RE.match(stem)
    if m:
        return m.group("speaker"), m.group("session_type"), m.group("session"), m.group("task").upper()

    m = LABEL_STEM_2PART_RE.match(stem)
    if m:
        return m.group("speaker"), "", m.group("session"), m.group("task").upper()

    return None


def parse_audio_identity(wav_path: Path, corpus_root: Path):
    """
    Resolve (speaker, session_type, session, task) for an audio file, which may
    either have a fully-qualified filename (same as labels) or a bare filename
    that relies on its parent folder(s) for speaker / session-type -- and that
    nesting depth differs by corpus (UXTD: speaker/file.wav; UXSSD/UPX:
    speaker/session_type/file.wav). We handle both by walking the path parts
    between corpus_root and the file, and picking out whichever part matches
    the speaker pattern; everything else in between is treated as session_type.
    """
    stem = wav_path.stem

    # Some corpora might name audio files fully-qualified too -- try that first.
    parsed = parse_label_stem(stem)
    if parsed:
        return parsed

    m = BARE_STEM_RE.match(stem)
    if not m:
        return None
    session, task = m.group("session"), m.group("task").upper()

    try:
        rel_parts = wav_path.relative_to(corpus_root).parts[:-1]  # exclude filename itself
    except ValueError:
        rel_parts = wav_path.parent.parts[-2:]  # fallback: last 2 dir names

    speaker = None
    session_type_parts = []
    for part in rel_parts:
        if SPEAKER_RE.match(part):
            speaker = part
        elif speaker is not None:
            # any folder *after* the speaker folder is part of the session type
            # (e.g. 'BL1', 'Mid', 'Post') -- folders *before* it (e.g. a wrapping
            # 'core' directory) are just ignored.
            session_type_parts.append(part)

    if speaker is None:
        return None

    session_type = "-".join(session_type_parts)
    return speaker, session_type, session, task


def _find_subdir(root: Path, name_contains: list, max_depth: int = 2):
    """
    Search under `root` for a directory whose lowercased name contains ALL of the
    strings in `name_contains`. Checks `root` itself, then its immediate children,
    then one level deeper -- covers both a flat layout (root/core-uxtd) and a
    nested one (root/UltraSuite/core-uxtd), without doing a full recursive walk
    of what might be a very large tree.
    """
    def matches(p: Path):
        n = p.name.lower()
        return p.is_dir() and all(s in n for s in name_contains)

    if not root.exists():
        return None
    if matches(root):
        return root

    candidates = [root]
    for _ in range(max_depth):
        next_candidates = []
        for c in candidates:
            if not c.is_dir():
                continue
            for child in c.iterdir():
                if matches(child):
                    return child
                next_candidates.append(child)
        candidates = next_candidates
    return None


def auto_discover(root: Path):
    """
    Given a single UltraSuite root folder, find core-uxtd, core-uxssd, core-upx,
    and the labels folder inside it. Returns a dict with any of these it found;
    missing ones are simply absent from the dict (caller falls back to explicit
    --core-*/--labels args, or errors out if still missing).
    """
    found = {}
    core_targets = {
        "uxtd": ["core", "uxtd"],
        "uxssd": ["core", "uxssd"],
        "upx": ["core", "upx"],
    }
    for corpus_name, needle in core_targets.items():
        hit = _find_subdir(root, needle)
        if hit:
            found[f"core_{corpus_name}"] = hit

    labels_hit = _find_subdir(root, ["label"])
    if labels_hit:
        found["labels"] = labels_hit

    return found


def index_labels(labels_root: Path):
    """
    Walk the labels root once and build a dict keyed by canonical identity key
    (see canonical_key()) -> dict of {label_type: path}. label_type in
    {transcription, phone_tg, word_tg, speaker_label, slt_label}.
    Files whose stem doesn't parse into a known identity are skipped with a
    warning (rather than silently dropped) since that usually signals a naming
    convention this script doesn't know about yet.
    """
    index = {}
    unparsed = []

    def add(key, kind, path):
        index.setdefault(key, {})[kind] = str(path)

    for p in labels_root.rglob("*"):
        if not p.is_file():
            continue
        parts_lower = [x.lower() for x in p.parts]

        kind = None
        if p.suffix == ".txt" and "transcriptions" in parts_lower:
            kind = "transcription"
        elif p.suffix == ".TextGrid" and "word_labels" in parts_lower:
            kind = "word_tg"
        elif p.suffix == ".TextGrid" and "phone_labels" in parts_lower:
            kind = "phone_tg"
        elif "speaker_labels" in parts_lower:
            kind = "speaker_label"
        elif "slt_labels" in parts_lower:
            kind = "slt_label"
        else:
            continue  # not a label type we track (includes .lab HTK files -- redundant with TextGrid)

        identity = parse_label_stem(p.stem)
        if identity is None:
            unparsed.append(str(p))
            continue
        key = canonical_key(*identity)
        add(key, kind, p)

    if unparsed:
        print(f"  WARNING: {len(unparsed)} label files didn't match a known naming convention, e.g.:")
        for u in unparsed[:5]:
            print(f"    {u}")

    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None,
                    help="Single top-level UltraSuite folder containing core-uxtd/uxssd/upx and labels-* as subfolders")
    ap.add_argument("--core-uxtd", type=Path, default=None)
    ap.add_argument("--core-uxssd", type=Path, default=None)
    ap.add_argument("--core-upx", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("manifest.csv"))
    args = ap.parse_args()

    discovered = {}
    if args.root is not None:
        print(f"Auto-discovering subfolders under {args.root} ...")
        discovered = auto_discover(args.root)
        for key, path in discovered.items():
            print(f"  found {key}: {path}")

    # Explicit flags always win over auto-discovery.
    core_uxtd = args.core_uxtd or discovered.get("core_uxtd")
    core_uxssd = args.core_uxssd or discovered.get("core_uxssd")
    core_upx = args.core_upx or discovered.get("core_upx")
    labels_root = args.labels or discovered.get("labels")

    missing = [name for name, val in [
        ("core-uxtd", core_uxtd), ("core-uxssd", core_uxssd),
        ("core-upx", core_upx), ("labels", labels_root),
    ] if val is None]
    if missing:
        print(f"\nERROR: could not resolve: {', '.join(missing)}")
        print("Either fix --root so auto-discovery finds them, or pass the missing "
              "ones explicitly with --core-uxtd / --core-uxssd / --core-upx / --labels.")
        raise SystemExit(1)

    print("Indexing labels tree (one pass)...")
    label_index = index_labels(labels_root)
    print(f"  indexed {len(label_index)} unique stems")

    corpora = {
        "uxtd": core_uxtd,
        "uxssd": core_uxssd,
        "upx": core_upx,
    }

    rows = []
    for corpus_name, root in corpora.items():
        if not root.exists():
            print(f"WARNING: {root} does not exist, skipping {corpus_name}")
            continue
        wavs = list(root.rglob("*.wav"))
        print(f"{corpus_name}: found {len(wavs)} wav files under {root}")
        unresolved = 0
        for wav_path in wavs:
            identity = parse_audio_identity(wav_path, root)
            if identity is None:
                unresolved += 1
                continue
            speaker, session_type, session, task = identity
            key = canonical_key(speaker, session_type, session, task)

            labels = label_index.get(key, {})
            rows.append({
                "corpus": corpus_name,
                "speaker_id": speaker,
                "session_type": session_type,
                "session_id": session,
                "prompt_type": task,
                "wav_path": str(wav_path),
                "transcription_path": labels.get("transcription", ""),
                "phone_textgrid_path": labels.get("phone_tg", ""),
                "word_textgrid_path": labels.get("word_tg", ""),
                "speaker_label_path": labels.get("speaker_label", ""),
                "slt_label_path": labels.get("slt_label", ""),
                "usable": task in USABLE_PROMPT_TYPES,
            })
        if unresolved:
            print(f"  !! {unresolved} wav files in {corpus_name} could not be resolved to an identity -- skipped")

    print(f"Writing {len(rows)} rows to {args.out}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "corpus", "speaker_id", "session_type", "session_id", "prompt_type", "wav_path",
            "transcription_path", "phone_textgrid_path", "word_textgrid_path",
            "speaker_label_path", "slt_label_path", "usable",
        ])
        writer.writeheader()
        writer.writerows(rows)

    usable_count = sum(1 for r in rows if r["usable"])
    have_phone_tg = sum(1 for r in rows if r["phone_textgrid_path"])
    have_speaker_label = sum(1 for r in rows if r["speaker_label_path"])
    have_slt = sum(1 for r in rows if r["slt_label_path"])
    print("\n--- Summary ---")
    print(f"Total wav files:              {len(rows)}")
    print(f"Usable (prompt type A-D):     {usable_count}")
    print(f"Have phone-level TextGrid:    {have_phone_tg}")
    print(f"Have speaker diarization:     {have_speaker_label}")
    print(f"Have SLT correctness labels:  {have_slt}")


if __name__ == "__main__":
    main()