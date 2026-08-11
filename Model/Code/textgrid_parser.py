"""
textgrid_parser.py

Minimal, dependency-free parser for Praat "long format" TextGrid files
(the format Montreal Forced Aligner writes, which is what UltraSuite's
phone_labels/TG and word_labels/TG directories contain).

NOTE: your project already has `textgrid_utils.py` for this. This module is
intentionally self-contained (no import of your existing utils, since I
don't have its function signatures) so the dataset loader below runs
standalone. If you'd rather wire this to your existing parse_textgrid() /
get_tier(), swap out `load_textgrid_tier()` in ultrasuite_dataset.py --
that's the only place this module is called from.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


@dataclass
class Tier:
    name: str
    xmin: float
    xmax: float
    intervals: List[Interval]


_ITEM_RE = re.compile(r'item \[\d+\]:')
_KV_RE = re.compile(r'(\w+)\s*=\s*"?([^"\n]*)"?\s*$')


def _parse_kv(line: str) -> Optional[tuple]:
    m = _KV_RE.search(line.strip())
    if not m:
        return None
    return m.group(1), m.group(2).strip()

def _unquote(s):
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def _parse_short(lines):
    # remove blank lines
    lines = [l.strip() for l in lines if l.strip()]

    i = 2  # skip header

    xmin = float(lines[i]); i += 1
    xmax = float(lines[i]); i += 1

    assert lines[i] == "<exists>"
    i += 1

    n_tiers = int(lines[i])
    i += 1

    tiers = []

    for _ in range(n_tiers):
        tier_type = _unquote(lines[i]); i += 1
        tier_name = _unquote(lines[i]); i += 1

        tier_xmin = float(lines[i]); i += 1
        tier_xmax = float(lines[i]); i += 1

        n_intervals = int(lines[i]); i += 1

        intervals = []

        for _ in range(n_intervals):
            ixmin = float(lines[i]); i += 1
            ixmax = float(lines[i]); i += 1
            text = _unquote(lines[i]); i += 1

            intervals.append(
                Interval(ixmin, ixmax, text)
            )

        tiers.append(
            Tier(
                name=tier_name,
                xmin=tier_xmin,
                xmax=tier_xmax,
                intervals=intervals,
            )
        )

    return tiers


def parse_textgrid(path: str) -> List[Tier]:
    """
    Parse a Praat long-format TextGrid into a list of Tier objects.
    Raises ValueError on files that don't look like long-format TextGrids
    (short-format is not handled -- MFA output is always long-format, but
    if you hand-edited a grid in Praat with "short text file" export this
    will fail loudly rather than silently returning nothing).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
        
    lines = text.splitlines()

    if "ooTextFile" not in text:
        raise ValueError(f"{path} does not look like a Praat TextGrid file")
    if 'File type = "ooTextFile short"' in text:
        return _parse_short(lines)

    tiers: List[Tier] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _ITEM_RE.search(line):
            # Walk forward to collect this tier's header + intervals.
            tier_name = None
            tier_xmin = None
            tier_xmax = None
            intervals: List[Interval] = []
            j = i + 1
            cur_xmin = cur_xmax = None
            cur_text = None
            while j < n:
                l2 = lines[j]
                if _ITEM_RE.search(l2):
                    break  # next tier starts
                kv = _parse_kv(l2)
                if kv:
                    key, val = kv
                    if key == "name":
                        tier_name = val
                    elif key == "xmin" and tier_xmin is None:
                        tier_xmin = float(val)
                    elif key == "xmax" and tier_xmax is None:
                        tier_xmax = float(val)
                    elif key == "xmin":
                        cur_xmin = float(val)
                    elif key == "xmax":
                        cur_xmax = float(val)
                    elif key == "text":
                        cur_text = val
                        if cur_xmin is not None and cur_xmax is not None:
                            intervals.append(Interval(cur_xmin, cur_xmax, cur_text))
                            cur_xmin = cur_xmax = cur_text = None
                j += 1
            if tier_name is not None:
                tiers.append(Tier(
                    name=tier_name,
                    xmin=tier_xmin if tier_xmin is not None else 0.0,
                    xmax=tier_xmax if tier_xmax is not None else 0.0,
                    intervals=intervals,
                ))
            i = j
        else:
            i += 1
    return tiers


def get_tier(tiers: List[Tier], name: str, fallback_contains: Optional[str] = None) -> Optional[Tier]:
    """
    Find a tier by exact name (case-insensitive), falling back to a
    substring search (e.g. fallback_contains="phone") if no exact match.
    Returns None if nothing matches -- callers should decide how to handle
    a genuinely absent tier rather than this module guessing for them.
    """
    name_l = name.lower()
    for t in tiers:
        if t.name.lower() == name_l:
            return t
    if fallback_contains:
        fc = fallback_contains.lower()
        for t in tiers:
            if fc in t.name.lower():
                return t
    return None