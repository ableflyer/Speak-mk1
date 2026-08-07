#!/usr/bin/env python3
"""
textgrid_utils.py

Minimal, dependency-free parser for Praat "long format" TextGrid files.
Handles the standard IntervalTier structure used by UltraSuite's word_labels
and phone_labels.

USAGE (as a library):
    from textgrid_utils import parse_textgrid, get_tier

    tg = parse_textgrid("01M-001B.TextGrid")
    tier_names = [t["name"] for t in tg["tiers"]]
    phone_tier = get_tier(tg, "phone")   # fuzzy match on tier name
    for start, end, label in phone_tier["intervals"]:
        print(start, end, label)

USAGE (as a script, to inspect a file or entire directory):
    python textgrid_utils.py path/to/file.TextGrid
    python textgrid_utils.py path/to/directory/
"""

import re
import sys
from pathlib import Path


def _parse_long_format(text: str) -> dict:
    """Parse a Praat 'ooTextFile' (long/full) format TextGrid."""
    result = {"xmin": None, "xmax": None, "tiers": []}

    header = text.split("item []")[0]
    m = re.search(r"xmin\s*=\s*([\d.eE+-]+)", header)
    if m:
        result["xmin"] = float(m.group(1))
    m = re.search(r"xmax\s*=\s*([\d.eE+-]+)", header)
    if m:
        result["xmax"] = float(m.group(1))

    tier_blocks = re.split(r"item\s*\[\d+\]\s*:", text)[1:]  # drop preamble

    for block in tier_blocks:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        tier_name = name_match.group(1) if name_match else "unknown"

        is_point_tier = 'class = "TextTier"' in block

        intervals = []
        if is_point_tier:
            for pm in re.finditer(
                r"points?\s*\[\d+\]\s*:\s*number\s*=\s*([\d.eE+-]+)\s*mark\s*=\s*\"([^\"]*)\"",
                block,
            ):
                t = float(pm.group(1))
                mark = pm.group(2)
                intervals.append((t, t, mark))
        else:
            for im in re.finditer(
                r"intervals?\s*\[\d+\]\s*:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*\"([^\"]*)\"",
                block,
            ):
                xmin = float(im.group(1))
                xmax = float(im.group(2))
                label = im.group(3)
                intervals.append((xmin, xmax, label))

        result["tiers"].append({
            "name": tier_name,
            "is_point_tier": is_point_tier,
            "intervals": intervals,
        })

    return result


def _tokenize_short_format(text: str):
    """
    The short format is just one 'value' per line, in a fixed positional order,
    with string values quoted. This yields each line stripped of whitespace,
    with surrounding quotes removed from string tokens (but keeping the raw
    string so we know whether it looked like a string or a number).
    """
    for line in text.splitlines():
        line = line.strip()
        if line == "":
            continue
        yield line


def _unquote(tok: str) -> str:
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1]
    return tok


def _parse_short_format(text: str) -> dict:
    """
    Parse a Praat 'ooTextFile short' format TextGrid. Structure (each item on
    its own line, no field names):

        File type = "ooTextFile short"
        Object class = "TextGrid"
        <blank>
        xmin
        xmax
        <exists>            (literal token, e.g. "<exists>")
        size                (number of tiers)
        for each tier:
            "IntervalTier" | "PointTier" | "TextTier"
            "tier name"
            tier xmin
            tier xmax
            n_intervals (or n_points)
            for each interval: xmin, xmax, "text"     (3 lines)
            for each point:    number, "mark"          (2 lines)
    """
    tokens = list(_tokenize_short_format(text))

    # tokens[0] = 'File type = "ooTextFile short"'
    # tokens[1] = 'Object class = "TextGrid"'
    idx = 2
    result = {"xmin": float(tokens[idx]), "xmax": float(tokens[idx + 1]), "tiers": []}
    idx += 2

    # tokens[idx] == '<exists>' -- skip it
    idx += 1
    n_tiers = int(tokens[idx])
    idx += 1

    for _ in range(n_tiers):
        tier_class = _unquote(tokens[idx]); idx += 1
        tier_name = _unquote(tokens[idx]); idx += 1
        _tier_xmin = float(tokens[idx]); idx += 1
        _tier_xmax = float(tokens[idx]); idx += 1
        n_items = int(tokens[idx]); idx += 1

        is_point_tier = tier_class in ("TextTier", "PointTier")
        intervals = []

        if is_point_tier:
            for _ in range(n_items):
                t = float(tokens[idx]); idx += 1
                mark = _unquote(tokens[idx]); idx += 1
                intervals.append((t, t, mark))
        else:
            for _ in range(n_items):
                xmin = float(tokens[idx]); idx += 1
                xmax = float(tokens[idx]); idx += 1
                label = _unquote(tokens[idx]); idx += 1
                intervals.append((xmin, xmax, label))

        result["tiers"].append({
            "name": tier_name,
            "is_point_tier": is_point_tier,
            "intervals": intervals,
        })

    return result


def parse_textgrid(path: str):
    """
    Parse a Praat TextGrid file, automatically detecting long vs short format.
    Returns: {"xmin": float, "xmax": float, "tiers": [ {"name": str, "is_point_tier": bool, "intervals": [(xmin, xmax, text), ...]} ]}
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "short" in first_line.lower():
        return _parse_short_format(text)
    return _parse_long_format(text)


def get_tier(tg: dict, name_substring: str):
    """Fuzzy (case-insensitive substring) lookup of a tier by name."""
    name_substring = name_substring.lower()
    for tier in tg["tiers"]:
        if name_substring in tier["name"].lower():
            return tier
    return None


def get_unique_labels(tier: dict, include_empty: bool = False) -> list:
    """Extract all unique words/labels from a given tier preserving order of appearance."""
    if not tier or "intervals" not in tier:
        return []

    seen = set()
    unique = []
    for _, _, label in tier["intervals"]:
        clean_label = label.strip()
        if not include_empty and not clean_label:
            continue
        if clean_label not in seen:
            seen.add(clean_label)
            unique.append(clean_label)
    return unique


def frame_labels_from_intervals(intervals, num_frames: int, frame_hop_sec: float, default_label=""):
    """
    Convert a list of (xmin, xmax, label) intervals into a per-frame label array,
    for aligning phone labels to log-mel frames.

    frame i covers time [i * frame_hop_sec, (i+1) * frame_hop_sec)
    """
    labels = [default_label] * num_frames
    for xmin, xmax, label in intervals:
        start_frame = int(xmin / frame_hop_sec)
        end_frame = int(xmax / frame_hop_sec)
        for f in range(max(0, start_frame), min(num_frames, end_frame + 1)):
            labels[f] = label
    return labels


def inspect_single_file(file_path: Path):
    """Inspect and print information for a single TextGrid file."""
    print(f"\n========================================")
    print(f"FILE: {file_path.name}")
    print(f"========================================")

    try:
        tg = parse_textgrid(str(file_path))
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    print(f"xmin={tg['xmin']} xmax={tg['xmax']}")

    for tier in tg["tiers"]:
        kind = "point" if tier["is_point_tier"] else "interval"
        unique_words = get_unique_labels(tier, include_empty=False)

        print(f"\n  Tier: '{tier['name']}' ({kind}, {len(tier['intervals'])} entries)")
        print(f"  Unique words/labels ({len(unique_words)}): {', '.join(unique_words) if unique_words else 'None'}")
        
        print("\n  Sample entries:")
        for xmin, xmax, label in tier["intervals"][:5]:
            print(f"    [{xmin:.3f} - {xmax:.3f}]  '{label}'")
        if len(tier["intervals"]) > 5:
            print(f"    ... ({len(tier['intervals']) - 5} more)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python textgrid_utils.py path/to/file_or_directory")
        sys.exit(1)

    target_path = Path(sys.argv[1])

    if target_path.is_file():
        inspect_single_file(target_path)

    elif target_path.is_dir():
        # Find all .TextGrid (and .lab if applicable) files recursively
        # Matching both uppercase and lowercase extensions
        files = list(target_path.rglob("*.TextGrid")) + list(target_path.rglob("*.textgrid"))
        
        # If your dataset uses .lab extensions for Praat files, uncomment the line below:
        # files.extend(list(target_path.rglob("*.lab")))

        if not files:
            print(f"No TextGrid files found in directory: {target_path}")
            sys.exit(0)

        print(f"Found {len(files)} TextGrid file(s) in '{target_path}'. Processing...\n")

        global_labels_by_tier = {}

        for filepath in sorted(files):
            inspect_single_file(filepath)
            
            # Collect global dataset statistics
            try:
                tg = parse_textgrid(str(filepath))
                for tier in tg["tiers"]:
                    tier_name = tier["name"]
                    if tier_name not in global_labels_by_tier:
                        global_labels_by_tier[tier_name] = set()
                    
                    for label in get_unique_labels(tier, include_empty=False):
                        global_labels_by_tier[tier_name].add(label)
            except Exception:
                continue

        # Print Dataset Summary across all files
        print(f"\n========================================")
        print(f"DATASET SUMMARY ({len(files)} files)")
        print(f"========================================")
        for tier_name, labels in global_labels_by_tier.items():
            print(f"Tier '{tier_name}' total unique labels across dataset ({len(labels)}):")
            print(f"  {', '.join(sorted(labels)) if labels else 'None'}\n")

    else:
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)