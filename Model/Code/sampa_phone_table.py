"""
sampa_phone_table.py

Phone inventory and articulatory feature tables for the UltraSuite UPX / UXSSD
phone tiers (tier_1), which use Unisyn-style SAMPA for Scottish Standard
English (this is CSTR Edinburgh data).

Confirmed against the actual label vocab pulled from your manifest
(phoneme_keys.txt): 48 unique symbols shared by both UPX and UXSSD, including
`sil` and `spn` (MFA silence / spoken-noise / unintelligible tokens).

IMPORTANT - things I could NOT verify without your doc/upx.lex, doc/uxssd.lex,
doc/README files (not provided to me), flagged below with UNCERTAIN=True:
    - `5`  : assumed dark/velarized L (allophone of /l/), common in Unisyn sets.
    - `W`  : assumed voiceless labial-velar approximant (Scottish "wh", e.g.
             "which" vs "witch" -- SSE typically keeps this contrast).
    - `a` vs `ae`: assumed `a` = short low central/front vowel (SSE's merged
             TRAP/PALM/BATH vowel), `ae` = a separate slightly different low
             front vowel. SSE vowel systems vary by speaker/age, so this
             split is the shakiest part of the table.
    - syllabic consonants `l= m= n=`: manner/place inherited from the
             corresponding non-syllabic consonant; this is standard practice
             but the SYLLABIC flag matters if your model has a stress/rhythm
             head downstream.

Recommendation: cross-check this table against doc/upx.lex or doc/uxssd.lex
(the pronunciation lexicons) before you rely on `place` labels for anything
you report externally -- given the documented ~40% POA accuracy ceiling,
you don't want a mislabeled ground truth table adding noise on top of a
genuine model limitation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Special (non-phonetic) tokens
# ---------------------------------------------------------------------------
SIL = "sil"
SPN = "spn"  # MFA "spoken noise" -- unintelligible / OOV segment
SPECIAL_TOKENS = {SIL, SPN}

# ---------------------------------------------------------------------------
# Feature label spaces
# ---------------------------------------------------------------------------
VOICING_LABELS = ["voiceless", "voiced"]            # 0, 1
MANNER_LABELS = [
    "vowel", "diphthong", "stop", "affricate", "fricative",
    "nasal", "liquid", "glide", "non_speech",
]
PLACE_LABELS = [
    "bilabial", "labiodental", "dental", "alveolar", "postalveolar",
    "palatal", "velar", "glottal", "labial_velar",
    "central_vowel", "front_vowel", "back_vowel", "n/a",
]

VOICING2IDX = {v: i for i, v in enumerate(VOICING_LABELS)}
MANNER2IDX = {v: i for i, v in enumerate(MANNER_LABELS)}
PLACE2IDX = {v: i for i, v in enumerate(PLACE_LABELS)}

IGNORE_INDEX = -100  # matches torch's default CrossEntropyLoss ignore_index


@dataclass(frozen=True)
class PhoneFeatures:
    symbol: str
    voicing: Optional[str]   # None -> IGNORE_INDEX (sil/spn)
    manner: str
    place: str
    syllabic: bool = False
    uncertain: bool = False  # flagged above: verify against your .lex files


# ---------------------------------------------------------------------------
# The 48-symbol table (order matches phoneme_keys.txt for easy diffing)
# ---------------------------------------------------------------------------
_TABLE: List[PhoneFeatures] = [
    PhoneFeatures("3",   "voiced",    "vowel",      "central_vowel"),           # NURSE
    PhoneFeatures("5",   "voiced",    "liquid",     "alveolar", uncertain=True), # dark/velar L
    PhoneFeatures("@",   "voiced",    "vowel",      "central_vowel"),           # schwa
    PhoneFeatures("@U",  "voiced",    "diphthong",  "back_vowel"),              # GOAT
    PhoneFeatures("D",   "voiced",    "fricative",  "dental"),                  # "th" (this)
    PhoneFeatures("E",   "voiced",    "vowel",      "front_vowel"),             # DRESS
    PhoneFeatures("E@",  "voiced",    "diphthong",  "front_vowel"),             # SQUARE
    PhoneFeatures("I",   "voiced",    "vowel",      "front_vowel"),             # KIT
    PhoneFeatures("I@",  "voiced",    "diphthong",  "front_vowel"),             # NEAR
    PhoneFeatures("N",   "voiced",    "nasal",      "velar"),                  # "ng"
    PhoneFeatures("O",   "voiced",    "vowel",      "back_vowel"),             # LOT/THOUGHT
    PhoneFeatures("OI",  "voiced",    "diphthong",  "back_vowel"),             # CHOICE
    PhoneFeatures("S",   "voiceless", "fricative",  "postalveolar"),           # "sh"
    PhoneFeatures("T",   "voiceless", "fricative",  "dental"),                # "th" (thin)
    PhoneFeatures("U@",  "voiced",    "diphthong",  "back_vowel"),             # CURE
    PhoneFeatures("V",   "voiced",    "vowel",      "central_vowel"),          # STRUT
    PhoneFeatures("W",   "voiceless", "glide",      "labial_velar", uncertain=True),  # "wh"
    PhoneFeatures("Z",   "voiced",    "fricative",  "postalveolar"),           # "zh" (measure)
    PhoneFeatures("a",   "voiced",    "vowel",      "front_vowel", uncertain=True),  # SSE low vowel
    PhoneFeatures("aI",  "voiced",    "diphthong",  "front_vowel"),            # PRICE
    PhoneFeatures("aU",  "voiced",    "diphthong",  "back_vowel"),             # MOUTH
    PhoneFeatures("ae",  "voiced",    "vowel",      "front_vowel", uncertain=True),  # TRAP
    PhoneFeatures("b",   "voiced",    "stop",       "bilabial"),
    PhoneFeatures("d",   "voiced",    "stop",       "alveolar"),
    PhoneFeatures("dZ",  "voiced",    "affricate",  "postalveolar"),           # "j" (judge)
    PhoneFeatures("eI",  "voiced",    "diphthong",  "front_vowel"),            # FACE
    PhoneFeatures("f",   "voiceless", "fricative",  "labiodental"),
    PhoneFeatures("g",   "voiced",    "stop",       "velar"),
    PhoneFeatures("h",   "voiceless", "fricative",  "glottal"),
    PhoneFeatures("i",   "voiced",    "vowel",      "front_vowel"),            # FLEECE
    PhoneFeatures("j",   "voiced",    "glide",      "palatal"),
    PhoneFeatures("k",   "voiceless", "stop",       "velar"),
    PhoneFeatures("l=",  "voiced",    "liquid",     "alveolar", syllabic=True),
    PhoneFeatures("m",   "voiced",    "nasal",      "bilabial"),
    PhoneFeatures("m=",  "voiced",    "nasal",      "bilabial", syllabic=True),
    PhoneFeatures("n",   "voiced",    "nasal",      "alveolar"),
    PhoneFeatures("n=",  "voiced",    "nasal",      "alveolar", syllabic=True),
    PhoneFeatures("p",   "voiceless", "stop",       "bilabial"),
    PhoneFeatures("r",   "voiced",    "liquid",     "alveolar"),               # (or postalveolar approx.)
    PhoneFeatures("s",   "voiceless", "fricative",  "alveolar"),
    PhoneFeatures(SIL,   None,        "non_speech", "n/a"),
    PhoneFeatures(SPN,   None,        "non_speech", "n/a"),
    PhoneFeatures("t",   "voiceless", "stop",       "alveolar"),
    PhoneFeatures("tS",  "voiceless", "affricate",  "postalveolar"),           # "ch"
    PhoneFeatures("u",   "voiced",    "vowel",      "back_vowel"),             # GOOSE
    PhoneFeatures("v",   "voiced",    "fricative",  "labiodental"),
    PhoneFeatures("w",   "voiced",    "glide",      "labial_velar"),
    PhoneFeatures("z",   "voiced",    "fricative",  "alveolar"),
]

PHONE_TABLE: Dict[str, PhoneFeatures] = {p.symbol: p for p in _TABLE}

# Canonical vocabulary ordering (deterministic index assignment for phone_id).
# Index 0 reserved for PAD, so downstream CTC/embedding code can rely on it.
PAD_TOKEN = "<pad>"
PHONE_VOCAB: List[str] = [PAD_TOKEN] + sorted(PHONE_TABLE.keys())
PHONE2IDX: Dict[str, int] = {p: i for i, p in enumerate(PHONE_VOCAB)}
IDX2PHONE: Dict[int, str] = {i: p for p, i in PHONE2IDX.items()}


def get_uncertain_symbols() -> List[str]:
    """Symbols whose feature mapping should be cross-checked against the
    UltraSuite .lex files before being trusted for reporting."""
    return sorted(p.symbol for p in _TABLE if p.uncertain)


def phone_to_features(symbol: str, strict: bool = False) -> Dict[str, int]:
    """
    Map a raw phone-tier label to (phone_id, voicing_id, manner_id, place_id).

    Unknown symbols (e.g. a new SAMPA variant not in this table, or stray
    annotator marks like '*'/'?') are handled explicitly rather than raising
    a silent KeyError:
      - strict=True  -> raises ValueError (use during a one-off vocab audit)
      - strict=False -> returns IGNORE_INDEX for every field except phone_id,
                         which is set to PHONE2IDX['<pad>'] (0), so a bad
                         label degrades to "ignored" rather than crashing
                         a training run or silently colliding with a real
                         phone index.
    """
    feats = PHONE_TABLE.get(symbol)
    if feats is None:
        if strict:
            raise ValueError(
                f"Unknown phone label '{symbol}' -- not in the 48-symbol "
                f"table built from your manifest. If this is a legitimate "
                f"new symbol, add it to sampa_phone_table.PHONE_TABLE first."
            )
        return {
            "phone_id": PHONE2IDX[PAD_TOKEN],
            "voicing": IGNORE_INDEX,
            "manner": IGNORE_INDEX,
            "place": IGNORE_INDEX,
        }

    return {
        "phone_id": PHONE2IDX[symbol],
        "voicing": IGNORE_INDEX if feats.voicing is None else VOICING2IDX[feats.voicing],
        "manner": MANNER2IDX[feats.manner],
        "place": PLACE2IDX[feats.place],
    }


if __name__ == "__main__":
    # Quick self-check: every symbol from your phoneme_keys.txt dump resolves.
    from_manifest = (
        "3 5 @ @U D E E@ I I@ N O OI S T U@ V W Z a aI aU ae b d dZ eI f g h "
        "i j k l= m m= n n= p r s sil spn t tS u v w z"
    ).split()
    assert len(from_manifest) == 48, len(from_manifest)
    missing = [s for s in from_manifest if s not in PHONE_TABLE]
    assert not missing, f"Missing from table: {missing}"
    print(f"All {len(from_manifest)} symbols resolved. Vocab size (incl. pad): {len(PHONE_VOCAB)}")
    print("Uncertain symbols to verify against doc/*.lex:", get_uncertain_symbols())