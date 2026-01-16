from __future__ import annotations

import re
from typing import Dict


_FILLERS = [
    r"\bmmm+\b",
    r"\bmmm\b",
    r"\beste\b",
    r"\beh\b",
    r"\bah\b",
    r"\bem\b",
    r"\baja\b",
]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_fillers(text: str) -> str:
    t = text
    for pat in _FILLERS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    return normalize_whitespace(t)


def normalize_units(text: str) -> str:
    """Normaliza unidades comunes sin ser agresivo."""
    t = text
    # mg -> mg, miligramos -> mg
    t = re.sub(r"\bmiligramos?\b", "mg", t, flags=re.IGNORECASE)
    t = re.sub(r"\bgramos?\b", "g", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmililitros?\b", "mL", t, flags=re.IGNORECASE)
    t = re.sub(r"\blitros?\b", "L", t, flags=re.IGNORECASE)
    t = re.sub(r"\bcent[ií]grados?\b", "°C", t, flags=re.IGNORECASE)
    return normalize_whitespace(t)


def normalize_text(text: str) -> str:
    t = normalize_whitespace(text or "")
    t = remove_fillers(t)
    t = normalize_units(t)
    return t


def make_debug_normalization_report(original: str) -> Dict[str, str]:
    return {
        "original": original or "",
        "normalized": normalize_text(original or ""),
    }
