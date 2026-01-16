from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
import re

try:
    import spacy  # optional
except ImportError:  # pragma: no cover
    spacy = None

from .normalization import normalize_text
from .sectioning import split_into_sections
from .entities import extract_demographics, extract_keywords, extract_medications, extract_vitals


_NLP_CACHE: dict[str, Any] = {}

# Split robusto (fallback) sin spaCy:
# - corta por ., !, ?, … o saltos de línea
# - evita cortar por abreviaturas comunes
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
_ABBREV_RE = re.compile(
    r"\b(dr|dra|sr|sra|srta|ing|lic|etc|p\.ej|ej|aprox|mg|ml|kg|cm|mmhg)\.$",
    re.IGNORECASE,
)


def get_nlp(lang: str = "es"):
    """NLP liviano: si spaCy está instalado, usa spacy.blank + sentencizer."""
    if spacy is None:
        return None
    key = f"blank_{lang}"
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]

    nlp = spacy.blank(lang)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _NLP_CACHE[key] = nlp
    return nlp


def _to_sentences(text: str, lang: str = "es") -> List[str]:
    # Preferir spaCy si existe
    nlp = get_nlp(lang)
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    # Fallback sin spaCy
    chunks = [c.strip() for c in _SENT_SPLIT_RE.split(text) if c.strip()]
    sentences: List[str] = []
    buffer = ""

    for c in chunks:
        if buffer:
            candidate = buffer + " " + c
        else:
            candidate = c

        # Si termina en abreviatura tipo "Dr." no cortamos todavía
        tail = candidate.split()[-1] if candidate.split() else ""
        if _ABBREV_RE.search(tail):
            buffer = candidate
            continue

        sentences.append(candidate)
        buffer = ""

    if buffer.strip():
        sentences.append(buffer.strip())

    return sentences


def extract_history(text: str, *, lang: str = "es") -> Dict[str, Any]:
    """Convierte texto crudo de consulta en un JSON estructurado (MVP)."""
    normalized = normalize_text(text or "")
    sentences = _to_sentences(normalized, lang=lang)
    sections = split_into_sections(sentences)

    vitals = extract_vitals(normalized)
    meds = extract_medications(normalized)
    demo = extract_demographics(normalized)
    keywords = extract_keywords(normalized)

    out: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat(),
            "lang": lang,
            "version": "mvp-0.1",
            "sentence_splitter": "spacy" if spacy is not None else "regex",
        },
        "text": {"normalized": normalized},
        "sections": sections,
        "entities": {
            "demographics": demo,
            "vitals": vitals,
            "medications": meds,
        },
        "keywords": keywords,
    }

    if "chief_complaint" in sections:
        out["chief_complaint"] = sections["chief_complaint"]
    if "assessment" in sections:
        out["assessment"] = sections["assessment"]
    if "plan" in sections:
        out["plan"] = sections["plan"]

    return out
