from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


@lru_cache(maxsize=1)
def load_etymology() -> Dict[str, Any]:
    p = Path(__file__).with_name("medical_etymology.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(term: str) -> str:
    t = (term or "").strip().lower()
    # Simplificación de tildes para matching básico
    t = t.translate(str.maketrans({
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u",
    }))
    return t


def explain_term(term: str) -> Dict[str, Any]:
    """Explica un término usando prefijos/raíces/sufijos.

    No pretende ser un diccionario clínico; es un apoyo educativo.
    """
    db = load_etymology()
    t = _normalize(term)

    prefixes = db.get("prefixes", {})
    suffixes = db.get("suffixes", {})
    roots = db.get("roots", {})

    # Soporte de ejemplos explícitos
    examples = db.get("examples", {})
    if t in {_normalize(k) for k in examples.keys()}:
        # buscar la key exacta
        key_exact = next(k for k in examples.keys() if _normalize(k) == t)
        parts = examples[key_exact]
        return {
            "term": term,
            "normalized": t,
            "parts": [
                {"morpheme": p, "meaning": prefixes.get(p) or roots.get(p) or suffixes.get(p)}
                for p in parts
            ],
        }

    matched: List[Dict[str, str]] = []

    # Prefijos más largos primero para evitar matches parciales
    for p in sorted(prefixes.keys(), key=len, reverse=True):
        if t.startswith(_normalize(p)):
            matched.append({"morpheme": p, "meaning": prefixes[p]})
            t = t[len(_normalize(p)) :]
            break

    # Sufijos (también más largos primero)
    suffix_match = None
    for s in sorted(suffixes.keys(), key=len, reverse=True):
        if t.endswith(_normalize(s)) and len(t) > len(_normalize(s)):
            suffix_match = s
            t = t[: -len(_normalize(s))]
            break

    # Raíz (si quedó algo) — buscamos si contiene una raíz conocida
    root_found = None
    for r in sorted(roots.keys(), key=len, reverse=True):
        if _normalize(r) in t and len(t) >= len(_normalize(r)):
            root_found = r
            break

    if root_found:
        matched.append({"morpheme": root_found, "meaning": roots[root_found]})

    if suffix_match:
        matched.append({"morpheme": suffix_match, "meaning": suffixes[suffix_match]})

    return {
        "term": term,
        "normalized": _normalize(term),
        "parts": matched,
    }


def highlight_terms(text: str, *, max_terms: int = 10) -> List[Dict[str, Any]]:
    """Extrae términos potenciales (palabras largas) y sugiere etimología."""
    if not text:
        return []

    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{6,}", text)
    seen = set()
    out: List[Dict[str, Any]] = []
    for w in words:
        key = _normalize(w)
        if key in seen:
            continue
        seen.add(key)

        expl = explain_term(w)
        if expl.get("parts"):
            out.append(expl)
        if len(out) >= max_terms:
            break
    return out
