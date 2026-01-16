from __future__ import annotations

import re
from typing import Dict, List, Tuple


# Secciones típicas (muy MVP). Puedes ajustar por especialidad.
SECTION_ALIASES = {
    "chief_complaint": [
        "motivo de consulta", "consulta por", "vengo por", "me trae", "me trajo",
    ],
    "hpi": [
        "padecimiento actual", "historia del padecimiento", "evolución", "desde hace",
    ],
    "pmh": [
        "antecedentes", "antecedentes personales", "padecimientos", "enfermedades previas",
    ],
    "allergies": [
        "alergias", "alérgico", "alérgica",
    ],
    "medications": [
        "medicamentos", "tratamiento actual", "toma", "está tomando",
    ],
    "pe": [
        "exploración", "exploracion", "exploración física", "signos vitales",
    ],
    "assessment": [
        "impresión diagnóstica", "diagnóstico", "diagnostico", "probable",
    ],
    "plan": [
        "plan", "tratamiento", "indicaciones", "recomendaciones", "seguimiento",
    ],
}


def _detect_section(sentence: str) -> str | None:
    s = (sentence or "").strip().lower()
    if not s:
        return None

    # Heurística: si contiene trigger fuerte al inicio, cambia sección
    for key, aliases in SECTION_ALIASES.items():
        for a in aliases:
            if s.startswith(a) or re.search(rf"\b{re.escape(a)}\b", s):
                return key
    return None


def split_into_sections(sentences: List[str]) -> Dict[str, str]:
    """Asigna oraciones a secciones con heurística.

    - Si detecta un trigger, cambia la sección actual.
    - Si no detecta nada, usa un "bucket" general.
    """
    current = "general"
    buckets: Dict[str, List[str]] = {"general": []}

    for sent in sentences:
        detected = _detect_section(sent)
        if detected:
            current = detected
            buckets.setdefault(current, [])
            # Si la oración es solo el encabezado, no la copies
            # (si trae contenido, la dejamos)
            if len(sent.split()) <= 3:
                continue

        buckets.setdefault(current, []).append(sent.strip())

    # join
    out: Dict[str, str] = {}
    for k, parts in buckets.items():
        text = " ".join([p for p in parts if p]).strip()
        if text:
            out[k] = text

    return out
