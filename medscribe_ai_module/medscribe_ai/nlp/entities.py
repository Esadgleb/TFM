from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Regex helpers
# -------------------------

def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def extract_vitals(text: str) -> Dict[str, Any]:
    """Extrae signos vitales básicos desde texto en español.

    No pretende cubrir TODO; se enfoca en lo más frecuente.
    """
    t = text or ""
    vitals: Dict[str, Any] = {}

    # TA / presión arterial: 120/80
    m = re.search(r"\b(?:ta|t\.a\.|presi[oó]n(?:\s+arterial)?)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", t, flags=re.IGNORECASE)
    if m:
        vitals["blood_pressure"] = {"systolic": int(m.group(1)), "diastolic": int(m.group(2)), "unit": "mmHg"}

    # FC / pulso: 70 lpm
    m = re.search(r"\b(?:fc|f\.c\.|frecuencia\s+card[ií]aca|pulso)\s*[:=]?\s*(\d{2,3})\b(?:\s*(?:lpm|lat\/min))?", t, flags=re.IGNORECASE)
    if m:
        vitals["heart_rate"] = {"value": int(m.group(1)), "unit": "bpm"}

    # FR: 18
    m = re.search(r"\b(?:fr|f\.r\.|frecuencia\s+respiratoria)\s*[:=]?\s*(\d{1,2})\b", t, flags=re.IGNORECASE)
    if m:
        vitals["resp_rate"] = {"value": int(m.group(1)), "unit": "rpm"}

    # Temp: 37.2
    m = re.search(r"\b(?:temp(?:eratura)?|t\b)\s*[:=]?\s*(\d{2}(?:[\.,]\d)?)\s*(?:°c|c)?\b", t, flags=re.IGNORECASE)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            vitals["temperature"] = {"value": val, "unit": "°C"}

    # SatO2: 98%
    m = re.search(r"\b(?:sato2|sat\s*o2|saturaci[oó]n(?:\s+de\s+ox[ií]geno)?)\s*[:=]?\s*(\d{2,3})\s*%\b", t, flags=re.IGNORECASE)
    if m:
        vitals["spo2"] = {"value": int(m.group(1)), "unit": "%"}

    # Peso: 70 kg
    m = re.search(r"\b(?:peso)\s*[:=]?\s*(\d{2,3}(?:[\.,]\d)?)\s*(?:kg|kilos?)\b", t, flags=re.IGNORECASE)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            vitals["weight"] = {"value": val, "unit": "kg"}

    # Talla/altura: 1.70 m o 170 cm
    m = re.search(r"\b(?:talla|altura|estatura)\s*[:=]?\s*(\d{1,3}(?:[\.,]\d)?)\s*(m|cm)\b", t, flags=re.IGNORECASE)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            unit = m.group(2).lower()
            if unit == "cm":
                vitals["height"] = {"value": val, "unit": "cm"}
            else:
                vitals["height"] = {"value": val, "unit": "m"}

    return vitals


_MED_FREQ_PAT = r"(?:cada\s+\d+\s*(?:h|horas?)|\d+\s*veces\s+al\s+d[ií]a|\b(?:c\/|q)\d+h\b|\b(?:bid|tid|qid|qd|qhs)\b)"


def extract_medications(text: str) -> List[Dict[str, Any]]:
    """Extrae menciones de medicamentos (heurístico).

    Busca patrones tipo:
      - paracetamol 500 mg cada 8 horas
      - ibuprofeno 400mg c/8h

    Nota: sin diccionario farmacológico todavía.
    """
    t = text or ""
    meds: List[Dict[str, Any]] = []

    # token medicamento simple: palabra(s) + dosis
    # Mantenerlo conservador para evitar falsos positivos
    pat = re.compile(
        rf"\b([a-záéíóúñ][a-záéíóúñ\-]+(?:\s+[a-záéíóúñ][a-záéíóúñ\-]+){{0,2}})\s+(\d{{1,4}}(?:[\.,]\d)?)\s*(mg|g|mcg|ug|mL|ml)\b(?:\s*(.*?))?(?=\.|,|;|\n|$)",
        flags=re.IGNORECASE,
    )

    for m in pat.finditer(t):
        name = m.group(1).strip()
        dose = _to_float(m.group(2))
        unit = (m.group(3) or "").strip()
        tail = (m.group(4) or "").strip()

        freq = None
        m_freq = re.search(_MED_FREQ_PAT, tail, flags=re.IGNORECASE)
        if m_freq:
            freq = m_freq.group(0).strip()

        meds.append({
            "name": name,
            "dose": {"value": dose, "unit": unit} if dose is not None else None,
            "frequency": freq,
            "raw": m.group(0).strip(),
        })

    # dedupe por raw
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in meds:
        r = it.get("raw")
        if r and r.lower() in seen:
            continue
        if r:
            seen.add(r.lower())
        out.append(it)
    return out


def extract_demographics(text: str) -> Dict[str, Any]:
    t = text or ""
    out: Dict[str, Any] = {}

    # Edad: "tengo 45 años" / "edad 45"
    m = re.search(r"\b(?:edad\s*[:=]?\s*|tengo\s+)(\d{1,3})\s*a[nñ]os?\b", t, flags=re.IGNORECASE)
    if m:
        try:
            out["age"] = int(m.group(1))
        except Exception:
            pass

    # Sexo: masculino/femenino, hombre/mujer
    m = re.search(r"\b(sexo\s*[:=]?\s*)(masculino|femenino|hombre|mujer)\b", t, flags=re.IGNORECASE)
    if m:
        val = m.group(2).lower()
        out["sex"] = "M" if val in ("masculino", "hombre") else "F"

    return out


def extract_keywords(text: str) -> List[str]:
    """Keywords rápidas (muy simple) para búsqueda/UX."""
    t = (text or "").lower()
    candidates = [
        "dolor", "fiebre", "tos", "diarrea", "náusea", "vomito", "vómito",
        "cefalea", "mareo", "fatiga", "disnea", "alergia", "hipertensión",
        "diabetes", "asma",
    ]
    found = [c for c in candidates if c in t]
    # normalize accents duplicates
    uniq: List[str] = []
    for f in found:
        if f not in uniq:
            uniq.append(f)
    return uniq
