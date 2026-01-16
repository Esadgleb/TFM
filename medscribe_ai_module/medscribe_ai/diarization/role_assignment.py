from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import re

_DOCTOR_CUES = [
    r"\b¿", r"\bcu[eé]ntame\b", r"\bdesde cu[aá]ndo\b", r"\bduele\b", r"\btoma(s)?\b",
    r"\balergia(s)?\b", r"\bantecedente(s)?\b", r"\bfiebre\b", r"\bn[aá]usea(s)?\b",
    r"\bexploraci[oó]n\b", r"\bdiagn[oó]stico\b", r"\btratamiento\b", r"\brecomendaci[oó]n\b",
    r"\bok\b", r"\baja\b", r"\baj[aá]\b", r"\bmuy bien\b", r"\bperfecto\b",
]

_PATIENT_CUES = [
    r"\bme duele\b", r"\btengo\b", r"\bme ca[ií]\b", r"\bdesde\b", r"\bhace\b",
    r"\bme pas[oó]\b", r"\bme siento\b", r"\bme peg[ué]\b", r"\bno puedo\b",
]

def _count_regex(text: str, patterns: List[str]) -> int:
    t = text.lower()
    return sum(len(re.findall(p, t)) for p in patterns)

def _count_questions(text: str) -> int:
    # signos de interrogación + patrones interrogativos comunes
    t = text.lower()
    return t.count("?") + t.count("¿") + len(re.findall(r"\b(qu[eé]|c[uú]ando|c[oó]mo|d[oó]nde|por qu[eé]|cu[aá]l)\b", t))

def _avg_len(utterances: List[str]) -> float:
    if not utterances:
        return 0.0
    lens = [len(u.split()) for u in utterances if u.strip()]
    return (sum(lens) / max(1, len(lens))) if lens else 0.0

def assign_roles_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Agrupa texto por speaker
    by_spk: Dict[str, List[str]] = {}
    for s in segments:
        spk = s.get("speaker", "speaker_0")
        txt = (s.get("text") or "").strip()
        if txt:
            by_spk.setdefault(spk, []).append(txt)

    speakers = list(by_spk.keys())
    if len(speakers) < 2:
        only = speakers[0] if speakers else "speaker_0"
        return {"doctor": only, "patient": only, "confidence": 0.0, "signals": {"reason": "single_speaker"}}

    # Score heurístico: doctor tiende a preguntar más + tener más “cues” de consulta
    scores: Dict[str, float] = {}
    signals: Dict[str, Any] = {}

    for spk, utts in by_spk.items():
        joined = " ".join(utts)
        q = _count_questions(joined)
        doc_c = _count_regex(joined, _DOCTOR_CUES)
        pat_c = _count_regex(joined, _PATIENT_CUES)
        avgw = _avg_len(utts)

        # score doctor: preguntas + cues doctor - cues paciente - (habla muy larga típica del paciente)
        score = (1.5 * q) + (1.0 * doc_c) - (0.5 * pat_c) - (0.05 * avgw)

        scores[spk] = float(score)
        signals[spk] = {"questions": q, "doctor_cues": doc_c, "patient_cues": pat_c, "avg_words": avgw, "score": score}

    # doctor = mayor score
    doctor = max(scores.items(), key=lambda kv: kv[1])[0]
    patient = [s for s in speakers if s != doctor][0]

    # confidence simple basado en diferencia
    vals = sorted(scores.values(), reverse=True)
    margin = vals[0] - vals[1]
    confidence = max(0.0, min(1.0, 0.5 + (margin / 10.0)))  # escala suave

    return {"doctor": doctor, "patient": patient, "confidence": float(confidence), "signals": signals}

def relabel_segments(segments: List[Dict[str, Any]], role_map: Dict[str, str]) -> List[Dict[str, Any]]:
    doc_spk = role_map.get("doctor")
    pat_spk = role_map.get("patient")
    out = []
    for s in segments:
        spk = s.get("speaker", "speaker_0")
        s2 = dict(s)
        if spk == doc_spk:
            s2["speaker_role"] = "doctor"
        elif spk == pat_spk:
            s2["speaker_role"] = "patient"
        else:
            s2["speaker_role"] = "unknown"
        out.append(s2)
    return out

def add_role_labels(segments: List[Dict[str, Any]], role_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_spk = role_map.get("doctor")
    pat_spk = role_map.get("patient")

    out: List[Dict[str, Any]] = []
    for s in segments or []:
        spk = s.get("speaker", "speaker_0")
        s2 = dict(s)
        if spk == doc_spk:
            s2["speaker_role"] = "doctor"
        elif spk == pat_spk:
            s2["speaker_role"] = "patient"
        else:
            s2["speaker_role"] = "unknown"
        out.append(s2)
    return out

