from __future__ import annotations

from typing import Any, Dict, Optional


from ..audio.asr_whisper import transcribe as transcribe_audio
from ..nlp.extractor import extract_history
from ..vocab.etymology import highlight_terms


def process_transcript_text(
    text: str,
    *,
    lang: str = "es",
    include_etymology: bool = True,
) -> Dict[str, Any]:
    """
    Procesa un transcript de texto (sin audio) y devuelve:
      - history (JSON clínico)
      - vocab (opcional: etimologías destacadas)
    """
    text = text or ""
    history = extract_history(text, lang=lang)

    out: Dict[str, Any] = {"history": history}
    if include_etymology:
        out["vocab"] = {"etymology_highlights": highlight_terms(text, max_terms=12)}
    return out


def _speaker_share_from_segments(segments: list[dict[str, Any]]) -> Dict[str, float]:
    """Regresa proporción de tiempo por speaker en base a start/end."""
    totals: Dict[str, float] = {}
    total = 0.0
    for s in segments or []:
        spk = s.get("speaker", "speaker_0")
        t0 = float(s.get("start", 0.0) or 0.0)
        t1 = float(s.get("end", 0.0) or 0.0)
        dur = max(0.0, t1 - t0)
        totals[spk] = totals.get(spk, 0.0) + dur
        total += dur
    if total <= 1e-9:
        return {}
    return {k: v / total for k, v in totals.items()}


def process_consultation_audio(
    audio_path: str,
    *,
    lang: str = "es",
    include_etymology: bool = True,
    whisper_kwargs: Optional[Dict[str, Any]] = None,
    diarize: bool = True,
    diar_n_speakers: int = 2,
    # ---- diarization knobs ----
    diar_vad_threshold: float = 0.5,
    diar_min_segment_s: float = 1.0,
    diar_max_segment_s: float = 8.0,
    diar_min_speaker_share: float = 0.10,  # <10% => colapsar
) -> Dict[str, Any]:
    """
    Pipeline end-to-end:
      audio -> (whisper) transcript -> (light diarization) speakers -> (role assignment) doctor/patient
      -> NLP extraction (prefer patient-only if available) -> output JSON
    """
    whisper_kwargs = whisper_kwargs or {}

    # 1) ASR (Whisper)
    transcript = transcribe_audio(audio_path, language=lang, **whisper_kwargs)

    diar_meta: Dict[str, Any] = {
        "enabled": bool(diarize),
        "n_speakers": diar_n_speakers,
        "vad_threshold": diar_vad_threshold,
        "min_segment_s": diar_min_segment_s,
        "max_segment_s": diar_max_segment_s,
        "fallback_applied": False,
        "speaker_share": {},
    }

    # 2) Optional: diarization + speaker assignment + role assignment
    if diarize:
        from ..diarization.light_diarizer import (
            diarize_audio,
            assign_speakers_to_whisper_segments,
        )
        from ..diarization.role_assignment import (
            assign_roles_from_segments,
            add_role_labels,
        )

        diar = diarize_audio(
            audio_path,
            n_speakers=diar_n_speakers,
            vad_threshold=diar_vad_threshold,
            min_segment_s=diar_min_segment_s,
            max_segment_s=diar_max_segment_s,
        )

        transcript["diarization"] = [ds.__dict__ for ds in diar]

        segments = transcript.get("segments", []) or []
        segments = assign_speakers_to_whisper_segments(segments, diar)
        transcript["segments"] = segments

        # fallback: si 2do speaker casi no aporta, colapsa todo a speaker_0
        share = _speaker_share_from_segments(transcript.get("segments", []))
        diar_meta["speaker_share"] = share

        if share:
            sorted_shares = sorted(share.items(), key=lambda kv: kv[1], reverse=True)
            if len(sorted_shares) >= 2:
                second_share = sorted_shares[1][1]
                if second_share < diar_min_speaker_share:
                    for seg in transcript.get("segments", []):
                        seg["speaker"] = "speaker_0"
                    diar_meta["fallback_applied"] = True

        # reconstruye texto por speaker
        transcript["text_by_speaker"] = {}
        for seg in transcript.get("segments", []):
            spk = seg.get("speaker", "speaker_0")
            transcript["text_by_speaker"].setdefault(spk, [])
            t = (seg.get("text") or "").strip()
            if t:
                transcript["text_by_speaker"][spk].append(t)

        transcript["text_by_speaker"] = {
            k: " ".join(v) for k, v in transcript["text_by_speaker"].items()
        }

        # role assignment doctor/patient + label segments
        role_map = assign_roles_from_segments(transcript.get("segments", []))
        transcript["roles"] = role_map
        transcript["segments"] = add_role_labels(transcript.get("segments", []), role_map)

    # siempre agrega metadata
    transcript["diarization_meta"] = diar_meta

    # 3) NLP: prefer patient-only if roles + text_by_speaker available
    roles = transcript.get("roles") or {}
    patient_spk = roles.get("patient")
    tbs = transcript.get("text_by_speaker") or {}

    text_for_nlp = transcript.get("text", "") or ""
    used_patient = False
    if patient_spk and tbs.get(patient_spk):
        text_for_nlp = tbs[patient_spk]
        used_patient = True

    transcript["nlp_source"] = {
        "mode": "patient_only" if used_patient else "full",
        "speaker": patient_spk if used_patient else None,
    }

    history = extract_history(text_for_nlp, lang=lang)

    out: Dict[str, Any] = {"transcript": transcript, "history": history}

    if include_etymology:
        out["vocab"] = {"etymology_highlights": highlight_terms(text_for_nlp, max_terms=12)}

    return out
