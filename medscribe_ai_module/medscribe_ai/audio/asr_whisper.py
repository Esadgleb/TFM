from __future__ import annotations

import os
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class WhisperSettings:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"


_MODEL_CACHE: dict[WhisperSettings, WhisperModel] = {}


def _get_env(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def settings_from_env() -> WhisperSettings:
    """Lee settings desde variables de entorno (para Docker/prod).

    Variables:
      - WHISPER_MODEL_SIZE (tiny/base/small/medium/large-v3...)
      - WHISPER_DEVICE (cpu/cuda)
      - WHISPER_COMPUTE_TYPE (int8/int8_float16/float16...)
    """
    return WhisperSettings(
        model_size=_get_env("WHISPER_MODEL_SIZE", "small"),
        device=_get_env("WHISPER_DEVICE", "cpu"),
        compute_type=_get_env("WHISPER_COMPUTE_TYPE", "int8"),
    )


def get_model(settings: Optional[WhisperSettings] = None) -> WhisperModel:
    s = settings or settings_from_env()
    model = _MODEL_CACHE.get(s)
    if model is None:
        model = WhisperModel(s.model_size, device=s.device, compute_type=s.compute_type)
        _MODEL_CACHE[s] = model
    return model


def ffprobe_duration_seconds(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.STDOUT,
        )
        s = out.decode("utf-8", errors="ignore").strip()
        return float(s) if s else None
    except Exception:
        return None


def transcribe(
    audio_path: str,
    *,
    language: Optional[str] = "es",
    task: str = "transcribe",
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    settings: Optional[WhisperSettings] = None,
) -> Dict[str, Any]:
    """Transcribe un audio con Faster-Whisper y normaliza la salida."""
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    s = settings or settings_from_env()
    model = get_model(s)

    t0 = time.time()
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
    )

    segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for i, seg in enumerate(segments_iter):
        text = (seg.text or "").strip()
        if text:
            full_text_parts.append(text)

        payload: Dict[str, Any] = {
            "id": i,
            "start": float(seg.start or 0.0),
            "end": float(seg.end or 0.0),
            "text": text,
            # diarización real se conecta después (por ahora no inferimos speaker)
            "speaker": "speaker_0",
        }

        if word_timestamps and getattr(seg, "words", None):
            payload["words"] = [
                {
                    "start": float(w.start or 0.0),
                    "end": float(w.end or 0.0),
                    "word": (w.word or "").strip(),
                    "probability": float(getattr(w, "probability", 0.0) or 0.0),
                }
                for w in seg.words
            ]

        segments.append(payload)

    elapsed = time.time() - t0
    duration = ffprobe_duration_seconds(audio_path)

    return {
        "ok": True,
        "model": {
            "provider": "faster-whisper",
            "size": s.model_size,
            "device": s.device,
            "compute_type": s.compute_type,
        },
        "audio": {"path": str(audio_path), "duration_seconds": duration},
        "language": getattr(info, "language", language) or language,
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "text": " ".join(full_text_parts).strip(),
        "segments": segments,
        "timing": {"elapsed_seconds": float(elapsed)},
    }
