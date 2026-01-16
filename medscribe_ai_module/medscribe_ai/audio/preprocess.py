from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import librosa


NormalizeMode = Literal["loudnorm", "rms", "peak", "none"]


def _db_to_amp(db: float) -> float:
    return 10 ** (db / 20.0)


def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)) + 1e-12)


def _ensure_parent(out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def preprocess_audio(
    in_path: str,
    out_path: str,
    *,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: NormalizeMode = "loudnorm",
    # para loudnorm:
    loudnorm_I: float = -16.0,   # LUFS
    loudnorm_TP: float = -1.5,   # True Peak dB
    loudnorm_LRA: float = 11.0,  # Loudness range
    # para normalización python:
    target_rms_db: float = -20.0,    # dBFS aprox
    target_peak_db: float = -1.0,    # dBFS
) -> str:
    """
    Preprocesa audio a WAV (PCM 16-bit), resample a target_sr, mono opcional,
    y normaliza volumen.

    - normalize="loudnorm": requiere ffmpeg (mejor opción)
    - normalize="rms" / "peak": puro Python (sin ffmpeg)
    - normalize="none": solo convierte/resamplea

    Devuelve out_path.
    """
    if not in_path or not Path(in_path).exists():
        raise FileNotFoundError(f"Audio file not found: {in_path}")

    _ensure_parent(out_path)

    # 1) Preferimos ffmpeg loudnorm si está disponible y se pidió.
    if normalize == "loudnorm" and ffmpeg_available():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-ac", "1" if mono else "2",
            "-ar", str(target_sr),
            "-af", f"loudnorm=I={loudnorm_I}:TP={loudnorm_TP}:LRA={loudnorm_LRA}",
            "-c:a", "pcm_s16le",
            str(out_path),
        ]
        subprocess.check_call(cmd)
        return out_path

    # 2) Fallback Python (sin ffmpeg) — carga, mono, resample
    wav, sr = sf.read(str(in_path), always_2d=True)
    wav = wav.astype(np.float32)

    if mono:
        wav = np.mean(wav, axis=1)
    else:
        wav = wav  # (N,2)

    if mono:
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    else:
        # resample canal por canal
        if sr != target_sr:
            ch0 = librosa.resample(wav[:, 0], orig_sr=sr, target_sr=target_sr).astype(np.float32)
            ch1 = librosa.resample(wav[:, 1], orig_sr=sr, target_sr=target_sr).astype(np.float32)
            wav = np.stack([ch0, ch1], axis=1)

    # 3) Normalización Python
    if normalize == "rms":
        tgt = _db_to_amp(target_rms_db)
        cur = _rms(wav if mono else wav.reshape(-1))
        gain = tgt / max(cur, 1e-12)
        wav = wav * gain

    elif normalize == "peak":
        tgt = _db_to_amp(target_peak_db)
        cur = _peak(wav if mono else wav.reshape(-1))
        gain = tgt / max(cur, 1e-12)
        wav = wav * gain

    # 4) Limit/clamp (evita clipping duro)
    wav = np.clip(wav, -1.0, 1.0)

    # 5) Guarda WAV PCM_16
    sf.write(str(out_path), wav, target_sr, subtype="PCM_16")
    return out_path
