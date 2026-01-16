from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import torch


def load_audio_mono_16k(path: str) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    # si no es 16k, re-muestreo simple con torch (sin scipy)
    if sr != 16000:
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,T]
        x = torch.nn.functional.interpolate(
            x, scale_factor=16000 / sr, mode="linear", align_corners=False
        )
        wav = x.squeeze().cpu().numpy()
    return wav.astype(np.float32)


def get_speech_timestamps_silero(
    audio_path: str,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> List[Tuple[float, float]]:
    """
    Devuelve timestamps (start,end) en segundos de segmentos con voz usando Silero VAD.
    """
    wav = load_audio_mono_16k(audio_path)

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    wav_torch = torch.tensor(wav, dtype=torch.float32)
    speech = get_speech_timestamps(
        wav_torch,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    out: List[Tuple[float, float]] = []
    for s in speech:
        out.append((s["start"] / 16000.0, s["end"] / 16000.0))
    return out
