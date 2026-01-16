from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import torch
import soundfile as sf
import librosa


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32), int(sr)


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wav
    x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,T]
    x = torch.nn.functional.interpolate(
        x, scale_factor=16000 / sr, mode="linear", align_corners=False
    )
    return x.squeeze().cpu().numpy().astype(np.float32)


def silero_vad_segments(
    audio_path: str,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> List[Tuple[float, float]]:
    wav, sr = load_audio_mono(audio_path)
    wav = resample_to_16k(wav, sr)

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    speech = get_speech_timestamps(
        torch.tensor(wav, dtype=torch.float32),
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    return [(s["start"] / 16000.0, s["end"] / 16000.0) for s in speech]


def chunk_segments(
    segs: List[Tuple[float, float]],
    *,
    max_len: float = 6.0,
    min_len: float = 1.0,
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in segs:
        dur = b - a
        if dur < min_len:
            continue
        if dur <= max_len:
            out.append((a, b))
        else:
            t = a
            while t < b:
                t2 = min(t + max_len, b)
                if (t2 - t) >= min_len:
                    out.append((t, t2))
                t = t2
    return out


def segment_mfcc_embedding(wav_16k: np.ndarray, start: float, end: float) -> np.ndarray:
    s0 = int(start * 16000)
    s1 = int(end * 16000)
    clip = wav_16k[s0:s1]
    if clip.size < int(0.8 * 16000):
        pad = int(0.8 * 16000) - clip.size
        clip = np.pad(clip, (0, pad))

    # MFCCs (13 coef) + deltas, agregamos estadísticos -> embedding fijo
    mfcc = librosa.feature.mfcc(y=clip, sr=16000, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = np.vstack([mfcc, d1, d2])  # (39, T)
    emb = np.concatenate([feats.mean(axis=1), feats.std(axis=1)], axis=0)  # (78,)
    emb = emb.astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def diarize_audio(
    audio_path: str,
    *,
    n_speakers: int = 2,
    vad_threshold: float = 0.5,
    max_segment_s: float = 6.0,
    min_segment_s: float = 1.0,
) -> List[DiarizationSegment]:
    segs = silero_vad_segments(audio_path, threshold=vad_threshold)
    segs = chunk_segments(segs, max_len=max_segment_s, min_len=min_segment_s)

    if not segs:
        return [DiarizationSegment(0.0, 0.0, "speaker_0")]

    wav, sr = load_audio_mono(audio_path)
    wav_16k = resample_to_16k(wav, sr)

    embs = np.vstack([segment_mfcc_embedding(wav_16k, a, b) for a, b in segs])

    if n_speakers <= 1 or len(segs) == 1:
        labels = np.zeros(len(segs), dtype=int)
    else:
        labels = AgglomerativeClustering(n_clusters=n_speakers).fit_predict(embs)

    diar = [DiarizationSegment(a, b, f"speaker_{int(lbl)}") for (a, b), lbl in zip(segs, labels)]

    diar.sort(key=lambda x: x.start)
    merged: List[DiarizationSegment] = []
    for s in diar:
        if not merged:
            merged.append(s); continue
        last = merged[-1]
        if s.speaker == last.speaker and s.start <= last.end + 0.2:
            merged[-1] = DiarizationSegment(last.start, max(last.end, s.end), last.speaker)
        else:
            merged.append(s)
    return merged


def assign_speakers_to_whisper_segments(
    whisper_segments: List[Dict[str, Any]],
    diar_segments: List[DiarizationSegment],
) -> List[Dict[str, Any]]:
    out = []
    for ws in whisper_segments:
        w0 = float(ws.get("start", 0.0))
        w1 = float(ws.get("end", 0.0))
        best_speaker = "speaker_0"
        best_overlap = 0.0
        for ds in diar_segments:
            overlap = max(0.0, min(w1, ds.end) - max(w0, ds.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds.speaker
        ws2 = dict(ws)
        ws2["speaker"] = best_speaker
        out.append(ws2)
    return out
