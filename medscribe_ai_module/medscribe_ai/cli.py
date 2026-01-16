from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from .audio.asr_whisper import transcribe as whisper_transcribe
from .audio.preprocess import preprocess_audio
from .pipelines.consultation import process_consultation_audio, process_transcript_text
from .vocab.etymology import explain_term, highlight_terms


def _read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="ignore")


def _dump(obj: Any, out: Optional[str]) -> None:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(s, encoding="utf-8")
    else:
        sys.stdout.write(s + "\n")


def cmd_transcribe(args: argparse.Namespace) -> None:
    result = whisper_transcribe(
        args.audio,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        word_timestamps=args.word_timestamps,
    )
    _dump(result, args.out)


def cmd_extract(args: argparse.Namespace) -> None:
    if args.text is not None:
        text = args.text
    elif args.text_file is not None:
        text = _read_text(args.text_file)
    else:
        text = sys.stdin.read()

    result = process_transcript_text(
        text,
        lang=args.language,
        include_etymology=not args.no_etymology,
    )
    _dump(result, args.out)


def cmd_preprocess_audio(args: argparse.Namespace) -> None:
    out_path = preprocess_audio(
        args.audio,
        args.out,
        target_sr=args.sr,
        mono=not args.stereo,
        normalize=args.normalize,
        target_rms_db=args.target_rms_db,
        target_peak_db=args.target_peak_db,
    )
    # salida minimalista (útil en scripts)
    sys.stdout.write(out_path + "\n")


def cmd_process_audio(args: argparse.Namespace) -> None:
    result = process_consultation_audio(
        args.audio,
        lang=args.language,
        include_etymology=not args.no_etymology,
        whisper_kwargs={
            "task": args.task,
            "beam_size": args.beam_size,
            "vad_filter": not args.no_vad,
            "word_timestamps": args.word_timestamps,
        },
        diarize=not args.no_diarize,
        diar_n_speakers=args.n_speakers,
        diar_vad_threshold=args.diar_vad_threshold,
        diar_min_segment_s=args.diar_min_seg,
        diar_max_segment_s=args.diar_max_seg,
        diar_min_speaker_share=args.diar_min_share,
    )
    _dump(result, args.out)


def cmd_run(args: argparse.Namespace) -> None:
    audio_in = args.audio
    audio_use = audio_in

    if args.preprocess:
        out_wav = args.preprocess_out
        if not out_wav:
            # si --out apunta a JSON, crea un *_clean.wav junto al json
            out_wav = str(Path(args.out).with_suffix("")) + "_clean.wav"

        audio_use = preprocess_audio(
            audio_in,
            out_wav,
            target_sr=args.sr,
            mono=not args.stereo,
            normalize=args.normalize,
            target_rms_db=args.target_rms_db,
            target_peak_db=args.target_peak_db,
        )

    result = process_consultation_audio(
        audio_use,
        lang=args.language,
        include_etymology=not args.no_etymology,
        whisper_kwargs={
            "task": args.task,
            "beam_size": args.beam_size,
            "vad_filter": not args.no_vad,
            "word_timestamps": args.word_timestamps,
        },
        diarize=not args.no_diarize,
        diar_n_speakers=args.n_speakers,
        diar_vad_threshold=args.diar_vad_threshold,
        diar_min_segment_s=args.diar_min_seg,
        diar_max_segment_s=args.diar_max_seg,
        diar_min_speaker_share=args.diar_min_share,
    )
    _dump(result, args.out)


def cmd_etymology(args: argparse.Namespace) -> None:
    _dump(explain_term(args.term), args.out)


def cmd_highlight(args: argparse.Namespace) -> None:
    if args.text is not None:
        text = args.text
    elif args.text_file is not None:
        text = _read_text(args.text_file)
    else:
        text = sys.stdin.read()

    _dump(highlight_terms(text, max_terms=args.max_terms), args.out)


def _add_whisper_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--language", default="es")
    sp.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    sp.add_argument("--beam-size", type=int, default=5)
    sp.add_argument("--no-vad", action="store_true")
    sp.add_argument("--word-timestamps", action="store_true")


def _add_diar_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--no-diarize", action="store_true", help="Disable diarization (light)")
    sp.add_argument("--n-speakers", type=int, default=2, help="Expected number of speakers")
    sp.add_argument("--diar-vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    sp.add_argument("--diar-min-seg", type=float, default=1.0, help="Min speech segment length (s)")
    sp.add_argument("--diar-max-seg", type=float, default=8.0, help="Max speech segment length (s)")
    sp.add_argument("--diar-min-share", type=float, default=0.10, help="Fallback share for 2nd speaker")


def _add_preprocess_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--sr", type=int, default=16000)
    sp.add_argument("--stereo", action="store_true")
    sp.add_argument("--normalize", choices=["loudnorm", "rms", "peak", "none"], default="loudnorm")
    sp.add_argument("--target-rms-db", type=float, default=-20.0)
    sp.add_argument("--target-peak-db", type=float, default=-1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medscribe-ai",
        description="MedScribe AI (audio + NLP) - módulo standalone",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.required = True

    # ---- run (preprocess + process) ----
    sp = sub.add_parser("run", help="preprocess (optional) + ASR + diarization + NLP")
    sp.add_argument("audio", help="Path to audio file")
    sp.add_argument("--out", required=True, help="Output JSON path")
    sp.add_argument("--preprocess", action="store_true", help="Enable preprocessing step")
    sp.add_argument("--preprocess-out", help="Where to write cleaned wav (optional)")
    _add_preprocess_args(sp)
    _add_whisper_args(sp)
    _add_diar_args(sp)
    sp.add_argument("--no-etymology", action="store_true")
    sp.set_defaults(func=cmd_run)

    # ---- preprocess-audio ----
    sp = sub.add_parser("preprocess-audio", help="Preprocess audio (mono/16k + normalize)")
    sp.add_argument("audio", help="Path to input audio (wav/mp3/m4a/...)")
    sp.add_argument("--out", required=True, help="Output WAV path")
    _add_preprocess_args(sp)
    sp.set_defaults(func=cmd_preprocess_audio)

    # ---- transcribe ----
    sp = sub.add_parser("transcribe", help="Transcribe audio with faster-whisper")
    sp.add_argument("audio", help="Path to audio file (wav/mp3/m4a/...)")
    _add_whisper_args(sp)
    sp.add_argument("--out", help="Output JSON path (defaults to stdout)")
    sp.set_defaults(func=cmd_transcribe)

    # ---- extract ----
    sp = sub.add_parser("extract", help="Extract clinical JSON from transcript text")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--text", help="Transcript text")
    g.add_argument("--text-file", help="Path to a UTF-8 .txt file")
    sp.add_argument("--language", default="es")
    sp.add_argument("--no-etymology", action="store_true")
    sp.add_argument("--out", help="Output JSON path (defaults to stdout)")
    sp.set_defaults(func=cmd_extract)

    # ---- process-audio ----
    sp = sub.add_parser("process-audio", help="Audio -> transcript -> clinical JSON")
    sp.add_argument("audio", help="Path to audio file")
    _add_whisper_args(sp)
    _add_diar_args(sp)
    sp.add_argument("--no-etymology", action="store_true")
    sp.add_argument("--out", help="Output JSON path (defaults to stdout)")
    sp.set_defaults(func=cmd_process_audio)

    # ---- etymology ----
    sp = sub.add_parser("etymology", help="Explain a medical term using the mini dictionary")
    sp.add_argument("term")
    sp.add_argument("--out", help="Output JSON path (defaults to stdout)")
    sp.set_defaults(func=cmd_etymology)

    # ---- highlight ----
    sp = sub.add_parser("highlight", help="Highlight terms and suggest etymology")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--text")
    g.add_argument("--text-file")
    sp.add_argument("--max-terms", type=int, default=10)
    sp.add_argument("--out", help="Output JSON path (defaults to stdout)")
    sp.set_defaults(func=cmd_highlight)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
