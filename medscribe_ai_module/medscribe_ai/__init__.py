"""MedScribe AI module (standalone)

Paquete independiente: audio/ASR + NLP clínico + vocabulario/etimologías.
La integración con Flask se hará después.
"""

__all__ = ["process_consultation_audio", "process_transcript_text"]


def process_consultation_audio(*args, **kwargs):
    from .pipelines.consultation import process_consultation_audio as _impl
    return _impl(*args, **kwargs)


def process_transcript_text(*args, **kwargs):
    from .pipelines.consultation import process_transcript_text as _impl
    return _impl(*args, **kwargs)
