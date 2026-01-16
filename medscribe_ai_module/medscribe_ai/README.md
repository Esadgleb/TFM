# MedScribe AI (módulo standalone)

Este repositorio contiene el **módulo de IA** del proyecto MedScribe, como un paquete **independiente** (sin Flask).

Incluye:

- **Audio / ASR:** transcripción local con **Whisper** vía `faster-whisper`.
- **NLP clínico (MVP):** normalización, seccionado por reglas y extracción heurística de entidades.
- **Vocabulario / etimologías médicas:** mini diccionario para explicar términos (prefijos/raíces/sufijos) y resaltar palabras “médicas”.
- **Scaffolding opcional para fine-tuning (LoRA):** script de referencia (no requerido para el MVP).

> La integración con Flask/DB/UI se deja **pendiente** a propósito: aquí la meta es poder **probar y evolucionar** el módulo de IA de forma aislada.

---

## Instalación

Requisitos:
- Python 3.10+
- (Recomendado) `ffmpeg`/`ffprobe` instalado en el sistema (para estimar duración y leer formatos comprimidos)

Instala dependencias:

```bash
pip install -r requirements.txt
```

---

## Uso rápido (CLI)

### 1) Transcribir audio

```bash
python -m medscribe_ai transcribe ./audio.m4a --language es --out transcript.json
```

### 2) Extraer JSON clínico desde texto

```bash
python -m medscribe_ai extract --text-file ./transcript.txt --out history.json
```

### 3) Pipeline completo (audio -> transcript -> JSON clínico)

```bash
python -m medscribe_ai process-audio ./audio.m4a --language es --out result.json
```

### 4) Etimología

```bash
python -m medscribe_ai etymology gastritis
python -m medscribe_ai highlight --text "Probable gastritis y neuralgia" --max-terms 10
```

---

## Salida del pipeline

`process-audio` devuelve algo así:

```json
{
  "transcription": {"text": "...", "segments": [...]},
  "history": {
    "sections": {"chief_complaint": "...", "assessment": "...", "plan": "..."},
    "entities": {"demographics": {...}, "vitals": {...}, "medications": [...]},
    "vocab": {"etymology_highlights": [...]}
  }
}
```

---

## Próximos upgrades sugeridos

- **Diarización real** (doctor vs paciente): VAD + embeddings + clustering o `pyannote-audio`.
- **Extracción de medicación** más robusta: diccionario + regex + normalización de unidades.
- **Esquema estable (SOAP / NOM / FHIR-ready):** fijar contrato de JSON para integrar después.
- **Fine-tuning LoRA** para “texto -> JSON clínico” (si decides incorporar LLM local).
