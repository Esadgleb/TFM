# MedScribe AI Module

Paquete **standalone** (sin integración a Flask) que implementa el **módulo de IA** de MedScribe:

- **ASR local** con Whisper via `faster-whisper`.
- **NLP clínico MVP** (reglas + extracción heurística) para producir un JSON estructurado.
- **Vocabulario / etimologías médicas** (mini diccionario) para UX/explicaciones.
- **Opcional**: scaffolding para fine-tuning (LoRA) si más adelante quieres un LLM local.

## Estructura

- `medscribe_ai/audio/` → ASR (Whisper)
- `medscribe_ai/nlp/` → normalización, secciones, entidades, extractor
- `medscribe_ai/vocab/` → etimologías
- `medscribe_ai/pipelines/` → pipelines end-to-end
- `medscribe_ai/llm/` → fine-tuning (opcional)

## Instalación

```bash
pip install -r requirements.txt
```

(Para LoRA)

```bash
pip install -r requirements-llm.txt
```

## CLI

```bash
python -m medscribe_ai --help
python -m medscribe_ai transcribe ./audio.m4a --language es --out transcript.json
python -m medscribe_ai extract --text-file ./examples/sample_transcript_es.txt --out history.json
python -m medscribe_ai process-audio ./audio.m4a --language es --out result.json
python -m medscribe_ai etymology gastritis
python -m medscribe_ai highlight --text "Probable gastritis y neuralgia" --max-terms 10
```

## Nota

Este repo está pensado para **probar** y **evolucionar** el módulo de IA de forma aislada.
La integración con el resto del sistema (DB/UI/Flask) se implementa después.
De momento solo hace distincion de entre medico y paciente via clasificacion de palabras clave que comunmnete usan los pacientes vs palabras clave que usa el medico en cuestion.
