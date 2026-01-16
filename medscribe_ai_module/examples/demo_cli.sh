#!/usr/bin/env bash
set -e

# Demo: extraer JSON desde un transcript de ejemplo
python -m medscribe_ai extract --text-file ./examples/sample_transcript_es.txt --out ./examples/out_history.json

# Demo: etimología
python -m medscribe_ai etymology gastritis
python -m medscribe_ai highlight --text "Probable gastritis y neuralgia" --max-terms 10

echo "Listo. Revisa ./examples/out_history.json"
