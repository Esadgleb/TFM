"""Fine-tuning (opcional) con LoRA.

Objetivo: afinar un LLM pequeño para tareas internas, por ejemplo:
- reescritura clínica (SOAP),
- normalización de abreviaturas,
- extracción de campos a JSON.

⚠️ Notas importantes
- Esto NO es necesario para el MVP.
- En CPU será muy lento; idealmente GPU.
- Mantén datos anonimizados (PHI) si entrenas con notas reales.

Formato de dataset esperado (JSONL):
  {"instruction": "...", "input": "...", "output": "..."}

Este script está pensado como punto de partida.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


def build_prompt(example: Dict[str, str]) -> str:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return f"Instrucción:\n{instruction}\n\nEntrada:\n{inp}\n\nRespuesta:\n"
    return f"Instrucción:\n{instruction}\n\nRespuesta:\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out", default="./llm_lora_out")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    # Dependencias extra sugeridas para LoRA real:
    # pip install peft datasets accelerate
    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise SystemExit(
            "Faltan dependencias para LoRA. Instala: peft datasets accelerate\n"
            "Ej: pip install peft datasets accelerate"
        ) from e

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(base, lora_cfg)

    def tokenize_fn(ex):
        prompt = build_prompt(ex)
        target = (ex.get("output") or "").strip()
        full = prompt + target + tok.eos_token
        out = tok(
            full,
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds_tok = ds.map(tokenize_fn, remove_columns=ds.column_names)

    train_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=ds_tok)
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
