"""Eye-level anti-VEGF VLM inference loop -> predictions JSON.

Produces, per eye and per prompt variant, the parsed predictions feeding:
  E1  CI (continue injection)        : ci_pred (1/0/'uncertain'), ci_text
  E1b VA / CST                       : va_pred, cst_pred
  E2  biomarkers (IRF/SRF/PED/HRF)   : bm_pred dict

Supports the with/without-KG RAG ablation via `kg` (a kg.GuidelineKG): when given,
guideline rules are injected as prompt context (EXPERIMENTAL_PROTOCOL.md §4.3).

Heavy model calls are isolated in run_inference so this module imports cheaply and
the loop can be driven from a subagent / notebook.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import data as data_mod
import prompts as P


def _exemplars_block(train_records, k: int = 2) -> str:
    """Render up to k few-shot exemplars (label only; images are passed separately
    only for backends that support multi-image — here we use textual exemplars)."""
    lines = []
    for r in train_records[:k]:
        ans = "continue" if r.continue_injection == 1 else "stop"
        bm = ", ".join(b for b, v in r.biomarkers.items() if v) or "no fluid"
        lines.append(f"- {r.diagnosis} eye, findings: {bm} -> {ans}")
    return "\n".join(lines)


def run_inference(
    backend,
    records: Sequence,
    variant: str = "Z0",
    kg=None,
    tasks: Sequence[str] = ("ci", "biomarkers"),
    train_records: Optional[Sequence] = None,
    max_eyes: Optional[int] = None,
) -> list[dict]:
    """Run `backend` over `records`. Returns one prediction dict per eye.

    backend : a loaded models.VLMBackend
    kg      : optional kg.GuidelineKG -> enables RAG context injection
    tasks   : subset of {'ci','biomarkers','va','cst'}
    """
    exemplars = _exemplars_block(train_records) if (variant == "F2" and train_records) else None
    preds: list[dict] = []

    for i, rec in enumerate(records):
        if max_eyes is not None and i >= max_eyes:
            break
        img = data_mod.representative_pre_bscan(rec)
        row: dict = {
            "eye_id": rec.eye_id,
            "diagnosis": rec.diagnosis,
            "variant": variant,
            "kg_context": kg is not None,
            "y_continue": rec.continue_injection,
            "y_va": rec.va,
            "y_cst": rec.cst,
            "y_biomarkers": rec.biomarkers,
        }
        if img is None:
            row["error"] = "no_pre_bscan"
            preds.append(row)
            continue

        kg_ctx = kg.render_rag_context() if kg is not None else None

        # Thinking-style models (Qwen3.6) write verbose CoT before the answer.
        # Use a large limit so the answer isn't cut off mid-reasoning.
        _is_thinking = getattr(backend, 'is_thinking_model', False) or \
                       'Qwen3.6' in getattr(backend, 'name', '')
        ci_tokens  = 2048 if _is_thinking else 200
        bm_tokens  = 1024 if _is_thinking else 100
        va_tokens  = 200  if _is_thinking else 80
        cst_tokens = 200  if _is_thinking else 80

        if "ci" in tasks:
            prompt = P.build_prompt(rec.diagnosis, variant, kg_context=kg_ctx, exemplars=exemplars)
            text = backend.generate(img, prompt, max_new_tokens=ci_tokens)
            row["ci_text"] = text
            row["ci_pred"] = P.parse_ci(text)
        if "biomarkers" in tasks:
            bm_text = backend.generate(img, P.BIOMARKER_PROMPT_JSON, max_new_tokens=bm_tokens)
            row["bm_text"] = bm_text
            row["bm_pred"] = P.parse_biomarkers_json(bm_text)
        if "va" in tasks:
            row["va_pred"] = P.parse_va(backend.generate(img, P.VA_PROMPT, max_new_tokens=va_tokens))
        if "cst" in tasks:
            row["cst_pred"] = P.parse_cst(backend.generate(img, P.CST_PROMPT, max_new_tokens=cst_tokens))

        preds.append(row)
    return preds


def save_predictions(preds: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(preds, f, indent=2, default=str, ensure_ascii=False)


def main(tier: str = "tier3", variant: str = "Z0", use_kg: bool = False, out: str = None,
         max_eyes: Optional[int] = None):
    import models

    records = data_mod.build_eye_records()
    train, test = data_mod.stratified_split(records)
    kg = None
    if use_kg:
        import kg as kg_mod
        kg = kg_mod.GuidelineKG.load_default()
    backend = models.get_backend(tier).load()
    preds = run_inference(backend, test, variant=variant, kg=kg,
                          tasks=("ci", "biomarkers", "va", "cst"),
                          train_records=train, max_eyes=max_eyes)
    out = out or f"predictions_{tier}_{variant}{'_kg' if use_kg else ''}.json"
    save_predictions(preds, out)
    print(f"wrote {len(preds)} predictions -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier3")
    ap.add_argument("--variant", default="Z0")
    ap.add_argument("--use-kg", action="store_true")
    ap.add_argument("--max-eyes", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.tier, a.variant, a.use_kg, a.out, a.max_eyes)
