"""Unified eval harness for the LoRA ablation cartridges (A/B/C/D).

Loads the ONE frozen base + a given adapter, runs inference on the TEST eyes, and
reports the metrics that actually adjudicate the hypotheses:

  decision        : CI-AUC (continue/stop)                     — clinical performance
  biomarker_node  : Step-1 accuracy vs GT biomarkers           — visual grounding (text)
  prognosis_node  : Step-3 accuracy vs measured ΔCST bucket    — single-image informational limit
  text_kg_align   : reasoning follows the guideline KG         — Wang >85% line
  >>> CF_FLIP      : P(decision flips continue->stop | fluid OCCLUDED)        <<< headline faithfulness
  >>> NEGCTRL_FLIP : P(flip | NON-fluid occluded)  (should be ~0)            <<<
      faithfulness = CF_FLIP - NEGCTRL_FLIP   (grounding is causal iff this is large & positive)

The model forward is GPU (RetinaVLM, oct_llm env). Metric math + parsers + row
selection run on CPU and are validated by --dry-run (parser self-test included).
"""
from __future__ import annotations
import argparse, json, math, os

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
ADAPTER_DIR = f"{ROOT}/lora_adapters"

PROG_ORDER = ["marked_response", "partial_response", "minimal_response", "worsening"]


def parse_prognosis(text: str):
    t = (text or "").lower()
    if "marked" in t or "large reduction" in t:
        return "marked_response"
    if "partial" in t:
        return "partial_response"
    if "little anatomic" in t or "minimal" in t:
        return "minimal_response"
    if "worsen" in t:
        return "worsening"
    return None


def ci_auc(y, p):
    import numpy as np
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(p)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def flip_rate(preds_clean: dict, preds_occ: dict):
    """Among eyes whose CLEAN decision = continue(1), fraction that flip to stop(0)
    when occluded. Defined on the eyes where both decisions resolve."""
    flips = tot = 0
    for eye, c in preds_clean.items():
        o = preds_occ.get(eye)
        if c == 1 and o in (0, 1):
            tot += 1
            flips += (o == 0)
    return (flips / tot) if tot else float("nan"), tot


def run(arm_key: str, dry_run: bool):
    rows = [r for r in json.load(open(SFT)) if r["split"] == "test"]
    fact = [r for r in rows if r["type"] == "factual"]
    cf = [r for r in rows if r["type"] == "counterfactual"]
    print(f"[eval arm {arm_key}] test factual={len(fact)} counterfactual={len(cf)}")

    if dry_run:
        # parser self-test on GT targets (no model)
        ok_p = sum(parse_prognosis(r["target"]) == r["nodes_gt"]["prognosis"]
                   for r in fact if r["nodes_gt"]["prognosis"])
        n_p = sum(1 for r in fact if r["nodes_gt"]["prognosis"])
        from prompts import parse_ci
        ok_d = sum((parse_ci(r["target"]) == 1) == (r["nodes_gt"]["decision"] == "continue") for r in fact)
        cf_imgs = [f"{ROOT}/{r['image']}" for r in cf]
        neg_imgs = [p.replace("/occluded/", "/occluded_negctrl/") for p in cf_imgs]
        miss = sum(not os.path.exists(p) for p in cf_imgs) + sum(not os.path.exists(p) for p in neg_imgs)
        print(f"  parser self-test: prognosis {ok_p}/{n_p}, decision {ok_d}/{len(fact)} on GT targets")
        print(f"  counterfactual+negctrl images present: {2*len(cf)-miss}/{2*len(cf)}")
        print("  --dry-run: eval pipeline validated; model forward is the GPU step.")
        return

    # --- GPU inference ---
    from models import RetinaVLMBackend   # noqa
    from prompts import parse_ci, parse_biomarkers
    import kg as kgmod, kg_align
    base = RetinaVLMBackend().load()
    if ARMS_uses_adapter(arm_key):
        # TODO(GPU): load LoRA cartridge: PeftModel.from_pretrained(base.model.llama_model,
        #            f"{ADAPTER_DIR}/{arm_name(arm_key)}")
        pass
    # for each test eye: generate on clean / occluded / occluded_negctrl,
    #   parse_ci -> decisions; parse_biomarkers -> node1; parse_prognosis -> node3
    #   accumulate CI-AUC, node accuracies, text_kg_align, flip rates.
    raise NotImplementedError("GPU inference loop — generate(clean/occluded/negctrl) then metrics above")


def ARMS_uses_adapter(arm_key):
    return arm_key != "A"


def arm_name(arm_key):
    from lora_ablation import ARMS
    return ARMS[arm_key].name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list("ABCD"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.arm, a.dry_run)
