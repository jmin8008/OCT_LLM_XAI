"""Assemble the multi-backbone instill matrix (issue #3) from the per-cell eval JSONs.

Reads whichever eval_*.json cells exist and prints a comparison matrix of the
headline metrics per (backbone x arm):
  continue_rate      — decision collapse (B->all-continue, D->all-stop)
  biomarker_node_acc — visible concept learned by SFT
  prognosis_node_acc — single-image prognosis (vs ΔCST), with majority baseline
  decision_balanced_acc / cf_flip_rate / faithfulness_gap

Headline: SFT learns the visible biomarker concept across backbones, but the
continue/stop DECISION and the ΔCST PROGNOSIS collapse / stay at majority — i.e.
the single-pre-image information limit + grounding⊥decision generalise across VLMs.

  python3 assemble_matrix.py            # prints table + writes sft_data/matrix.{json,md}
"""
from __future__ import annotations
import json, os

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data"

# backbone -> arm -> eval json filename (handles RetinaVLM's legacy naming).
# *_meta = metadata-augmented prompt ablation (age/gender/drug/preVA/preCST in prompt).
# v0.3: every cell uses the uniform harness naming eval_{tier}_{arm}.json.
# (The legacy eval_{A_baseline,B_sft_text,C_attn_guide,D_counterfactual}.json are v0.2
#  RetinaVLM outputs and are NOT used here — tier3 now points at eval_tier3_*.json.)
CELLS = {
    "RetinaVLM (tier3)":    {**{a: f"eval_tier3_{a}.json"  for a in "ABCD"},
                             "A_meta": "eval_tier3_A_meta.json",  "B_meta": "eval_tier3_B_meta.json"},
    "LLaVA-Med (tier2)":    {**{a: f"eval_tier2_{a}.json"  for a in "ABCD"},
                             "A_meta": "eval_tier2_A_meta.json",  "B_meta": "eval_tier2_B_meta.json"},
    "Qwen3.6-27B (tier1c)": {**{a: f"eval_tier1c_{a}.json" for a in "ABCD"},
                             "A_meta": "eval_tier1c_A_meta.json", "B_meta": "eval_tier1c_B_meta.json"},
}
ARM_ORDER = ["A", "A_meta", "B", "B_meta", "C", "D"]
METRICS = [("continue_rate", "cont"), ("biomarker_node_acc", "bm"),
           ("prognosis_node_acc", "prog"), ("prognosis_majority_baseline", "maj"),
           ("response_node_acc", "resp3"), ("response_majority_baseline", "rMaj"),
           ("response_goodpoor_balacc", "rGPbal"),
           ("decision_balanced_acc", "balAcc"), ("cf_flip_rate", "cfFlip"),
           ("faithfulness_gap", "faithGap")]


def load(fn):
    p = f"{SFT}/{fn}"
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("report", {})


def fmt(v):
    return "  -  " if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main():
    matrix = {}
    lines = ["# Multi-backbone instill matrix (issue #3)\n",
             "Headline: SFT learns the visible biomarker concept across backbones, but the",
             "continue/stop decision and the ΔCST prognosis collapse / stay at majority.\n"]
    hdr = f"| {'backbone':22} | {'arm':6} | " + " | ".join(f"{s:>7}" for _, s in METRICS) + " |"
    sep = "|" + "-" * 24 + "|--------|" + "|".join(["-" * 9] * len(METRICS)) + "|"
    print(hdr); print(sep)
    lines += [hdr, sep]
    for bb, arms in CELLS.items():
        matrix[bb] = {}
        for arm in ARM_ORDER:
            r = load(arms.get(arm, ""))
            if r is None:
                continue
            matrix[bb][arm] = r
            row = f"| {bb:22} | {arm:6} | " + " | ".join(f"{fmt(r.get(m)):>7}" for m, _ in METRICS) + " |"
            print(row); lines.append(row)
        print(sep); lines.append(sep)

    json.dump(matrix, open(f"{SFT}/matrix.json", "w"), indent=1)
    open(f"{SFT}/matrix.md", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {SFT}/matrix.json and matrix.md")


if __name__ == "__main__":
    main()
