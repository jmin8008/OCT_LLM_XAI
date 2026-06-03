"""Evaluate an arm (baseline A or trained B) on the 35 test eyes.

Generates the model's answer on each test eye's CLEAN image (and OCCLUDED +
NEG-CTRL for the counterfactual flip-rate), then scores:
  - decision        : balanced accuracy + continue-rate (over-continue check) + CI 'AUC'
  - biomarker_node  : per-biomarker accuracy of Step-1 vs GT biomarkers
  - prognosis_node  : 4-class accuracy of Step-3 vs measured ΔCST bucket (vs majority)
  - text_kg_align   : parsed reasoning follows the guideline KG
  - cf_flip / negctrl_flip : decision flip when fluid (vs non-fluid) is occluded

Env (GPU2): oct_llm + LD_PRELOAD + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2.
  python3 -u eval_lora_b.py --adapter none                      # arm A baseline
  python3 -u eval_lora_b.py --adapter ../lora_adapters/B_sft_text   # arm B
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
META = f"{ROOT}/fluid_masks_v2/metadata_v2.json"
CODE = f"{ROOT}/code"
PROG_THR = {"marked_response": -100.0, "partial_response": -25.0, "minimal_response": 25.0}


def prog_bucket(d):
    if d is None: return None
    if d <= -100: return "marked_response"
    if d <= -25: return "partial_response"
    if d <= 25: return "minimal_response"
    return "worsening"


def parse_prognosis(t):
    t = (t or "").lower()
    if "marked" in t or "large reduction" in t: return "marked_response"
    if "partial" in t: return "partial_response"
    if "little anatomic" in t or "minimal" in t or "stable" in t: return "minimal_response"
    if "worsen" in t: return "worsening"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--max-new", type=int, default=220)
    a = ap.parse_args()
    sys.path.insert(0, CODE)
    import torch, models
    from prompts import parse_ci, parse_biomarkers
    import kg as kgmod, kg_align

    meta = {m["eye_id"]: m for m in json.load(open(META))}
    rows = [r for r in json.load(open(SFT)) if r["split"] == "test" and r["type"] == "factual"]
    K = kgmod.GuidelineKG.load_default()

    backend = models.RetinaVLMBackend(device="cuda").load()
    arm = "A_baseline"
    if a.adapter != "none":
        from peft import PeftModel
        backend._inner.llama_model = PeftModel.from_pretrained(
            backend._inner.llama_model, a.adapter).eval()
        arm = os.path.basename(a.adapter.rstrip("/"))
    print(f"[eval {arm}] {len(rows)} test eyes", flush=True)

    from PIL import Image
    def gen(eye, sub):
        img = Image.open(f"{ROOT}/fluid_masks_v2/{sub}/{eye}.png").convert("RGB")
        return backend.generate(img, rows0[eye]["prompt"], max_new_tokens=a.max_new)

    rows0 = {r["eye_id"]: r for r in rows}
    preds = {}
    bm_correct = bm_tot = 0
    prog_correct = prog_tot = 0
    kg_flags = []
    clean_dec, occ_dec, neg_dec = {}, {}, {}
    for r in rows:
        eye = r["eye_id"]; gt = r["nodes_gt"]
        out = gen(eye, "clean")
        dec = parse_ci(out)
        clean_dec[eye] = dec
        # biomarker node
        bmp = parse_biomarkers(out)
        for k in ("IRF", "SRF", "PED"):
            v = bmp.get(k)
            if v in (0, 1):
                bm_tot += 1; bm_correct += int(v == gt["biomarkers"][k])
        # prognosis node
        pp = parse_prognosis(out)
        if gt["prognosis"]:
            prog_tot += 1; prog_correct += int(pp == gt["prognosis"])
        # text-KG
        if dec in (0, 1) and all(bmp.get(k) in (0, 1) for k in ("IRF", "SRF", "PED")):
            f = kg_align.text_kg_aligned({"ci_pred": dec, "bm_pred": {**{k: bmp[k] for k in ("IRF","SRF","PED")}, "HRF": 0}}, K)
            if f is not None: kg_flags.append(f)
        preds[eye] = {"out": out, "dec": dec, "bm": bmp, "prog": pp,
                      "gt_dec": gt["decision"], "gt_prog": gt["prognosis"], "delta_cst": gt["delta_cst"]}
        # counterfactual (only fluid eyes have occluded images)
        if os.path.exists(f"{ROOT}/fluid_masks_v2/occluded/{eye}.png"):
            occ_dec[eye] = parse_ci(gen(eye, "occluded"))
            neg_dec[eye] = parse_ci(gen(eye, "occluded_negctrl"))

    # metrics
    import numpy as np
    y = [1 if preds[e]["gt_dec"] == "continue" else 0 for e in preds]
    p = [preds[e]["dec"] if preds[e]["dec"] in (0, 1) else 0 for e in preds]
    y, p = np.array(y), np.array(p)
    tp = ((p == 1) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    sens = tp / max((y == 1).sum(), 1); spec = tn / max((y == 0).sum(), 1)
    bal_acc = 0.5 * (sens + spec)
    cont_rate = (p == 1).mean()

    def flip(clean, occ):
        fl = tot = 0
        for e, c in clean.items():
            o = occ.get(e)
            if c == 1 and o in (0, 1):
                tot += 1; fl += (o == 0)
        return (fl / tot if tot else float("nan")), tot

    cf_fr, cf_n = flip(clean_dec, occ_dec)
    ng_fr, ng_n = flip(clean_dec, neg_dec)
    maj = max(np.bincount([["marked_response","partial_response","minimal_response","worsening"].index(preds[e]["gt_prog"]) for e in preds if preds[e]["gt_prog"]])) if prog_tot else 0

    report = {
        "arm": arm, "n": len(rows),
        "decision_balanced_acc": round(float(bal_acc), 3),
        "decision_sensitivity": round(float(sens), 3), "decision_specificity": round(float(spec), 3),
        "continue_rate": round(float(cont_rate), 3),
        "biomarker_node_acc": round(bm_correct / bm_tot, 3) if bm_tot else None,
        "biomarker_node_n": bm_tot,
        "prognosis_node_acc": round(prog_correct / prog_tot, 3) if prog_tot else None,
        "prognosis_node_n": prog_tot,
        "prognosis_majority_baseline": round(maj / prog_tot, 3) if prog_tot else None,
        "text_kg_align": round(float(np.mean(kg_flags)), 3) if kg_flags else None,
        "text_kg_n": len(kg_flags),
        "cf_flip_rate": round(cf_fr, 3) if cf_n else None, "cf_flip_n": cf_n,
        "negctrl_flip_rate": round(ng_fr, 3) if ng_n else None, "negctrl_flip_n": ng_n,
        "faithfulness_gap": round(cf_fr - ng_fr, 3) if (cf_n and ng_n) else None,
    }
    out_path = f"{ROOT}/sft_data/eval_{arm}.json"
    json.dump({"report": report, "preds": preds}, open(out_path, "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print(f"out: {out_path}", flush=True)


if __name__ == "__main__":
    main()
