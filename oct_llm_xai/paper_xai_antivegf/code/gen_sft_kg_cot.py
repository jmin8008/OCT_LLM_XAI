"""Generate the multi-hop KG chain-of-thought SFT dataset (factual + counterfactual).

Per eye (PRE-treatment image is the ONLY model input), the target text walks a
4-layer KG chain — Visual -> Pathophysiology -> Prognosis -> Decision — where:
  - Visual node       : the GT image biomarkers (IRF/SRF/PED).            [GT-supervised]
  - Pathophysiology   : derived from the visual node (image-grounded).
  - Prognosis         : a PREDICTED treatment-response category, the ground truth of
                        which is the MEASURED ΔCST bucket. The model must infer it
                        from the image alone — its accuracy is scored separately
                        (single-image informational-limit finding).        [ΔCST-supervised]
  - Decision          : the recorded continue/stop label.                  [label-supervised]

The connective narrative is GUIDELINE-based, NOT deterministic: when the response
(ΔCST) and the recorded decision diverge (~32% of eyes; AUC(ΔCST->continue)=0.68),
the text says so honestly ("...despite the response, injections were continued,
indicating residual activity") rather than asserting a contradictory "therefore".

Counterfactual pair: for fluid-present eyes we additionally emit a target keyed to
the OCCLUDED image (fluid removed) -> Visual=no fluid -> dry -> no active disease ->
Decision: stop. This trains decision dependence on the fluid pixels (the causal
signal), and is meaningful as a CONTRAST only for factual=continue eyes (logged).

Run (aptos2021 env):
  PYTHONNOUSERSITE=1 conda run -n aptos2021 python3 -u gen_sft_kg_cot.py
"""
from __future__ import annotations
import json, os

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
META = f"{ROOT}/fluid_masks_v2/metadata_v2.json"
KG = f"{ROOT}/code/antivegf_guideline_kg_v2.json"
OUT = f"{ROOT}/sft_data"

PROMPT = (
    "This is a pre-treatment macular OCT B-scan of a patient with {dx}. Reason step by step:\n"
    "Step 1 (Visual findings): state which retinal fluid biomarkers are present — IRF "
    "(intraretinal fluid), SRF (subretinal fluid), PED (pigment epithelial detachment).\n"
    "Step 2 (Pathophysiology): infer the underlying disease activity.\n"
    "Step 3 (Predicted treatment response): predict the likely anatomic response of the "
    "central subfield thickness (CST) to anti-VEGF therapy.\n"
    "Step 4 (Decision): give the final anti-VEGF continuation decision.\n"
    "End your answer with exactly 'Decision: continue' or 'Decision: stop'."
)


def prognosis_bucket(delta_cst, thr):
    if delta_cst is None:
        return None
    if delta_cst <= thr["marked_response"]["max"]:
        return "marked_response"
    if delta_cst <= thr["partial_response"]["max"]:
        return "partial_response"
    if delta_cst <= thr["minimal_response"]["max"]:
        return "minimal_response"
    return "worsening"


def visual_clause(bm):
    present = [k for k in ("IRF", "SRF", "PED") if bm.get(k)]
    absent = [k for k in ("IRF", "SRF", "PED") if not bm.get(k)]
    p = ", ".join(present) + " present" if present else ""
    a = ", ".join(absent) + " absent" if absent else ""
    return "; ".join(x for x in (p, a) if x)


def build(node_narr, bm, prog_id, decision, guideline, divergent, occluded):
    """Render the 4-step CoT. node_narr: id->narrative map."""
    has_fluid = bool(bm.get("IRF") or bm.get("SRF"))
    # Step 1 visual
    if occluded:
        s1 = "IRF absent, SRF absent (the intra-/subretinal fluid region is not present on this scan)."
        patho = "dry_macula"
    else:
        s1 = visual_clause(bm) + "."
        patho = "active_exudation" if has_fluid else ("ped_indeterminate" if bm.get("PED") else "dry_macula")
    # Step 2 pathophysiology
    s2 = f"These findings indicate {node_narr[patho]}."
    # Step 3 prognosis
    if occluded:
        prog = "no_active_disease"
        s3 = f"With no active fluid, there is {node_narr[prog]}; the macula is anatomically stable."
    else:
        prog = prog_id
        s3 = f"On this single pre-treatment image the expected course is {node_narr[prog]}." if prog else \
             "The anatomic response is indeterminate from this single image."
    # Step 4 decision (guideline-based, honest about divergence; prognosis-aware, no
    # unconditional response claims — Step 3 already states the measured response).
    if occluded:
        s4 = "With no active exudation to treat, anti-VEGF is not indicated. Decision: stop"
    elif patho == "active_exudation":
        if decision == "continue":
            s4 = "Active exudation favors continued therapy, consistent with the recorded management. Decision: continue"
        else:  # divergent: fluid but stopped
            s4 = ("Although baseline imaging showed active fluid, the treated macula stabilized and "
                  "therapy was stopped. Decision: stop")
    elif patho == "dry_macula":
        if decision == "stop":
            s4 = "A dry macula favors stopping, consistent with the recorded management. Decision: stop"
        else:  # divergent: dry but continued
            s4 = ("Although the macula appeared dry on this scan, ongoing therapy was elected for "
                  "presumed residual activity. Decision: continue")
    else:  # ped_indeterminate
        if decision == "continue":
            s4 = ("PED activity is indeterminate on imaging; ongoing therapy was elected for presumed "
                  "active disease. Decision: continue")
        else:
            s4 = "PED without active intra-/subretinal fluid; therapy was stopped under observation. Decision: stop"
    return (f"Step 1 (Visual findings): {s1}\n"
            f"Step 2 (Pathophysiology): {s2}\n"
            f"Step 3 (Predicted treatment response): {s3}\n"
            f"Step 4 (Decision): {s4}"), patho, prog


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(META))
    kg = json.load(open(KG))
    node_narr = {n["id"]: n.get("narrative", n["label"]) for n in kg["nodes"]}
    thr = kg["prognosis_thresholds_delta_cst_um"]

    rows, cf_worklist = [], []
    n_div = n_cf = 0
    for m in meta:
        eye, dx = m["eye_id"], m["diagnosis"]
        bm = m["biomarkers"]
        label = m["continue_injection"]
        decision = "continue" if label == 1 else "stop"
        prog_id = prognosis_bucket(m["delta_cst"], thr)
        has_fluid = bool(bm.get("IRF") or bm.get("SRF"))
        # guideline class + divergence defined by pathophysiology (matches Step-4 narrative)
        if has_fluid:
            patho_cls, guideline = "active_exudation", "continue"
        elif bm.get("PED"):
            patho_cls, guideline = "ped_indeterminate", "case_dependent"
        else:
            patho_cls, guideline = "dry_macula", "stop"
        divergent = (patho_cls == "active_exudation" and decision == "stop") or \
                    (patho_cls == "dry_macula" and decision == "continue")
        if divergent:
            n_div += 1

        target, patho, prog = build(node_narr, bm, prog_id, decision, guideline, divergent, occluded=False)
        rows.append({
            "id": f"{eye}_factual", "eye_id": eye, "split": m["split"], "type": "factual",
            "image": f"fluid_masks_v2/clean/{eye}.png",
            "prompt": PROMPT.format(dx=dx), "target": target,
            "nodes_gt": {  # for per-node eval
                "biomarkers": {k: int(bm.get(k, 0)) for k in ("IRF", "SRF", "PED")},
                "pathophysiology": patho,
                "prognosis": prog_id,           # measured ΔCST bucket = prognosis-node GT
                "delta_cst": m["delta_cst"],
                "decision": decision,
            },
            "guideline_suggestion": guideline,
            "response_decision_divergent": divergent,
        })

        # counterfactual (only where fluid exists to occlude)
        if has_fluid:
            n_cf += 1
            cf_target, _, _ = build(node_narr, bm, prog_id, decision, guideline, divergent, occluded=True)
            rows.append({
                "id": f"{eye}_cf", "eye_id": eye, "split": m["split"], "type": "counterfactual",
                "image": f"fluid_masks_v2/occluded/{eye}.png",
                "prompt": PROMPT.format(dx=dx), "target": cf_target,
                "nodes_gt": {"biomarkers": {"IRF": 0, "SRF": 0, "PED": int(bm.get("PED", 0))},
                             "pathophysiology": "dry_macula", "prognosis": "no_active_disease",
                             "decision": "stop"},
                "cf_contrast": (decision == "continue"),   # meaningful flip only if factual was continue
            })
            cf_worklist.append({"eye_id": eye, "mask_key": eye})

    json.dump(rows, open(f"{OUT}/sft_kg_cot.json", "w"), indent=1, ensure_ascii=False)
    json.dump(cf_worklist, open(f"{OUT}/occlusion_worklist.json", "w"), indent=1)

    fac = [r for r in rows if r["type"] == "factual"]
    cf = [r for r in rows if r["type"] == "counterfactual"]
    cf_contrast = sum(r["cf_contrast"] for r in cf)
    print(f"SFT rows: {len(rows)} (factual={len(fac)}, counterfactual={len(cf)})")
    print(f"  response/decision divergent (honest-narrative) eyes: {n_div}/{len(fac)}")
    print(f"  counterfactual pairs: {n_cf}  (meaningful flip-contrast [factual=continue]: {cf_contrast})")
    from collections import Counter
    print("  prognosis-node GT dist:", dict(Counter(r["nodes_gt"]["prognosis"] for r in fac)))
    print(f"out: {OUT}/sft_kg_cot.json, occlusion_worklist.json")
    print("\n--- sample factual (divergent eye) ---")
    ex = next(r for r in fac if r["response_decision_divergent"])
    print(ex["eye_id"], "| label-decision:", ex["nodes_gt"]["decision"], "| ΔCST:", ex["nodes_gt"]["delta_cst"])
    print(ex["target"])
    print("\n--- sample counterfactual ---")
    print(cf[0]["eye_id"]); print(cf[0]["target"])


if __name__ == "__main__":
    main()
