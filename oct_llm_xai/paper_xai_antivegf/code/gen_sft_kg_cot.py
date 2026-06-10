"""Generate the multi-hop KG chain-of-thought SFT dataset — v0.3 (causal-reordered).

The 4-layer chain is now clinically ordered:  Visual -> Pathophysiology -> DECISION
-> RESPONSE  (v0.2 was Visual -> Prognosis -> Decision, which had the clinician
"predict the future then decide" — causally backwards).

Per eye (PRE-treatment image is the ONLY model input; DIAGNOSIS IS NOT GIVEN):
  - Step 1 Visual         : GT image biomarkers (IRF/SRF/PED).               [GT pre-fluid]
  - Step 2 Pathophysiology: disease activity INFERRED from the image (no dx hint).
  - Step 3 DECISION       : guideline-based continue/stop from the findings (+ optional
                            metadata). Supervised by the recorded continue/stop label.
  - Step 4 RESPONSE       : the COMPOSITE treatment response predicted GIVEN the decision
                            — anatomic ΔCST + functional ΔVA + pre->post fluid resolution
                            -> good_responder / poor_responder. Supervised by the measured
                            outcome (single-image informational-limit finding lives here).

Diagnosis removed from the prompt so Step 2 is a genuine inference, not a relabel.
Neovascular response is judged by ΔCST + ΔVA + fluid, NOT CST alone.

Counterfactual pair (fluid-present eyes): occluded image -> no fluid -> dry -> Decision
stop -> no_active_disease response. Trains decision dependence on the fluid pixels.

Run (aptos2021 env; needs pandas for post-fluid):
  PYTHONNOUSERSITE=1 conda run -n aptos2021 python3 -u gen_sft_kg_cot.py
"""
from __future__ import annotations
import json, os

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
META = f"{ROOT}/fluid_masks_v2/metadata_v2.json"
KG = f"{ROOT}/code/antivegf_guideline_kg_v2.json"
OUT = f"{ROOT}/sft_data"
PIC_CSV = ("/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/data_response/anti-vegf-dataset/"
           "APTOS-2021/Final Datasets/train_anno_pic.csv")

PROMPT = (
    "This is a pre-treatment macular OCT B-scan. Reason step by step:\n"
    "Step 1 (Visual findings): state which retinal fluid biomarkers are present — IRF "
    "(intraretinal fluid), SRF (subretinal fluid), PED (pigment epithelial detachment).\n"
    "Step 2 (Pathophysiology): from the image alone, infer the underlying disease activity.\n"
    "Step 3 (Clinical decision): based on the findings and any provided clinical context, "
    "decide per guideline whether to continue or stop anti-VEGF therapy. State exactly "
    "'Decision: continue' or 'Decision: stop'.\n"
    "Step 4 (Predicted treatment response): predict the composite response to anti-VEGF — "
    "anatomic central-subfield-thickness change, visual-acuity change, and fluid resolution "
    "— and conclude whether this is a good responder or a poor responder."
)

# anatomic ΔCST bucket -> narrative phrase (keeps 'marked/partial/minimal' parseable)
ANATOMIC = {
    "marked_response": "a marked reduction in central subfield thickness",
    "partial_response": "a partial reduction in central subfield thickness",
    "minimal_response": "minimal anatomic change in central subfield thickness",
    "worsening": "anatomic worsening of central subfield thickness",
}


def anatomic_bucket(d):
    if d is None: return None
    if d <= -100: return "marked_response"
    if d <= -25: return "partial_response"
    if d <= 25: return "minimal_response"
    return "worsening"


def va_dir(dv):
    if dv is None: return "indeterminate"
    if dv >= 0.1: return "improved"
    if dv <= -0.1: return "declined"
    return "stable"


def composite_responder(delta_cst, delta_va):
    """Good responder if meaningful anatomic OR functional improvement, else poor."""
    anat = (delta_cst is not None and delta_cst <= -25)
    func = (delta_va is not None and delta_va >= 0.1)
    return "good_responder" if (anat or func) else "poor_responder"


def load_post_fluid():
    """Per-eye post-treatment fluid status from the pic table (IRF/SRF post). Returns
    eye_id -> 'resolved'|'persistent'|'na' relative to the pre-fluid label."""
    try:
        import pandas as pd
    except Exception:
        return {}
    df = pd.read_csv(PIC_CSV); df.columns = [c.strip() for c in df.columns]
    df["inj"] = df["injection"].astype(str).str.lower()
    post = df[df.inj.str.startswith("post")].groupby("patient ID")[["IRF", "SRF"]].max()
    return {eid: bool(r["IRF"] or r["SRF"]) for eid, r in post.iterrows()}


def visual_clause(bm):
    present = [k for k in ("IRF", "SRF", "PED") if bm.get(k)]
    absent = [k for k in ("IRF", "SRF", "PED") if not bm.get(k)]
    p = ", ".join(present) + " present" if present else ""
    a = ", ".join(absent) + " absent" if absent else ""
    return "; ".join(x for x in (p, a) if x)


def build(node_narr, bm, decision, delta_cst, delta_va, post_fluid_present, occluded):
    """Render the v0.3 reordered CoT: Visual -> Patho -> DECISION -> RESPONSE."""
    has_fluid = bool(bm.get("IRF") or bm.get("SRF"))
    # ---- Step 1: visual ----
    if occluded:
        s1 = "IRF absent, SRF absent (the intra-/subretinal fluid region is not present on this scan)."
        patho = "dry_macula"
    else:
        s1 = visual_clause(bm) + "."
        patho = "active_exudation" if has_fluid else ("ped_indeterminate" if bm.get("PED") else "dry_macula")
    # ---- Step 2: pathophysiology (image-inferred, no diagnosis) ----
    s2 = f"From the imaging findings alone, these indicate {node_narr[patho]}."
    # ---- Step 3: DECISION (guideline-based toward recorded label; no outcome leakage) ----
    if occluded:
        s3 = "With no active exudation to treat, guideline favors observation. Decision: stop"
    elif patho == "active_exudation":
        s3 = ("Active exudation favors continued anti-VEGF per guideline. Decision: continue"
              if decision == "continue" else
              "Active exudation by imaging would favor continuation; the recorded management, "
              "reflecting clinical factors beyond this single scan, was to stop. Decision: stop")
    elif patho == "dry_macula":
        s3 = ("A quiescent dry macula favors stopping per guideline. Decision: stop"
              if decision == "stop" else
              "The macula appears dry on imaging; continuation was nonetheless elected for "
              "presumed residual activity. Decision: continue")
    else:  # ped_indeterminate
        s3 = ("PED activity is indeterminate on imaging; continuation was elected for presumed "
              "active disease. Decision: continue" if decision == "continue" else
              "PED without active intra-/subretinal fluid; observation was elected. Decision: stop")
    # ---- Step 4: RESPONSE (composite outcome, DECISION-AWARE phrasing) ----
    # continue -> forward "under continued therapy ..."; stop+good -> retrospective
    # "cumulative therapy achieved a good response, so it was stopped" (no assumptive
    # "if given it would respond" contradiction for an eye the clinician already stopped);
    # stop+poor -> "response was poor, therapy stopped for limited benefit".
    if occluded or patho == "dry_macula":
        s4 = ("No active disease for anti-VEGF to act on; the macula is expected to remain "
              "anatomically stable under observation.")
    else:
        anat_ph = ANATOMIC.get(anatomic_bucket(delta_cst), "an indeterminate anatomic course")
        va = va_dir(delta_va)
        fluid_ph = ("retinal fluid resolved" if (has_fluid and post_fluid_present is False)
                    else "retinal fluid persisted" if (has_fluid and post_fluid_present is True)
                    else "fluid status indeterminate")
        comp = composite_responder(delta_cst, delta_va)
        if decision == "continue":
            s4 = (f"Under continued anti-VEGF the expected course is {anat_ph}, visual acuity "
                  f"{va}, with {fluid_ph}. Overall this is a "
                  f"{'good responder' if comp == 'good_responder' else 'poor responder'}.")
        elif comp == "good_responder":
            s4 = (f"Cumulative anti-VEGF achieved a good response — {anat_ph}, visual acuity {va} — "
                  f"so therapy was stopped, with a stable prognosis expected.")
        else:
            s4 = (f"The response to anti-VEGF was poor — {anat_ph}, visual acuity {va}, with "
                  f"{fluid_ph} — and therapy was stopped given limited benefit; a guarded prognosis.")
    target = (f"Step 1 (Visual findings): {s1}\n"
              f"Step 2 (Pathophysiology): {s2}\n"
              f"Step 3 (Clinical decision): {s3}\n"
              f"Step 4 (Predicted treatment response): {s4}")
    return target, patho


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(META))
    kg = json.load(open(KG))
    node_narr = {n["id"]: n.get("narrative", n["label"]) for n in kg["nodes"]}
    post_fluid = load_post_fluid()
    print(f"post-fluid labels for {len(post_fluid)} eyes", flush=True)

    rows, cf_worklist = [], []
    n_div = n_cf = 0
    from collections import Counter
    resp_dist = Counter()
    for m in meta:
        eye = m["eye_id"]
        bm = m["biomarkers"]
        decision = "continue" if m["continue_injection"] == 1 else "stop"
        dcst, dva = m["delta_cst"], m["delta_va"]
        has_fluid = bool(bm.get("IRF") or bm.get("SRF"))
        pf = post_fluid.get(eye)                       # True=fluid persists, False=resolved, None=na
        # pathophysiology class + divergence (guideline vs recorded decision)
        if has_fluid:
            patho_cls = "active_exudation"
        elif bm.get("PED"):
            patho_cls = "ped_indeterminate"
        else:
            patho_cls = "dry_macula"
        divergent = (patho_cls == "active_exudation" and decision == "stop") or \
                    (patho_cls == "dry_macula" and decision == "continue")
        n_div += int(divergent)

        # response GT (composite); dry/no-fluid eyes -> no_active_disease
        if has_fluid or bm.get("PED"):
            response = composite_responder(dcst, dva)
        else:
            response = "no_active_disease"
        resp_dist[response] += 1
        anat = anatomic_bucket(dcst)

        target, patho = build(node_narr, bm, decision, dcst, dva, pf, occluded=False)
        rows.append({
            "id": f"{eye}_factual", "eye_id": eye, "split": m["split"], "type": "factual",
            "image": f"fluid_masks_v2/clean/{eye}.png",
            "prompt": PROMPT, "target": target,
            "nodes_gt": {
                "biomarkers": {k: int(bm.get(k, 0)) for k in ("IRF", "SRF", "PED")},
                "pathophysiology": patho,
                "decision": decision,                        # Step-3 GT (recorded label)
                "response": response,                        # Step-4 composite GT
                "prognosis": anat,                           # ΔCST bucket (kept for parse_prognosis eval)
                "delta_cst": dcst, "delta_va": dva,
                "va_dir": va_dir(dva),
                # only meaningful when there WAS pre-fluid to resolve
                "fluid_resolution": ("na" if (not has_fluid or pf is None) else ("persistent" if pf else "resolved")),
            },
            "response_decision_divergent": divergent,
        })

        if has_fluid:           # counterfactual occlusion pair
            n_cf += 1
            cf_target, _ = build(node_narr, bm, decision, dcst, dva, pf, occluded=True)
            rows.append({
                "id": f"{eye}_cf", "eye_id": eye, "split": m["split"], "type": "counterfactual",
                "image": f"fluid_masks_v2/occluded/{eye}.png", "prompt": PROMPT, "target": cf_target,
                "nodes_gt": {"biomarkers": {"IRF": 0, "SRF": 0, "PED": int(bm.get("PED", 0))},
                             "pathophysiology": "dry_macula", "decision": "stop",
                             "response": "no_active_disease", "prognosis": None},
                "cf_contrast": (decision == "continue"),
            })
            cf_worklist.append({"eye_id": eye, "mask_key": eye})

    json.dump(rows, open(f"{OUT}/sft_kg_cot.json", "w"), indent=1, ensure_ascii=False)
    json.dump(cf_worklist, open(f"{OUT}/occlusion_worklist.json", "w"), indent=1)

    fac = [r for r in rows if r["type"] == "factual"]
    print(f"SFT rows: {len(rows)} (factual={len(fac)}, cf={len(rows)-len(fac)})")
    print(f"  divergent (guideline vs decision): {n_div}/{len(fac)}")
    print(f"  Step-4 response GT dist: {dict(resp_dist)}")
    print(f"  decision dist: {dict(Counter(r['nodes_gt']['decision'] for r in fac))}")
    print(f"out: {OUT}/sft_kg_cot.json")
    print("\n--- sample factual (active, continue) ---")
    ex = next(r for r in fac if r["nodes_gt"]["pathophysiology"] == "active_exudation" and r["nodes_gt"]["decision"] == "continue")
    print(ex["eye_id"], "| resp:", ex["nodes_gt"]["response"], "| ΔCST", ex["nodes_gt"]["delta_cst"], "ΔVA", ex["nodes_gt"]["delta_va"])
    print(ex["target"])
    print("\n--- sample divergent (active, stop) ---")
    exd = next(r for r in fac if r["response_decision_divergent"])
    print(exd["eye_id"]); print(exd["target"])


if __name__ == "__main__":
    main()
