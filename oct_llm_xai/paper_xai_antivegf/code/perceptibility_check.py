"""Issue #1 — OCCLUSION PERCEPTIBILITY CHECK.

Does RetinaVLM (base A and/or the B SFT adapter) actually PERCEIVE the cv2-inpaint
occlusion?  For each test fluid eye we feed the SAME prompt with three images —
  clean            : original B-scan
  occluded         : fluid cells inpainted out  -> macula should read "dry"
  occluded_negctrl : equal # of NON-fluid cells inpainted (same rows)
— and compare the model's free-text answer, parsed biomarkers, and decision.

Why this matters (the confound it resolves):
  Arm D trains on (occluded -> "Decision: stop") counterfactual pairs.  If the model
  cannot even SEE the fluid removal — i.e. its output on `occluded` is identical to
  `clean` — then D's counterfactual signal is pixel-decoupled noise, and the "D
  collapses to all-stop" result is a *label-prior* artefact, NOT evidence about
  grounding.  Conversely, if the model's FLUID biomarker report drops on `occluded`
  but NOT on `occluded_negctrl`, faithful perceptibility exists and the D collapse is
  a genuine finding.

Generation is greedy (mini_gpt4.query: do_sample=False), so every text delta is
attributable to the image, not sampling noise.

Env (GPU2): oct_llm + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2.
  base only:        python3 -u perceptibility_check.py --arms A
  base + B adapter: python3 -u perceptibility_check.py --arms A B
"""
from __future__ import annotations
import argparse, difflib, json, math, os, sys

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
CODE = f"{ROOT}/code"
B_ADAPTER = f"{ROOT}/lora_adapters/B_sft_text"
FLUID_BM = ("IRF", "SRF", "PED")          # biomarkers the occlusion is meant to remove


def present(v):
    """1 if reported present, else False (absent or not-mentioned/nan)."""
    return v == 1


def fluid_set(bm):
    """Set of fluid biomarkers reported PRESENT."""
    return {k for k in FLUID_BM if present(bm.get(k))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--max-new", type=int, default=220)
    a = ap.parse_args()
    sys.path.insert(0, CODE)
    import torch, models
    from prompts import parse_ci, parse_biomarkers

    rows = [r for r in json.load(open(SFT)) if r["split"] == "test" and r["type"] == "factual"]
    eyes = [r["eye_id"] for r in rows
            if os.path.exists(f"{ROOT}/fluid_masks_v2/occluded/{r['eye_id']}.png")]
    rows0 = {r["eye_id"]: r for r in rows}
    print(f"[perceptibility] {len(eyes)} test fluid eyes (have occluded img)", flush=True)

    backend = models.RetinaVLMBackend(device="cuda").load()
    from PIL import Image

    def gen(eye, sub):
        img = Image.open(f"{ROOT}/fluid_masks_v2/{sub}/{eye}.png").convert("RGB")
        return backend.generate(img, rows0[eye]["prompt"], max_new_tokens=a.max_new)

    all_reports = {}
    for arm in a.arms:
        if arm == "B":
            from peft import PeftModel
            backend._inner.llama_model = PeftModel.from_pretrained(
                backend._inner.llama_model, B_ADAPTER).eval()
            print(f"[arm B] attached adapter {B_ADAPTER}", flush=True)

        per_eye = {}
        for eye in eyes:
            rec = {}
            for sub in ("clean", "occluded", "occluded_negctrl"):
                t = gen(eye, sub)
                rec[sub] = {"out": t, "dec": parse_ci(t),
                            "bm": {k: parse_biomarkers(t).get(k) for k in FLUID_BM}}
            # text-level perceptibility (greedy => any change is the image's doing)
            cl = rec["clean"]["out"]
            rec["sim_occ"] = round(difflib.SequenceMatcher(None, cl, rec["occluded"]["out"]).ratio(), 4)
            rec["sim_neg"] = round(difflib.SequenceMatcher(None, cl, rec["occluded_negctrl"]["out"]).ratio(), 4)
            rec["identical_occ"] = cl == rec["occluded"]["out"]
            rec["identical_neg"] = cl == rec["occluded_negctrl"]["out"]
            # biomarker-level: fluid bm reported present on clean that DROP to absent
            fs_clean = fluid_set(rec["clean"]["bm"])
            rec["fluid_clean"] = sorted(fs_clean)
            rec["fluid_dropped_occ"] = sorted(fs_clean - fluid_set(rec["occluded"]["bm"]))
            rec["fluid_dropped_neg"] = sorted(fs_clean - fluid_set(rec["occluded_negctrl"]["bm"]))
            # decision-level
            rec["dec_clean"] = rec["clean"]["dec"]
            rec["dec_occ"] = rec["occluded"]["dec"]
            rec["dec_neg"] = rec["occluded_negctrl"]["dec"]
            per_eye[eye] = rec
            print(f"  [{arm}] {eye:>5} sim_occ={rec['sim_occ']:.3f} sim_neg={rec['sim_neg']:.3f} "
                  f"dropΔocc={rec['fluid_dropped_occ']} dropΔneg={rec['fluid_dropped_neg']} "
                  f"dec {rec['dec_clean']}->{rec['dec_occ']}/{rec['dec_neg']}", flush=True)

        # ---- aggregate ----
        n = len(per_eye)
        def frac(pred):
            return round(sum(pred(r) for r in per_eye.values()) / n, 3)
        import numpy as np
        sims_occ = [r["sim_occ"] for r in per_eye.values()]
        sims_neg = [r["sim_neg"] for r in per_eye.values()]
        n_fluid_clean = sum(bool(r["fluid_clean"]) for r in per_eye.values())

        # decision flip among eyes the model called continue on clean
        def dec_flip(key):
            tot = fl = 0
            for r in per_eye.values():
                if r["dec_clean"] == 1 and r[key] in (0, 1):
                    tot += 1; fl += (r[key] == 0)
            return (round(fl / tot, 3) if tot else None), tot

        flip_occ, flip_occ_n = dec_flip("dec_occ")
        flip_neg, flip_neg_n = dec_flip("dec_neg")

        report = {
            "arm": arm, "n_eyes": n,
            # text-level perceptibility
            "text_identical_occ": frac(lambda r: r["identical_occ"]),
            "text_identical_neg": frac(lambda r: r["identical_neg"]),
            "mean_sim_occ": round(float(np.mean(sims_occ)), 4),
            "mean_sim_neg": round(float(np.mean(sims_neg)), 4),
            # biomarker-level perceptibility (the key signal)
            "n_with_fluid_clean": n_fluid_clean,
            "fluid_drop_rate_occ": frac(lambda r: bool(r["fluid_dropped_occ"])),
            "fluid_drop_rate_neg": frac(lambda r: bool(r["fluid_dropped_neg"])),
            "perceptibility_gap_bm": round(
                frac(lambda r: bool(r["fluid_dropped_occ"]))
                - frac(lambda r: bool(r["fluid_dropped_neg"])), 3),
            # decision-level
            "dec_flip_rate_occ": flip_occ, "dec_flip_occ_n": flip_occ_n,
            "dec_flip_rate_neg": flip_neg, "dec_flip_neg_n": flip_neg_n,
            "perceptibility_gap_dec": (round(flip_occ - flip_neg, 3)
                                       if (flip_occ is not None and flip_neg is not None) else None),
        }
        all_reports[arm] = {"report": report, "per_eye": per_eye}
        print(f"\n=== arm {arm} perceptibility report ===", flush=True)
        print(json.dumps(report, indent=1), flush=True)

    out_path = f"{ROOT}/sft_data/perceptibility_check.json"
    json.dump(all_reports, open(out_path, "w"), indent=1)
    print(f"\nout: {out_path}", flush=True)

    # ---- one-line verdict per arm ----
    print("\n=== VERDICT ===", flush=True)
    for arm, d in all_reports.items():
        r = d["report"]
        if r["text_identical_occ"] >= 0.9:
            v = "IMPERCEPTIBLE (output ~identical clean vs occluded) -> arm D signal VOID (confound confirmed)"
        elif r["perceptibility_gap_bm"] >= 0.2:
            v = "FAITHFULLY PERCEPTIBLE (fluid bm drops on occluded, not negctrl) -> D collapse is a real finding"
        else:
            v = "PERCEPTIBLE-but-NONSPECIFIC (output changes but not fluid-specific) -> D causal claim unsupported"
        print(f"  arm {arm}: {v}", flush=True)


if __name__ == "__main__":
    main()
