"""Issue 2.2 — Per-slice biomarker eval (지각 regime → Table A).

Evaluates IRF/SRF/PED *presence* PER B-SCAN SLICE (not per eye), using the APTOS
per-image labels in train_anno_pic.csv. Only test-eye pre-injection slices are
scored (no train leakage — instilled adapters trained on train-eye representative
slices). Reports each biomarker INDEPENDENTLY (no "finds biomarkers well" aggregate):
balanced accuracy + present/absent counts + eye-clustered bootstrap CI. IRF is
present-biased (~0.87) → flagged; SRF/PED (balanced) carry the real instill signal.

Preprocessing mirrors gen_dataset_v2.render_eye exactly (macular crop → contrast
stretch → resize to UPSCALE_W) so per-slice images match what the model was trained/
evaluated on. Prompt = the 4-step CoT PROMPT (same as harness) so Table A is directly
comparable to the eye-level biomarker_node_acc in matrix.

  DRY-RUN (no GPU): python3 -u perslice_biomarker.py --dry-run
  EVAL  (GPU):      python3 -u perslice_biomarker.py --tier tier3 --arm B
"""
from __future__ import annotations
import argparse, json, math, os, sys

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
CODE = f"{ROOT}/code"
SPLIT = f"{ROOT}/sft_data/split_v04.json"
sys.path.insert(0, CODE)

from data import extract_macular_oct_crop, TRAIN_IMG_ROOT, OCT_X_START, load_pic_table
from gen_dataset_v2 import contrast_stretch, UPSCALE_W
from gen_sft_kg_cot import PROMPT          # the 4-step CoT prompt (train/eval consistent)
from prompts import parse_biomarkers

BMS = ("IRF", "SRF", "PED")


# --------------------------------------------------------------------------- data
def test_eyes():
    sp = json.load(open(SPLIT))
    return {k for k, v in sp.items() if not k.startswith("_") and v == "test"}


def enumerate_slices(eyes):
    """Per-slice records for the given eyes' PRE-injection B-scans.
    Returns list of {eye, name, path, IRF, SRF, PED} (existing files only)."""
    df = load_pic_table()
    df = df[df["injection"].str.lower().str.startswith("pre")]
    df = df[df["patient ID"].isin(eyes)]
    recs, missing = [], 0
    for _, r in df.iterrows():
        eye = str(r["patient ID"]).strip()
        name = str(r["image name"]).strip()
        path = os.path.join(TRAIN_IMG_ROOT, eye, "Pre Injection OCT Images", f"{name}.jpg")
        if not os.path.exists(path):
            missing += 1
            continue
        recs.append({"eye": eye, "name": name, "path": path,
                     "IRF": int(r["IRF"]), "SRF": int(r["SRF"]), "PED": int(r["PED"])})
    if missing:
        print(f"  ⚠️ {missing} CSV pre-rows had no matching .jpg (skipped)", flush=True)
    return recs


def render_slice(path):
    """Same preprocessing as gen_dataset_v2.render_eye, applied to ONE slice file."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.size[0] > OCT_X_START + 50:           # composite fundus+OCT -> macular crop
        img = extract_macular_oct_crop(img)
    img = contrast_stretch(img)
    w, h = img.size
    return img.resize((UPSCALE_W, int(round(h * UPSCALE_W / w))), Image.LANCZOS)


# --------------------------------------------------------------------------- score
def _balacc(y, p):
    """Balanced accuracy from binary y/p lists (ignoring nan preds)."""
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
    npos = sum(1 for a in y if a == 1); nneg = sum(1 for a in y if a == 0)
    sens = tp / npos if npos else float("nan")
    spec = tn / nneg if nneg else float("nan")
    return 0.5 * (sens + spec), sens, spec


def score(recs, preds, n_boot=2000, seed=0):
    """Per-biomarker independent metrics + eye-clustered bootstrap 95% CI.
    preds: {(eye,name): {IRF/SRF/PED: 0/1/nan}}."""
    import numpy as np
    rng = np.random.default_rng(seed)
    eyes = sorted({r["eye"] for r in recs})
    by_eye = {e: [r for r in recs if r["eye"] == e] for e in eyes}
    out = {}
    for bm in BMS:
        pairs = [(r[bm], preds.get((r["eye"], r["name"]), {}).get(bm)) for r in recs]
        pairs = [(y, p) for y, p in pairs if p in (0, 1)]          # drop unparsed
        y = [a for a, _ in pairs]; p = [b for _, b in pairs]
        ba, sens, spec = _balacc(y, p)
        # eye-clustered bootstrap: resample eyes with replacement
        boot = []
        for _ in range(n_boot):
            samp = rng.choice(eyes, size=len(eyes), replace=True)
            yy, pp = [], []
            for e in samp:
                for r in by_eye[e]:
                    q = preds.get((r["eye"], r["name"]), {}).get(bm)
                    if q in (0, 1):
                        yy.append(r[bm]); pp.append(q)
            if yy:
                boot.append(_balacc(yy, pp)[0])
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (float("nan"),) * 2
        out[bm] = {"bal_acc": round(ba, 3), "sens": round(sens, 3), "spec": round(spec, 3),
                   "ci95": [round(ci[0], 3), round(ci[1], 3)],
                   "n_present": sum(1 for a in y if a == 1), "n_absent": sum(1 for a in y if a == 0),
                   "n_scored": len(y), "n_unparsed": sum(1 for r in recs if preds.get((r["eye"], r["name"]), {}).get(bm) not in (0, 1))}
    return out


# --------------------------------------------------------------------------- runners
def dry_run():
    eyes = test_eyes()
    recs = enumerate_slices(eyes)
    print(f"[dry-run] test eyes={len(eyes)} | enumerated pre-slices={len(recs)} "
          f"(eye당 {len(recs)/max(len(eyes),1):.1f})", flush=True)
    print(f"{'bm':4} {'present':>8} {'absent':>7} {'absent%':>8}")
    for bm in BMS:
        pr = sum(r[bm] == 1 for r in recs); ab = sum(r[bm] == 0 for r in recs)
        print(f"{bm:4} {pr:>8} {ab:>7} {ab/len(recs):>8.2f}")
    # preprocessing smoke on first 3 slices
    print("preprocess smoke (rendered size):", flush=True)
    for r in recs[:3]:
        img = render_slice(r["path"])
        print(f"  {r['eye']}/{r['name']}.jpg -> {img.size}", flush=True)
    print("OK — GPU eval will run the 4-step prompt on these slices.", flush=True)


def run(tier, arm, max_new=220, meta=False, limit=None):
    import backbones
    sfx = "_meta" if meta else ""
    eyes = test_eyes()
    recs = enumerate_slices(eyes)
    if limit:
        recs = recs[:limit]
    print(f"[perslice {tier} {arm}{sfx}] {len(eyes)} test eyes, {len(recs)} slices", flush=True)
    bb = backbones.get_backbone(tier, device="cuda").load()
    if arm != "A":
        bb.load_adapter(f"{ROOT}/lora_adapters/{tier}_{arm}{sfx}")
    preds = {}
    for i, r in enumerate(recs):
        out = bb.generate(render_slice(r["path"]), PROMPT, max_new=max_new)
        preds[(r["eye"], r["name"])] = parse_biomarkers(out)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)
    report = score(recs, preds)
    out_path = f"{ROOT}/sft_data/perslice_{tier}_{arm}{sfx}.json"
    json.dump({"tier": tier, "arm": arm, "meta": meta, "n_slices": len(recs),
               "n_eyes": len(eyes), "per_biomarker": report}, open(out_path, "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print(f"out: {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tier")
    ap.add_argument("--arm", choices=["A", "B", "C", "D"])
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.dry_run or not a.tier:
        dry_run()
    else:
        run(a.tier, a.arm, a.max_new, a.meta, a.limit)
