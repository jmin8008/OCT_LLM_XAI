"""3xN instill harness: train/eval arms A/B/C/D on any backbone (issue #3).

  TRAIN (B/C/D):  python3 -u harness.py --tier tier2 --arm C --mode train
  EVAL  (A/B/C/D): python3 -u harness.py --tier tier2 --arm C --mode eval
  smoke:          python3 -u harness.py --tier tier2 --arm C --mode train --smoke

Arm A = zero-shot (eval only, no adapter). Adapters saved at lora_adapters/{tier}_{arm}.
Eval scoring mirrors eval_lora_b.py (decision / biomarker-node / prognosis-node vs
ΔCST / text-KG / counterfactual flip-rate). Reports -> sft_data/eval_{tier}_{arm}.json.

Env (GPU2): oct_llm + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
CODE = f"{ROOT}/code"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
MASKS_NPZ = f"{ROOT}/fluid_masks_v2/masks_12x12.npz"
META_JSON = f"{ROOT}/fluid_masks_v2/metadata_v2.json"
sys.path.insert(0, CODE)


def adapter_dir(tier, arm):
    return f"{ROOT}/lora_adapters/{tier}_{arm}"


def augment_prompt(prompt, m):
    """Metadata ablation: inject the PRE-treatment patient context (age/gender/drug/
    baseline VA & CST) — never the post/Δ outcomes (those are the label). Inserted
    right before the reasoning instruction."""
    ctx = (f"Patient context: age {m['age']}, {m['gender']}; planned anti-VEGF agent: "
           f"{m['drug']}; baseline visual acuity (VA) {m['pre_va']:.2f}, baseline central "
           f"subfield thickness (CST) {m['pre_cst']:.0f} um.")
    if "Reason step by step:" in prompt:
        return prompt.replace("Reason step by step:", ctx + "\nReason step by step:", 1)
    return ctx + "\n" + prompt


def apply_meta(rows):
    """Rewrite each row's prompt in place with the metadata-augmented version."""
    meta = {r["eye_id"]: r for r in json.load(open(META_JSON))}
    for r in rows:
        if r["eye_id"] in meta:
            r["prompt"] = augment_prompt(r["prompt"], meta[r["eye_id"]])
    return rows


# --------------------------------------------------------------------------- train
def train(tier, arm, epochs, lr, lam, clip, smoke, meta=False):
    import torch, backbones
    assert arm in ("B", "C", "D"), "A is zero-shot (eval only)"
    allrows = [r for r in json.load(open(SFT)) if r["split"] == "train"]
    rows = allrows if arm == "D" else [r for r in allrows if r["type"] == "factual"]
    if meta:
        rows = apply_meta(rows)
    nf = sum(r["type"] == "factual" for r in rows)
    sfx = "_meta" if meta else ""
    print(f"[{tier} arm {arm}{sfx}] {len(rows)} rows (factual={nf}, cf={len(rows)-nf}) smoke={smoke}", flush=True)

    masks = {}
    if arm == "C":
        masks = {k: np.asarray(v) for k, v in np.load(MASKS_NPZ).items()}
        print(f"[{tier} C] masks={len(masks)} compare_grid={backbones.COMPARE_GRID} lambda={lam}", flush=True)

    bb = backbones.get_backbone(tier, device="cuda")
    if tier == "tier1c":
        bb.quantized = True            # 27B only fits as 4-bit QLoRA for training
        print("[tier1c] 4-bit QLoRA training", flush=True)
    bb.load()
    trainable = bb.attach_lora()
    print(f"[{tier}] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    if arm == "C":
        bb.enable_eager()

    def step(row):
        if arm == "C":
            total, lm_v, kl_v = bb.attn_loss(row, masks, lam)
            return total, lm_v, kl_v
        return bb.lm_loss(row), None, None

    opt = torch.optim.AdamW(trainable, lr=lr)

    if smoke:
        loss = None
        for i in range(2):
            loss, lm_v, kl_v = step(rows[i])
            opt.zero_grad(); loss.backward()
            g = torch.nn.utils.clip_grad_norm_(trainable, clip)
            opt.step()
            tag = f"lm={lm_v:.4f} kl={'skip' if kl_v is None else f'{kl_v:.4f}'}" if arm == "C" else f"loss={loss.item():.4f}"
            print(f"  smoke {i}: {tag} grad={g.item():.3f} finite={torch.isfinite(loss).item()}", flush=True)
        print("SMOKE OK" if torch.isfinite(loss) else "SMOKE NAN", flush=True)
        return

    rng = __import__("random").Random(42)
    s = 0; skipped = 0
    for ep in range(epochs):
        order = list(range(len(rows))); rng.shuffle(order)
        run = rk = 0.0; nk = 0; t0 = time.time()
        for j in order:
            loss, lm_v, kl_v = step(rows[j])
            opt.zero_grad()
            if not torch.isfinite(loss):           # bad forward — drop, don't poison weights
                skipped += 1; continue
            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(trainable, clip)
            if not torch.isfinite(g):              # exploding/NaN grad — skip the step
                opt.zero_grad(); skipped += 1; continue
            opt.step()
            run += loss.item(); s += 1
            if kl_v is not None:
                rk += kl_v; nk += 1
            if s % 30 == 0:
                extra = f" kl~{rk/max(nk,1):.4f}" if arm == "C" else ""
                print(f"  ep{ep} step{s} loss={run/30:.4f}{extra} skip={skipped} ({time.time()-t0:.0f}s)", flush=True)
                run = rk = 0.0; nk = 0
        print(f"epoch {ep} done (skipped={skipped})", flush=True)

    out = adapter_dir(tier, arm + ("_meta" if meta else ""))
    bb.save_adapter(out)
    print(f"saved adapter -> {out}", flush=True)


# --------------------------------------------------------------------------- eval
def prog_bucket(d):
    if d is None: return None
    if d <= -100: return "marked_response"
    if d <= -25: return "partial_response"
    if d <= 25: return "minimal_response"
    return "worsening"


def evaluate(tier, arm, max_new, meta=False):
    import backbones
    from prompts import parse_ci, parse_biomarkers, parse_response
    import kg as kgmod, kg_align
    from eval_lora_b import parse_prognosis

    rows = [r for r in json.load(open(SFT)) if r["split"] == "test" and r["type"] == "factual"]
    if meta:
        rows = apply_meta(rows)
    rows0 = {r["eye_id"]: r for r in rows}
    K = kgmod.GuidelineKG.load_default()
    sfx = "_meta" if meta else ""
    print(f"[eval {tier} {arm}{sfx}] {len(rows)} test eyes", flush=True)

    bb = backbones.get_backbone(tier, device="cuda").load()
    if arm != "A":
        bb.load_adapter(adapter_dir(tier, arm + sfx))
        print(f"  loaded adapter {adapter_dir(tier, arm + sfx)}", flush=True)

    from PIL import Image
    def gen(eye, sub):
        img = Image.open(f"{ROOT}/fluid_masks_v2/{sub}/{eye}.png").convert("RGB")
        return bb.generate(img, rows0[eye]["prompt"], max_new=max_new)

    preds = {}
    bm_c = bm_t = pr_c = pr_t = 0
    kg_flags = []
    clean_dec, occ_dec, neg_dec = {}, {}, {}
    resp_pred, resp_gt = {}, {}        # v0.3 Step-4 composite responder node
    for r in rows:
        eye = r["eye_id"]; gt = r["nodes_gt"]
        out = gen(eye, "clean")
        dec = parse_ci(out); clean_dec[eye] = dec
        bmp = parse_biomarkers(out)
        for k in ("IRF", "SRF", "PED"):
            if bmp.get(k) in (0, 1):
                bm_t += 1; bm_c += int(bmp[k] == gt["biomarkers"][k])
        pp = parse_prognosis(out)
        if gt["prognosis"]:
            pr_t += 1; pr_c += int(pp == gt["prognosis"])
        rsp = parse_response(out)
        resp_pred[eye] = rsp; resp_gt[eye] = gt.get("response")
        if dec in (0, 1) and all(bmp.get(k) in (0, 1) for k in ("IRF", "SRF", "PED")):
            f = kg_align.text_kg_aligned({"ci_pred": dec, "bm_pred": {**{k: bmp[k] for k in ("IRF","SRF","PED")}, "HRF": 0}}, K)
            if f is not None: kg_flags.append(f)
        preds[eye] = {"out": out, "dec": dec, "bm": bmp, "prog": pp, "resp": rsp,
                      "gt_dec": gt["decision"], "gt_prog": gt["prognosis"],
                      "gt_resp": gt.get("response"), "delta_cst": gt["delta_cst"]}
        if os.path.exists(f"{ROOT}/fluid_masks_v2/occluded/{eye}.png"):
            occ_dec[eye] = parse_ci(gen(eye, "occluded"))
            neg_dec[eye] = parse_ci(gen(eye, "occluded_negctrl"))

    y = np.array([1 if preds[e]["gt_dec"] == "continue" else 0 for e in preds])
    p = np.array([preds[e]["dec"] if preds[e]["dec"] in (0, 1) else 0 for e in preds])
    tp = ((p == 1) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    sens = tp / max((y == 1).sum(), 1); spec = tn / max((y == 0).sum(), 1)

    def flip(clean, occ):
        fl = tot = 0
        for e, c in clean.items():
            o = occ.get(e)
            if c == 1 and o in (0, 1):
                tot += 1; fl += (o == 0)
        return (fl / tot if tot else float("nan")), tot
    cf_fr, cf_n = flip(clean_dec, occ_dec)
    ng_fr, ng_n = flip(clean_dec, neg_dec)
    maj = max(np.bincount([["marked_response","partial_response","minimal_response","worsening"].index(preds[e]["gt_prog"])
                           for e in preds if preds[e]["gt_prog"]])) if pr_t else 0

    # v0.3 Step-4 composite responder node: 3-class acc + good/poor binary balanced acc
    rsp_eyes = [e for e in preds if resp_gt[e] is not None]
    rsp_c = sum(int(resp_pred[e] == resp_gt[e]) for e in rsp_eyes if resp_pred[e] is not None)
    rsp_t = sum(int(resp_pred[e] is not None) for e in rsp_eyes)
    from collections import Counter
    rmaj = max(Counter(resp_gt[e] for e in rsp_eyes).values()) if rsp_eyes else 0
    # good-vs-poor binary (exclude no_active_disease)
    bg = [e for e in rsp_eyes if resp_gt[e] in ("good_responder", "poor_responder") and resp_pred[e] in ("good_responder", "poor_responder")]
    gg = sum(resp_gt[e] == "good_responder" for e in bg); pp_ = sum(resp_gt[e] == "poor_responder" for e in bg)
    sg = sum(resp_pred[e] == "good_responder" and resp_gt[e] == "good_responder" for e in bg)
    sp = sum(resp_pred[e] == "poor_responder" and resp_gt[e] == "poor_responder" for e in bg)
    resp_bal = 0.5 * (sg / max(gg, 1) + sp / max(pp_, 1)) if bg else None

    report = {
        "tier": tier, "arm": arm, "meta": meta, "n": len(rows),
        "decision_balanced_acc": round(float(0.5 * (sens + spec)), 3),
        "decision_sensitivity": round(float(sens), 3), "decision_specificity": round(float(spec), 3),
        "continue_rate": round(float((p == 1).mean()), 3),
        "biomarker_node_acc": round(bm_c / bm_t, 3) if bm_t else None, "biomarker_node_n": bm_t,
        "prognosis_node_acc": round(pr_c / pr_t, 3) if pr_t else None, "prognosis_node_n": pr_t,
        "prognosis_majority_baseline": round(maj / pr_t, 3) if pr_t else None,
        "response_node_acc": round(rsp_c / rsp_t, 3) if rsp_t else None, "response_node_n": rsp_t,
        "response_majority_baseline": round(rmaj / len(rsp_eyes), 3) if rsp_eyes else None,
        "response_goodpoor_balacc": round(float(resp_bal), 3) if resp_bal is not None else None,
        "response_goodpoor_n": len(bg),
        "text_kg_align": round(float(np.mean(kg_flags)), 3) if kg_flags else None, "text_kg_n": len(kg_flags),
        "cf_flip_rate": round(cf_fr, 3) if cf_n else None, "cf_flip_n": cf_n,
        "negctrl_flip_rate": round(ng_fr, 3) if ng_n else None, "negctrl_flip_n": ng_n,
        "faithfulness_gap": round(cf_fr - ng_fr, 3) if (cf_n and ng_n) else None,
    }
    out_path = f"{ROOT}/sft_data/eval_{tier}_{arm}{sfx}.json"
    json.dump({"report": report, "preds": preds}, open(out_path, "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print(f"out: {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--mode", required=True, choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--attn-lambda", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--meta", action="store_true", help="metadata-augmented prompt ablation")
    a = ap.parse_args()
    if a.mode == "train":
        train(a.tier, a.arm, a.epochs, a.lr, a.attn_lambda, a.clip, a.smoke, a.meta)
    else:
        evaluate(a.tier, a.arm, a.max_new, a.meta)
