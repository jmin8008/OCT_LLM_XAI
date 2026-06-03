"""Regenerate tier3 RetinaVLM attention-rollout maps for ALL 36 test eyes,
saving the raw 6x6 map_values so FER can be recomputed with the Claude-vision
real fluid masks (vs the old constant lower-center fake mask).

- prompt: Z1 (biomarker-guided) — same input as predictions_tier3_Z1.json / KG-align.
- query_token_idx = -1 (last prompt token): mini_gpt4.attention() is a prompt-only
  forward pass (no generation), so the last token is the only well-defined query.
- per eye: 6x6 rollout map, FER(real mask), FER(const fake mask), attention entropy.

Run in oct_llm env on GPU2 (see CLAUDE.md / troubleshoot.md for the nvjitlink fix).
"""
import sys, os, json, math
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import data, prompts, rollout, saliency, models

FMASK = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/fluid_masks/masks_6x6.json"
real_masks = {k: np.array(v) for k, v in json.load(open(FMASK)).items()}
const_mask = np.zeros((8, 8)); const_mask[4:8, 2:6] = 1.0
OUT = os.path.join(CODE_DIR, "xai_e3_tier3_rollout_realmask.json")


def entropy_of(amap):
    p = np.asarray(amap, dtype=np.float64).clip(0, None).ravel()
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def main():
    recs = data.build_eye_records()
    _, test = data.stratified_split(recs, test_size=0.15, seed=42)
    print(f"[init] {len(test)} test eyes", flush=True)

    backend = models.get_backend("tier3", device="cuda").load()
    inner = backend._inner                      # MiniGPT4 module
    llm = inner.llama_model
    llm.config._attn_implementation = "eager"   # needed for output_attentions
    try:
        for layer in llm.model.layers:
            layer.self_attn._attn_implementation = "eager"
    except Exception as e:
        print("[warn] could not set per-layer eager:", str(e)[:80], flush=True)
    print("[ok] RetinaVLM loaded, eager set", flush=True)

    results = []
    for i, eye in enumerate(test):
        rec = {
            "eye_id": eye.eye_id, "diagnosis": eye.diagnosis,
            "continue_injection": eye.continue_injection,
            "has_fluid": int(eye.biomarkers.get("IRF", 0) == 1 or eye.biomarkers.get("SRF", 0) == 1
                             or eye.biomarkers.get("PED", 0) == 1),
        }
        img = data.representative_pre_bscan(eye)
        if img is None:
            rec["error"] = "no_pre_bscan"
            results.append(rec); print(f"  [{i+1}/{len(test)}] {eye.eye_id} no_bscan", flush=True); continue
        try:
            prompt = prompts.build_prompt(eye.diagnosis, "Z1")
            bundle = backend.attention(img, prompt)
            amap = rollout.rollout_from_retinavlm(bundle, query_token_idx=-1)  # 6x6
            rec["map_shape"] = list(amap.shape)
            rec["map_values"] = amap.tolist()
            rec["attention_entropy"] = entropy_of(amap)
            rec["fer_const"] = saliency.fluid_energy_ratio(amap, const_mask)
            rec["fer_real"] = (saliency.fluid_energy_ratio(amap, real_masks[eye.eye_id])
                               if eye.eye_id in real_masks else float("nan"))
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} {eye.diagnosis} "
                  f"FER_const={rec['fer_const']:.4f} FER_real={rec['fer_real']:.4f} "
                  f"ent={rec['attention_entropy']:.3f}", flush=True)
        except Exception as e:
            rec["error"] = str(e)[:160]
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} ERROR {rec['error']}", flush=True)
        results.append(rec)

    json.dump(results, open(OUT, "w"), indent=1)
    ok = [r for r in results if "map_values" in r]
    fl = [r for r in ok if r["has_fluid"] == 1 and not math.isnan(r.get("fer_real", float("nan")))]
    print(f"\n[done] {len(ok)}/{len(test)} maps saved -> {OUT}", flush=True)
    if fl:
        print(f"  fluid eyes n={len(fl)}: "
              f"FER_const={np.mean([r['fer_const'] for r in fl]):.4f}  "
              f"FER_real={np.mean([r['fer_real'] for r in fl]):.4f}", flush=True)


if __name__ == "__main__":
    main()
