"""Stage 2 — inference-time attention steering (AGE-VLM-style instill, no training).

Monkeypatch LLaMA's eager_attention_forward to ADD a positive bias to the
pre-softmax attention scores at the FLUID image-token key columns (from the
Claude-vision real 6x6 masks). This makes every token attend MORE to fluid
image patches during generation. We then ask:

  (a) does the steering actually raise FER?  (validation the hook works)
  (b) does forcing more fluid attention CHANGE the clinical output
      (ci_pred / CI-AUC / KG-align)?  (does grounding drive the decision?)

Bias sweep: 0 (baseline) / 3 (moderate, ~20x weight) / 6 (extreme, ~400x).
Image tokens = 36 (6x6), absolute key positions = pre_len + fluid_idx; pre_len
is constant (template puts <ImageHere> before the question).

oct_llm env, GPU2; LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1.
"""
import sys, os, json, math
import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import data, prompts, rollout, saliency, models
import transformers.models.llama.modeling_llama as ml

FMASK = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/fluid_masks/masks_6x6.json"
real_masks = {k: np.array(v) for k, v in json.load(open(FMASK)).items()}
BIASES = [0.0, 3.0, 6.0]

# ---- attention-steering monkeypatch -------------------------------------
STATE = {"cols": None, "bias": 0.0}


def patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states = ml.repeat_kv(key, module.num_key_value_groups)
    value_states = ml.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]
    if STATE["bias"] != 0.0 and STATE["cols"] is not None:
        kv = attn_weights.shape[-1]
        cols = STATE["cols"]
        cols = cols[cols < kv]
        if cols.numel():
            attn_weights[..., cols] = attn_weights[..., cols] + STATE["bias"]
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def main():
    recs = data.build_eye_records()
    _, test = data.stratified_split(recs, test_size=0.15, seed=42)
    print(f"[init] {len(test)} test eyes", flush=True)

    backend = models.get_backend("tier3", device="cuda").load()
    inner = backend._inner
    llm = inner.llama_model
    llm.config._attn_implementation = "eager"
    ml.eager_attention_forward = patched_eager
    try:
        ml.ALL_ATTENTION_FUNCTIONS.register("eager", patched_eager)
    except Exception as e:
        print("[warn] registry register:", str(e)[:60], flush=True)
    dev = next(llm.parameters()).device
    print("[ok] RetinaVLM loaded, eager patched", flush=True)

    # pre_len (constant): one no-bias attention forward
    img0 = None
    for e in test:
        img0 = data.representative_pre_bscan(e)
        if img0 is not None:
            dx0 = e.diagnosis; break
    b0 = inner.attention(backend._img_tensor(img0), [prompts.build_prompt(dx0, "Z1")])
    pre_len, img_len, post_len = b0[2][0]
    pre_len, img_len = int(pre_len), int(img_len)
    print(f"[geom] pre_len={pre_len} img_len={img_len} (expect 36)", flush=True)

    ci_prompt_cache = {}
    preds = {b: [] for b in BIASES}
    fer_log = []
    for i, eye in enumerate(test):
        img = data.representative_pre_bscan(eye)
        mask = real_masks.get(eye.eye_id)
        if img is None or mask is None:
            for b in BIASES:
                preds[b].append({"eye_id": eye.eye_id, "error": "no_bscan_or_mask"})
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} skip", flush=True); continue
        img_t = backend._img_tensor(img)
        fidx = np.where(mask.flatten() == 1)[0]
        cols = torch.tensor(pre_len + fidx, dtype=torch.long, device=dev)
        ci_prompt = prompts.build_prompt(eye.diagnosis, "Z1")
        try:
            # (a) FER validation: bias 0 vs max
            STATE["cols"] = cols; STATE["bias"] = 0.0
            fer0 = float(saliency.fluid_energy_ratio(
                rollout.rollout_from_retinavlm(inner.attention(img_t, [ci_prompt]), -1), mask))
            STATE["bias"] = BIASES[-1]
            ferB = float(saliency.fluid_energy_ratio(
                rollout.rollout_from_retinavlm(inner.attention(img_t, [ci_prompt]), -1), mask))
            STATE["bias"] = 0.0; STATE["cols"] = None
            fer_log.append({"eye_id": eye.eye_id, "fer_bias0": fer0,
                            f"fer_bias{BIASES[-1]:.0f}": ferB})

            # (b) clinical output under each bias
            for b in BIASES:
                STATE["cols"] = cols; STATE["bias"] = b
                ci_text = backend.generate(img, ci_prompt, max_new_tokens=150)
                bm_text = backend.generate(img, prompts.BIOMARKER_PROMPT_JSON, max_new_tokens=96)
                STATE["bias"] = 0.0; STATE["cols"] = None
                preds[b].append({
                    "eye_id": eye.eye_id, "diagnosis": eye.diagnosis,
                    "y_continue": eye.continue_injection, "y_biomarkers": eye.biomarkers,
                    "ci_text": ci_text, "ci_pred": prompts.parse_ci(ci_text),
                    "bm_text": bm_text, "bm_pred": prompts.parse_biomarkers_json(bm_text),
                })
            ci_by_b = {b: preds[b][-1]["ci_pred"] for b in BIASES}
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} {eye.diagnosis} "
                  f"FER {fer0:.3f}->{ferB:.3f} | ci@bias {ci_by_b}", flush=True)
        except Exception as ex:
            STATE["bias"] = 0.0; STATE["cols"] = None
            for b in BIASES:
                preds[b].append({"eye_id": eye.eye_id, "error": str(ex)[:140]})
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} ERROR {str(ex)[:120]}", flush=True)

    for b in BIASES:
        out = os.path.join(CODE_DIR, f"predictions_tier3_Z1_instill_b{b:.0f}.json")
        json.dump(preds[b], open(out, "w"), indent=1)
    json.dump(fer_log, open(os.path.join(CODE_DIR, "xai_stage2_fer_steer.json"), "w"), indent=1)

    ok = [r for r in fer_log if "fer_bias0" in r]
    print(f"\n[done] {len(ok)} eyes -> predictions_tier3_Z1_instill_b*.json + xai_stage2_fer_steer.json", flush=True)
    if ok:
        f0 = np.array([r["fer_bias0"] for r in ok])
        fB = np.array([r[f"fer_bias{BIASES[-1]:.0f}"] for r in ok])
        print(f"  FER bias0={f0.mean():.4f} -> bias{BIASES[-1]:.0f}={fB.mean():.4f} "
              f"({fB.mean()/max(f0.mean(),1e-9):.1f}x)  [steering validation]", flush=True)
    for b in BIASES:
        ci = [r.get("ci_pred") for r in preds[b] if r.get("ci_pred") in (0, 1)]
        cont = sum(c == 1 for c in ci)
        print(f"  bias={b}: ci_resolved={len(ci)}  continue={cont} stop={len(ci)-cont}", flush=True)


if __name__ == "__main__":
    main()
