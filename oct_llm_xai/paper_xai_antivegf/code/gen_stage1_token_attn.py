"""Stage 1 — token-conditioned attention: does RetinaVLM attend to fluid MORE when
the decision token is "continue" vs "stop"?

mini_gpt4.attention() is a forward-only pass but accepts answer_preamble, so we
teacher-force a decision word as the answer and read the rollout attention AT that
token (query=-1 = last token of the [prompt, img, question, <word>] sequence).

Per eye:
  - fer_forced_continue : FER at a forced " continue" decision token   (paired)
  - fer_forced_stop     : FER at a forced " stop" decision token       (paired)
  - ci_pred             : model's natural Z1 decision (generate)
  - fer_natural_decision: FER at the model's ACTUAL decision token in its answer
  - fer_last_token      : FER at the last prompt token (the §1c baseline)
all FER use the Claude-vision real 6x6 masks. Saved -> xai_stage1_token_attn.json.

oct_llm env, GPU2; LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1.
"""
import sys, os, json, math
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import data, prompts, rollout, saliency, models

FMASK = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/fluid_masks/masks_6x6.json"
real_masks = {k: np.array(v) for k, v in json.load(open(FMASK)).items()}
OUT = os.path.join(CODE_DIR, "xai_stage1_token_attn.json")


def find_last_token_pos(seq, word, tokenizer):
    """Index of the last token of the last occurrence of `word` in token list `seq`."""
    cands = []
    for form in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = tokenizer(form, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if ids:
            cands.append(ids)
    for ids in cands:
        n = len(ids)
        for i in range(len(seq) - n, -1, -1):
            if seq[i:i + n] == ids:
                return i + n - 1
    return None


def fer_at(bundle, q_idx, mask):
    amap = rollout.rollout_from_retinavlm(bundle, query_token_idx=q_idx)
    return float(saliency.fluid_energy_ratio(amap, mask))


def main():
    recs = data.build_eye_records()
    _, test = data.stratified_split(recs, test_size=0.15, seed=42)
    print(f"[init] {len(test)} test eyes", flush=True)

    backend = models.get_backend("tier3", device="cuda").load()
    inner = backend._inner
    llm = inner.llama_model
    llm.config._attn_implementation = "eager"
    try:
        for layer in llm.model.layers:
            layer.self_attn._attn_implementation = "eager"
    except Exception as e:
        print("[warn] per-layer eager:", str(e)[:80], flush=True)
    tok = inner.get_tokenizer()
    print("[ok] RetinaVLM loaded", flush=True)

    results = []
    for i, eye in enumerate(test):
        rec = {"eye_id": eye.eye_id, "diagnosis": eye.diagnosis,
               "y_continue": eye.continue_injection,
               "has_fluid": int(eye.biomarkers.get("IRF", 0) == 1 or eye.biomarkers.get("SRF", 0) == 1
                                or eye.biomarkers.get("PED", 0) == 1)}
        img = data.representative_pre_bscan(eye)
        mask = real_masks.get(eye.eye_id)
        if img is None or mask is None:
            rec["error"] = "no_bscan_or_mask"; results.append(rec)
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} skip ({rec['error']})", flush=True); continue
        try:
            prompt = prompts.build_prompt(eye.diagnosis, "Z1")
            img_t = backend._img_tensor(img)

            # paired forced-token probe (same eye/image, forced decision word)
            b_cont = inner.attention(img_t, [prompt], answer_preamble=[" continue"])
            b_stop = inner.attention(img_t, [prompt], answer_preamble=[" stop"])
            rec["fer_forced_continue"] = fer_at(b_cont, -1, mask)
            rec["fer_forced_stop"] = fer_at(b_stop, -1, mask)

            # natural decision (prompt-end baseline FER already in rollout_realmask.json §1c)
            answer = backend.generate(img, prompt)
            rec["ci_pred"] = prompts.parse_ci(answer)
            word = {1: "continue", 0: "stop"}.get(rec["ci_pred"])
            if word is not None:
                b_nat = inner.attention(img_t, [prompt], answer_preamble=[answer])
                seq = b_nat[1][0].tolist()
                pos = find_last_token_pos(seq, word, tok)
                if pos is not None:
                    T = len(seq)
                    rec["fer_natural_decision"] = fer_at(b_nat, pos - T, mask)
                    rec["nat_pos"] = pos; rec["nat_T"] = T
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} {eye.diagnosis} ci={rec.get('ci_pred')} "
                  f"FER cont={rec['fer_forced_continue']:.4f} stop={rec['fer_forced_stop']:.4f} "
                  f"nat={rec.get('fer_natural_decision', float('nan')):.4f}", flush=True)
        except Exception as e:
            rec["error"] = str(e)[:160]
            print(f"  [{i+1}/{len(test)}] {eye.eye_id} ERROR {rec['error']}", flush=True)
        results.append(rec)

    json.dump(results, open(OUT, "w"), indent=1)
    ok = [r for r in results if "fer_forced_continue" in r]
    print(f"\n[done] {len(ok)}/{len(test)} -> {OUT}", flush=True)
    if ok:
        fc = np.array([r["fer_forced_continue"] for r in ok])
        fs = np.array([r["fer_forced_stop"] for r in ok])
        from scipy.stats import wilcoxon
        print(f"  paired FER@continue={fc.mean():.4f}  FER@stop={fs.mean():.4f}  "
              f"delta(cont-stop)={np.mean(fc-fs):+.4f}", flush=True)
        try:
            w, p = wilcoxon(fc, fs)
            print(f"  Wilcoxon signed-rank (continue vs stop): p={p:.4f}", flush=True)
        except Exception as e:
            print("  wilcoxon:", str(e)[:60], flush=True)


if __name__ == "__main__":
    main()
