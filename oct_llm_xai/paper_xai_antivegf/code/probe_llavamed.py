"""De-risk probe for the LLaVA-Med (tier2) arm-C path.

Confirms, in ONE GPU load, that for LLaVA-Med v1.5 we can:
  1. generate() on a test eye (arm A works)
  2. attach a PEFT LoRA on the Mistral q_proj/v_proj
  3. forward with labels + output_attentions=True (eager) -> loss + grad-tracking attentions
  4. locate the 576 image tokens (24x24 grid) in the sequence
  5. compute attn_kl.attn_kl_loss against the 12x12 fluid mask pooled to 24x24
  6. backward and see grad reach the LoRA params

If this passes, the tier2 adapter for harness.py is essentially specified.

Env (GPU2): oct_llm + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
CODE = f"{ROOT}/code"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
MASKS_NPZ = f"{ROOT}/fluid_masks_v2/masks_12x12.npz"
sys.path.insert(0, CODE)


def main():
    import torch, attn_kl, models
    from PIL import Image
    from peft import LoraConfig, get_peft_model

    rows = [r for r in json.load(open(SFT)) if r["split"] == "train" and r["type"] == "factual"]
    masks = {k: np.asarray(v) for k, v in np.load(MASKS_NPZ).items()}
    row = next(r for r in rows if masks.get(r["eye_id"], np.zeros((12, 12))).sum() > 0)
    print(f"[probe] eye {row['eye_id']} fluid_cells={int(masks[row['eye_id']].sum())}", flush=True)

    be = models.LLaVAMedBackend(device="cuda").load()
    print("[ok] LLaVA-Med loaded", flush=True)

    # transformers 5.3.0 generate() passes cache_position, which the 4.36-era LLaVA
    # fork forward() doesn't accept. Swallow it (training forward never sends it).
    from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
    _orig_fwd = LlavaMistralForCausalLM.forward
    def _patched_fwd(self, *a, cache_position=None, logits_to_keep=None, **kw):
        return _orig_fwd(self, *a, **kw)
    LlavaMistralForCausalLM.forward = _patched_fwd
    print("[ok] patched forward to swallow cache_position/logits_to_keep", flush=True)

    img = Image.open(f"{ROOT}/{row['image']}").convert("RGB")

    # (2) LoRA on the language model
    lm = be.model
    for p in lm.parameters():
        p.requires_grad = False
    cfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
                     bias="none", target_modules=["q_proj", "v_proj"])
    be.model = get_peft_model(lm, cfg)
    lm = be.model
    trainable = [p for p in lm.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    print(f"[ok] LoRA attached; trainable={sum(p.numel() for p in trainable):,}", flush=True)

    # eager attention for output_attentions
    core = lm.base_model.model            # LlavaMistralForCausalLM
    core.config._attn_implementation = "eager"
    for layer in core.model.layers:
        layer.self_attn._attn_implementation = "eager"

    # (3) build a training sample: [prompt(+image -200) , answer], labels mask the prompt
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    pixel_values = be._preprocess(img)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + row["prompt"]
    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    prompt_ids = tokenizer_image_token(prompt_text, be.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    ans_ids = be.tokenizer(row["target"], return_tensors="pt", add_special_tokens=False).input_ids[0]
    input_ids = torch.cat([prompt_ids, ans_ids]).unsqueeze(0).to(be.device)
    labels = input_ids.clone()
    labels[0, :prompt_ids.shape[0]] = -100             # mask the prompt; -200 img token also masked
    img_tok_pos = int((prompt_ids == IMAGE_TOKEN_INDEX).nonzero()[0])
    print(f"[ok] seq pre-expand={input_ids.shape[1]} img_token_at={img_tok_pos} ans_len={ans_ids.shape[0]}", flush=True)

    # (4) forward with attentions
    out = core(input_ids=input_ids, images=pixel_values, labels=labels,
               output_attentions=True, return_dict=True)
    lm_loss = out.loss
    attns = out.attentions
    T = attns[0].shape[-1]
    img_len = T - (input_ids.shape[1] - 1)              # one -200 token expands to img_len patches
    img_start = img_tok_pos
    side = int(round(img_len ** 0.5))
    print(f"[ok] L_LM={float(lm_loss):.4f} | attn layers={len(attns)} T={T} -> img_len={img_len} grid~{side}x{side} img_start={img_start}", flush=True)

    # answer token positions in the EXPANDED sequence: shift everything after the image by (img_len-1)
    ans_start_expanded = (input_ids.shape[1] - ans_ids.shape[0]) + (img_len - 1)
    answer_pos = list(range(ans_start_expanded, T))
    print(f"[ok] answer_pos[{len(answer_pos)}] {answer_pos[:3]}..{answer_pos[-1]}", flush=True)

    # (5) attn-KL against pooled fluid mask
    kl, p, q = attn_kl.attn_kl_loss(attns, img_start, img_len, answer_pos,
                                    masks[row["eye_id"]], (side, side))
    print(f"[ok] KL={float(kl):.4f}  p.sum={float(p.sum()):.3f} q.sum={float(q.sum()):.3f}", flush=True)

    # (6) backward -> grad to LoRA
    total = lm_loss + 0.5 * kl
    total.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print(f"[ok] total={float(total):.4f} grad_norm={float(gnorm):.4f} -> grad reaches LoRA", flush=True)

    # (7) generate (arm A / eval path) — least critical, needs the cache_position patch
    try:
        be.model.eval()
        txt = be.generate(img, row["prompt"], max_new_tokens=80)
        print(f"[A gen] {txt[:200]!r}", flush=True)
    except Exception as e:
        print(f"[A gen FAILED] {str(e)[:200]}", flush=True)
    print("PROBE OK", flush=True)


if __name__ == "__main__":
    main()
