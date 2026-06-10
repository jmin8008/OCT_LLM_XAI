"""Arm B/C/D trainer: LoRA SFT of RetinaVLM on the multi-hop KG CoT.

Loss per arm:
  B  L_LM                                  (causal CE on the ANSWER tokens, factual rows)
  C  L_LM + lambda * KL(rollout_attn||fluid)  (factual rows + attention-alignment, see attn_kl.py)
  D  L_LM on factual + occluded->stop pairs   (counterfactual data, no extra loss term)

L_LM is mini_gpt4.forward(samples): form_input() masks image+prompt+pad to -100 and
keeps only the answer tokens as labels.  We attach a PEFT LoRA cartridge to the
frozen LLaMA backbone and optimise the adapter.

Arm C additionally captures the LLaMA self-attention (output_attentions=True, eager),
rolls it out differentiably (attn_kl.torch_attention_rollout), takes the answer
tokens' attention onto the 36 IMAGE tokens (6x6 grid), and minimises the forward KL
to the eye's 12x12 fluid mask AVERAGE-POOLED down to 6x6.  Grad flows into q_proj/v_proj.

Env (GPU2): oct_llm + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2.
  smoke:  python3 -u train_lora_b.py --arm C --smoke
  train:  python3 -u train_lora_b.py --arm C --epochs 3 --lr 1e-4 --attn-lambda 0.5
"""
from __future__ import annotations
import argparse, json, os, sys, time

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
MASKS_NPZ = f"{ROOT}/fluid_masks_v2/masks_12x12.npz"
ADAPTER_DIRS = {"B": f"{ROOT}/lora_adapters/B_sft_text",
                "C": f"{ROOT}/lora_adapters/C_attn_guide",
                "D": f"{ROOT}/lora_adapters/D_counterfactual"}
CODE = f"{ROOT}/code"
LORA_TARGETS = ["q_proj", "v_proj"]
MAX_TXT_LEN = 256          # override; default config is 144 and would truncate the CoT
IMG_GRID = (6, 6)          # RetinaVLM: 36 image tokens -> 6x6; mask 12x12 pools to this


def build_backend():
    sys.path.insert(0, CODE)
    import models
    backend = models.RetinaVLMBackend(device="cuda").load()
    return backend


def attach_lora(inner):
    import torch
    from peft import LoraConfig, get_peft_model
    # freeze the whole inner module first
    for p in inner.parameters():
        p.requires_grad = False
    cfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
                     bias="none", target_modules=LORA_TARGETS)
    inner.llama_model = get_peft_model(inner.llama_model, cfg)
    # keep LoRA params in fp32 for stable optimisation (base stays fp16)
    trainable = []
    for n, p in inner.llama_model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            trainable.append(p)
    return trainable


def make_sample(backend, row):
    """One training sample dict for mini_gpt4.forward."""
    from PIL import Image
    img = Image.open(f"{ROOT}/{row['image']}").convert("RGB")
    img_t = backend._img_tensor(img)                       # [1,C,H,W] normalized, on device
    return {"Image": img_t, "Question": [row["prompt"]], "Answer": [row["target"]]}


def set_eager_attention(inner):
    """output_attentions=True only returns real attention weights under eager attn.
    Works whether or not the LLaMA is PEFT-wrapped."""
    llm = inner.llama_model
    base = getattr(llm, "base_model", llm)
    core = getattr(base, "model", base)                    # PeftModel.base_model.model = LlamaForCausalLM
    try:
        core.config._attn_implementation = "eager"
    except Exception:
        pass
    try:
        llm.config._attn_implementation = "eager"
    except Exception:
        pass
    layers = core.model.layers if hasattr(core, "model") else core.layers
    n = 0
    for layer in layers:
        layer.self_attn._attn_implementation = "eager"
        n += 1
    return n


def forward_with_attn(backend, row, masks, lam, grid_hw):
    """Arm-C forward: returns (total_loss, lm_value, kl_value_or_None).

    Runs mini_gpt4.form_input to build the full [prompt|image|answer] sequence, then
    calls the LLaMA with output_attentions=True (eager) so the attention rollout is
    differentiable.  Image tokens are located by the -1 sentinel prompt_wrap writes
    into inputs_tokens; answer tokens are the targets != -100 positions.
    """
    import torch, attn_kl
    inner = backend._inner
    sample = make_sample(backend, row)
    inputs_embeds, inputs_tokens, attention_mask, targets = inner.form_input(sample)

    img_pos = (inputs_tokens[0] == -1).nonzero(as_tuple=True)[0]
    img_start, img_len = int(img_pos[0]), int(img_pos.numel())
    answer_pos = (targets[0] != -100).nonzero(as_tuple=True)[0].tolist()

    if not inner.config.model.language_model.load_in_8bit:
        inner.llama_model = inner.llama_model.to(inputs_embeds.device)
    with inner.maybe_autocast():
        outputs = inner.llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                    return_dict=True, labels=targets, output_attentions=True)
    lm_loss = outputs.loss.to(targets.device)

    mask = masks.get(row["eye_id"])
    kl_val = None
    total = lm_loss
    if mask is not None and answer_pos and outputs.attentions is not None:
        kl, _, _ = attn_kl.attn_kl_loss(outputs.attentions, img_start, img_len,
                                        answer_pos, mask, grid_hw)
        if kl is not None:
            total = lm_loss + lam * kl
            kl_val = float(kl.detach())
    return total, float(lm_loss.detach()), kl_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B", "C", "D"], default="B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--attn-lambda", type=float, default=0.5, help="arm C: KL weight")
    a = ap.parse_args()

    import torch
    ADAPTER_OUT = ADAPTER_DIRS[a.arm]
    allrows = [r for r in json.load(open(SFT)) if r["split"] == "train"]
    # B/C: factual only. D: factual + counterfactual (occluded->stop) pairs.
    rows = allrows if a.arm == "D" else [r for r in allrows if r["type"] == "factual"]
    nf = sum(r["type"] == "factual" for r in rows); nc = len(rows) - nf
    print(f"[arm {a.arm}] {len(rows)} train rows (factual={nf}, counterfactual={nc}) | smoke={a.smoke}", flush=True)

    # arm C: fluid masks + eager attention for the differentiable rollout KL term
    masks = {}
    if a.arm == "C":
        import numpy as np
        npz = np.load(MASKS_NPZ)
        masks = {k: np.asarray(npz[k]) for k in npz.files}
        n_fluid = sum(int(m.sum() > 0) for m in masks.values())
        print(f"[arm C] masks={len(masks)} ({n_fluid} with fluid) | grid={IMG_GRID} lambda={a.attn_lambda}", flush=True)

    backend = build_backend()
    inner = backend._inner
    inner.max_txt_len = MAX_TXT_LEN
    trainable = attach_lora(inner)
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in inner.parameters())
    print(f"trainable params: {n_tr:,} / {n_all:,} ({100*n_tr/n_all:.3f}%)", flush=True)

    if a.arm == "C":
        n_layers = set_eager_attention(inner)
        print(f"[arm C] eager attention set on {n_layers} layers", flush=True)

    def step_loss(row):
        """Per-arm loss for one row. Returns (loss_tensor, log_str)."""
        if a.arm == "C":
            total, lm_v, kl_v = forward_with_attn(backend, row, masks, a.attn_lambda, IMG_GRID)
            return total, f"lm={lm_v:.4f} kl={'skip' if kl_v is None else f'{kl_v:.4f}'}"
        loss = inner.forward(make_sample(backend, row))
        return loss, f"loss={loss.item():.4f}"

    opt = torch.optim.AdamW(trainable, lr=a.lr)
    inner.train()

    if a.smoke:
        loss = None
        for i in range(2):
            loss, log = step_loss(rows[i])
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            print(f"  smoke step {i}: {log} grad_norm={gnorm.item():.3f} finite={torch.isfinite(loss).item()}", flush=True)
        print("SMOKE OK" if torch.isfinite(loss) else "SMOKE NAN", flush=True)
        return

    rng = __import__("random").Random(42)
    step = 0
    for ep in range(a.epochs):
        order = list(range(len(rows))); rng.shuffle(order)
        run = 0.0; run_kl = 0.0; n_kl = 0
        t0 = time.time()
        for j in order:
            if a.arm == "C":
                loss, lm_v, kl_v = forward_with_attn(backend, rows[j], masks, a.attn_lambda, IMG_GRID)
                if kl_v is not None:
                    run_kl += kl_v; n_kl += 1
            else:
                loss = inner.forward(make_sample(backend, rows[j]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            run += loss.item(); step += 1
            if step % 30 == 0:
                extra = f" kl~{run_kl/max(n_kl,1):.4f}(n{n_kl})" if a.arm == "C" else ""
                print(f"  ep{ep} step{step} loss={run/30:.4f}{extra} ({(time.time()-t0):.0f}s)", flush=True)
                run = 0.0; run_kl = 0.0; n_kl = 0
        print(f"epoch {ep} done", flush=True)

    os.makedirs(ADAPTER_OUT, exist_ok=True)
    inner.llama_model.save_pretrained(ADAPTER_OUT)
    print(f"saved adapter -> {ADAPTER_OUT}", flush=True)


if __name__ == "__main__":
    main()
