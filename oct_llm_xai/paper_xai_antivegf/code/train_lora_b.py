"""Arm B trainer: LoRA SFT of RetinaVLM on the multi-hop KG CoT (factual rows only).

Loss = L_LM (causal CE on the ANSWER tokens only) — already implemented by
mini_gpt4.forward(samples): form_input() masks image+prompt+pad to -100 and keeps
only the answer tokens as labels. We just attach a PEFT LoRA cartridge to the
frozen LLaMA backbone and optimise the adapter.

Env (GPU2): oct_llm + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2.
  smoke:  python3 -u train_lora_b.py --smoke
  train:  python3 -u train_lora_b.py --epochs 3 --lr 1e-4
"""
from __future__ import annotations
import argparse, json, os, sys, time

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
ADAPTER_DIRS = {"B": f"{ROOT}/lora_adapters/B_sft_text",
                "D": f"{ROOT}/lora_adapters/D_counterfactual"}
CODE = f"{ROOT}/code"
LORA_TARGETS = ["q_proj", "v_proj"]
MAX_TXT_LEN = 256          # override; default config is 144 and would truncate the CoT


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B", "D"], default="B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    a = ap.parse_args()

    import torch
    ADAPTER_OUT = ADAPTER_DIRS[a.arm]
    allrows = [r for r in json.load(open(SFT)) if r["split"] == "train"]
    # B: factual only. D: factual + counterfactual (occluded->stop) pairs.
    rows = allrows if a.arm == "D" else [r for r in allrows if r["type"] == "factual"]
    nf = sum(r["type"] == "factual" for r in rows); nc = len(rows) - nf
    print(f"[arm {a.arm}] {len(rows)} train rows (factual={nf}, counterfactual={nc}) | smoke={a.smoke}", flush=True)

    backend = build_backend()
    inner = backend._inner
    inner.max_txt_len = MAX_TXT_LEN
    trainable = attach_lora(inner)
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in inner.parameters())
    print(f"trainable params: {n_tr:,} / {n_all:,} ({100*n_tr/n_all:.3f}%)", flush=True)

    opt = torch.optim.AdamW(trainable, lr=a.lr)
    inner.train()

    if a.smoke:
        for i in range(2):
            loss = inner.forward(make_sample(backend, rows[i]))
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            print(f"  smoke step {i}: loss={loss.item():.4f} grad_norm={gnorm.item():.3f} finite={torch.isfinite(loss).item()}", flush=True)
        print("SMOKE OK" if torch.isfinite(loss) else "SMOKE NAN", flush=True)
        return

    rng = __import__("random").Random(42)
    step = 0
    for ep in range(a.epochs):
        order = list(range(len(rows))); rng.shuffle(order)
        run = 0.0
        t0 = time.time()
        for j in order:
            loss = inner.forward(make_sample(backend, rows[j]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            run += loss.item(); step += 1
            if step % 30 == 0:
                print(f"  ep{ep} step{step} loss={run/30:.4f} ({(time.time()-t0):.0f}s)", flush=True)
                run = 0.0
        print(f"epoch {ep} done", flush=True)

    os.makedirs(ADAPTER_OUT, exist_ok=True)
    inner.llama_model.save_pretrained(ADAPTER_OUT)
    print(f"saved adapter -> {ADAPTER_OUT}", flush=True)


if __name__ == "__main__":
    main()
