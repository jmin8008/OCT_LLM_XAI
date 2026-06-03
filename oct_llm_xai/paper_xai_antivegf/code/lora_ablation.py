"""LoRA modular-ablation scaffold (A/B/C/D cartridges) for the XAI-instill study.

ONE frozen RetinaVLM base + four swappable LoRA adapters ("cartridges"), so the
four arms differ ONLY in their training signal (perfect variable control; no FFT,
no weight contamination):

  A  baseline        no adapter (zero-shot)                      — control anchor
  B  SFT-text        L_LM on factual CoT targets                 — "does medical SFT alone help?"
  C  +attn-guide     L_LM + lambda * KL(rollout_attn || fluid)   — attention alignment (TEST, not assumed)
  D  +counterfactual L_LM on factual + occluded pairs            — trains decision dependence on fluid pixels

Faithfulness is judged by EVAL (eval_ablation.py), not by training loss: the headline
metric is the counterfactual FLIP-RATE (occlude fluid -> decision flips) vs the
negative-control flip-rate (occlude non-fluid -> should NOT flip).

STATUS: data pipeline + arm/LoRA wiring are runnable in --dry-run (no GPU). The
model forward / loss-on-target-tokens / attention-capture hook are marked TODO(GPU)
— they require the 16 GB RetinaVLM weights + GPU and must be validated there.

Run:
  PYTHONNOUSERSITE=1 conda run -n aptos2021 python3 -u lora_ablation.py --arm B --dry-run
  (real train, GPU env oct_llm:)  python3 -u lora_ablation.py --arm B
"""
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass, field

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
MASKS = f"{ROOT}/fluid_masks_v2/masks_12x12.npz"
ADAPTER_DIR = f"{ROOT}/lora_adapters"

# LoRA targets: RetinaVLM backbone is LLaMA (mini_gpt4.py wraps llama_model; line 323
# already handles a lora-wrapped model). Attach to self-attn projections.
LORA_TARGETS = ["q_proj", "v_proj"]
LORA_CFG = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")


@dataclass
class ArmConfig:
    name: str
    use_adapter: bool
    use_lm_loss: bool
    use_attn_loss: bool          # C only: KL(attn || fluid_mask)
    use_counterfactual: bool     # D only: include occluded->stop rows
    attn_lambda: float = 0.0
    epochs: int = 3
    lr: float = 1e-4
    description: str = ""


ARMS = {
    "A": ArmConfig("A_baseline", use_adapter=False, use_lm_loss=False, use_attn_loss=False,
                   use_counterfactual=False, description="zero-shot, no cartridge"),
    "B": ArmConfig("B_sft_text", use_adapter=True, use_lm_loss=True, use_attn_loss=False,
                   use_counterfactual=False, description="L_LM on factual CoT only"),
    "C": ArmConfig("C_attn_guide", use_adapter=True, use_lm_loss=True, use_attn_loss=True,
                   use_counterfactual=False, attn_lambda=0.5,
                   description="L_LM + lambda*KL(rollout_attn||fluid_mask)"),
    "D": ArmConfig("D_counterfactual", use_adapter=True, use_lm_loss=True, use_attn_loss=False,
                   use_counterfactual=True, description="L_LM on factual + occluded counterfactual pairs"),
}


def load_rows(arm: ArmConfig, split: str = "train") -> list[dict]:
    """Select the SFT rows this arm trains on."""
    rows = [r for r in json.load(open(SFT)) if r["split"] == split]
    if not arm.use_counterfactual:
        rows = [r for r in rows if r["type"] == "factual"]
    # arm C needs the fluid mask attached for the attn loss
    return rows


def attach_lora(base_model):
    """TODO(GPU): wrap RetinaVLM's llama_model with PEFT LoRA on LORA_TARGETS.
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(target_modules=LORA_TARGETS, task_type='CAUSAL_LM', **LORA_CFG)
    base_model.model.llama_model = get_peft_model(base_model.model.llama_model, cfg)
    return base_model
    """
    raise NotImplementedError("attach_lora: wire PEFT onto mini_gpt4.llama_model (GPU step)")


def compute_losses(arm, batch, model, kg=None):
    """TODO(GPU): assemble the per-arm loss.
      L_LM : causal-LM CE on the TARGET tokens only (mask the prompt tokens).
      L_attn (arm C): KL( rollout image-attn for the decision token  ||  fluid_mask_norm ),
                      reusing rollout.image_attention_map + masks_12x12.
      D: same L_LM but the dataloader already includes occluded->stop rows, so the
         causal signal is learned from data (no extra loss term).
    Returns dict(total=..., lm=..., attn=...).
    """
    raise NotImplementedError("compute_losses: forward + target-token CE (+attn KL) — GPU step")


def train(arm_key: str, dry_run: bool):
    arm = ARMS[arm_key]
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    rows = load_rows(arm)
    masks_present = os.path.exists(MASKS)

    # --- data-pipeline validation (runs without GPU) ---
    n_fact = sum(r["type"] == "factual" for r in rows)
    n_cf = sum(r["type"] == "counterfactual" for r in rows)
    missing_img = [r["id"] for r in rows if not os.path.exists(f"{ROOT}/{r['image']}")]
    print(f"[arm {arm_key}: {arm.name}] {arm.description}")
    print(f"  train rows: {len(rows)} (factual={n_fact}, counterfactual={n_cf})")
    print(f"  use_adapter={arm.use_adapter} lm_loss={arm.use_lm_loss} "
          f"attn_loss={arm.use_attn_loss}(λ={arm.attn_lambda}) cf={arm.use_counterfactual}")
    print(f"  LoRA targets={LORA_TARGETS} cfg={LORA_CFG}")
    print(f"  masks available (for arm C attn loss): {masks_present}")
    if missing_img:
        print(f"  !! missing {len(missing_img)} images, e.g. {missing_img[:3]}")
    else:
        print(f"  all {len(rows)} referenced images exist ✓")

    if arm_key == "A":
        print("  arm A is zero-shot — no training; eval the base directly.")
        return
    if dry_run:
        print("  --dry-run: data pipeline validated; skipping model load/train (GPU step).")
        return

    # --- real training (GPU env) ---
    from models import RetinaVLMBackend   # noqa: lazy
    base = RetinaVLMBackend().load()      # frozen 16GB base
    model = attach_lora(base)             # TODO(GPU)
    # ... dataloader(rows) -> for epoch: for batch: compute_losses(arm,...).backward() ...
    # save_pretrained(f"{ADAPTER_DIR}/{arm.name}")   # the swappable cartridge
    raise NotImplementedError("training loop body — GPU step (see compute_losses TODO)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    train(a.arm, a.dry_run)
