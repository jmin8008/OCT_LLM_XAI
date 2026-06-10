"""Uniform backbone adapters for the 3xN instill matrix (issue #3).

Each adapter exposes ONE interface so harness.py can run arms A/B/C/D identically
across RetinaVLM (tier3), LLaVA-Med (tier2), and Qwen3.6 (tier1c):

    bb = get_backbone("tier2").load()
    trainable = bb.attach_lora()           # B/C/D
    bb.enable_eager()                      # C only (output_attentions needs eager)
    loss = bb.lm_loss(row)                 # B/D   -> scalar
    total, lm, kl = bb.attn_loss(row, masks, lam)   # C  -> L_LM + lam*KL(rollout||fluid)
    bb.save_adapter(path)                  # after training
    bb.load_adapter(path)                  # before eval (A: skip)
    text = bb.generate(pil_image, prompt)  # eval

The attention term is attn_kl.attn_kl_loss with a COMMON 6x6 compare grid, so every
backbone's grounding is supervised/measured at the same resolution regardless of its
native image-token grid (RetinaVLM 6x6, LLaVA-Med 24x24, Qwen dynamic).
"""
from __future__ import annotations
import os, sys
import numpy as np

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
CODE = f"{ROOT}/code"
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import attn_kl

LORA_TARGETS = ["q_proj", "v_proj"]
LORA_KW = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")
COMPARE_GRID = (6, 6)
MAX_TXT_LEN = 256


class Backbone:
    tier = name = ""
    native_grid = (6, 6)

    def __init__(self, device="cuda"):
        self.device = device

    def load(self): raise NotImplementedError
    def attach_lora(self): raise NotImplementedError
    def enable_eager(self): pass
    def lm_loss(self, row): raise NotImplementedError
    def attn_loss(self, row, masks, lam): raise NotImplementedError
    def generate(self, image, prompt, max_new=220): raise NotImplementedError
    def save_adapter(self, path): raise NotImplementedError
    def load_adapter(self, path): raise NotImplementedError

    def _img(self, row):
        from PIL import Image
        return Image.open(f"{ROOT}/{row['image']}").convert("RGB")


# ---------------------------------------------------------------------------
# Tier 3 — RetinaVLM (mini_gpt4 / LLaMA-3). Reuses the proven train_lora_b path.
# ---------------------------------------------------------------------------
class RetinaVLM(Backbone):
    tier, name, native_grid = "tier3", "RetinaVLM-Specialist", (6, 6)

    def load(self):
        import models
        self.be = models.RetinaVLMBackend(device=self.device).load()
        self.inner = self.be._inner
        self.inner.max_txt_len = MAX_TXT_LEN
        return self

    def attach_lora(self):
        import train_lora_b as T
        return T.attach_lora(self.inner)

    def enable_eager(self):
        import train_lora_b as T
        T.set_eager_attention(self.inner)

    def lm_loss(self, row):
        import train_lora_b as T
        return self.inner.forward(T.make_sample(self.be, row))

    def attn_loss(self, row, masks, lam):
        import train_lora_b as T
        return T.forward_with_attn(self.be, row, masks, lam, self.native_grid)

    def generate(self, image, prompt, max_new=220):
        return self.be.generate(image, prompt, max_new_tokens=max_new)

    def save_adapter(self, path):
        os.makedirs(path, exist_ok=True)
        self.inner.llama_model.save_pretrained(path)

    def load_adapter(self, path):
        from peft import PeftModel
        self.inner.llama_model = PeftModel.from_pretrained(self.inner.llama_model, path).eval()


# ---------------------------------------------------------------------------
# Tier 2 — LLaVA-Med v1.5 (Mistral-7B, LLaVA fork). 576 image tokens = 24x24.
# ---------------------------------------------------------------------------
class LLaVAMed(Backbone):
    tier, name, native_grid = "tier2", "LLaVA-Med-v1.5", (24, 24)

    def load(self):
        import models
        self.be = models.LLaVAMedBackend(device=self.device).load()
        # tf 5.3.0 generate() passes cache_position/logits_to_keep that the 4.36-era
        # fork forward() rejects; swallow them (training forward never sends them).
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
        if not getattr(LlavaMistralForCausalLM, "_xai_patched", False):
            _orig = LlavaMistralForCausalLM.forward
            def _patched(self, *a, cache_position=None, logits_to_keep=None, **kw):
                return _orig(self, *a, **kw)
            LlavaMistralForCausalLM.forward = _patched
            LlavaMistralForCausalLM._xai_patched = True
        return self

    @property
    def core(self):
        m = self.be.model
        return m.base_model.model if hasattr(m, "base_model") else m

    def attach_lora(self):
        import torch
        from peft import LoraConfig, get_peft_model
        lm = self.be.model
        for p in lm.parameters():
            p.requires_grad = False
        cfg = LoraConfig(task_type="CAUSAL_LM", target_modules=LORA_TARGETS, **LORA_KW)
        self.be.model = get_peft_model(lm, cfg)
        trainable = [p for p in self.be.model.parameters() if p.requires_grad]
        for p in trainable:
            p.data = p.data.float()
        return trainable

    def enable_eager(self):
        c = self.core
        c.config._attn_implementation = "eager"
        for layer in c.model.layers:
            layer.self_attn._attn_implementation = "eager"

    def _build(self, row):
        """input_ids ([prompt(+ -200 img) , answer]), labels (prompt masked),
        pixel_values, img_start (position of the -200 token)."""
        import torch
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        img = self._img(row)
        pv = self.be._preprocess(img)
        qs = DEFAULT_IMAGE_TOKEN + "\n" + row["prompt"]
        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_ids = tokenizer_image_token(conv.get_prompt(), self.be.tokenizer,
                                           IMAGE_TOKEN_INDEX, return_tensors="pt")
        ans_ids = self.be.tokenizer(row["target"], return_tensors="pt",
                                    add_special_tokens=False).input_ids[0]
        input_ids = torch.cat([prompt_ids, ans_ids]).unsqueeze(0).to(self.device)
        labels = input_ids.clone()
        labels[0, :prompt_ids.shape[0]] = -100
        img_start = int((prompt_ids == IMAGE_TOKEN_INDEX).nonzero()[0])
        return input_ids, labels, pv, img_start, int(ans_ids.shape[0])

    def lm_loss(self, row):
        import torch
        input_ids, labels, pv, _, _ = self._build(row)
        with torch.autocast("cuda", dtype=torch.bfloat16):   # range-safe vs fp16 overflow
            out = self.core(input_ids=input_ids, images=pv, labels=labels, return_dict=True)
        return out.loss

    def attn_loss(self, row, masks, lam):
        import torch
        input_ids, labels, pv, img_start, ans_len = self._build(row)
        with torch.autocast("cuda", dtype=torch.bfloat16):   # range-safe vs fp16 overflow
            out = self.core(input_ids=input_ids, images=pv, labels=labels,
                            output_attentions=True, return_dict=True)
        lm_loss = out.loss
        attns = out.attentions
        T = attns[0].shape[-1]
        img_len = T - (input_ids.shape[1] - 1)          # the single -200 expands to img_len
        gh = gw = int(round(img_len ** 0.5))
        ans_start = (input_ids.shape[1] - ans_len) + (img_len - 1)
        answer_pos = list(range(ans_start, T))
        mask = masks.get(row["eye_id"])
        kl_val, total = None, lm_loss
        if mask is not None and answer_pos:
            kl, _, _ = attn_kl.attn_kl_loss(attns, img_start, img_len, answer_pos,
                                            mask, (gh, gw), compare_hw=COMPARE_GRID)
            if kl is not None:
                total = lm_loss + lam * kl
                kl_val = float(kl.detach())
        return total, float(lm_loss.detach()), kl_val

    def generate(self, image, prompt, max_new=220):
        return self.be.generate(image, prompt, max_new_tokens=max_new)

    def save_adapter(self, path):
        os.makedirs(path, exist_ok=True)
        self.be.model.save_pretrained(path)

    def load_adapter(self, path):
        from peft import PeftModel
        self.be.model = PeftModel.from_pretrained(self.be.model, path).eval()


# ---------------------------------------------------------------------------
# Tier 1c — Qwen3.6-27B (Qwen3_5ForConditionalGeneration). Dynamic image grid;
# the processor PRE-EXPANDS image_token_id (248056), so no post-expansion shift.
# 64 text layers -> output_attentions (arm C) is memory-heavy; watch for OOM.
# ---------------------------------------------------------------------------
class QwenVL(Backbone):
    tier, name = "tier1c", "Qwen3.6-27B"
    native_grid = None                          # per-image, from image_grid_thw
    IMAGE_TOKEN_ID = 248056
    MERGE = 2

    quantized = False           # set True for 4-bit QLoRA training (27B won't fit fp16)

    def load(self):
        import models, torch
        if self.quantized:
            # QLoRA: 4-bit NF4 weights so the 27B fits + LoRA on one 80GB GPU.
            from transformers import BitsAndBytesConfig, AutoProcessor
            try:
                from transformers import Qwen3_5ForConditionalGeneration as _Model
            except ImportError:
                from transformers import AutoModelForImageTextToText as _Model
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_use_double_quant=True)
            self.be = models.QwenVLBackend(device=self.device)
            self.be.processor = AutoProcessor.from_pretrained(models.QWEN_MODEL_ID)
            self.be.model = _Model.from_pretrained(
                models.QWEN_MODEL_ID, quantization_config=bnb, device_map=self.device,
                torch_dtype=torch.bfloat16, attn_implementation="eager").eval()
        else:
            self.be = models.QwenVLBackend(device=self.device).load()  # full bf16 (gen/eval)
        self.proc = self.be.processor
        return self

    @property
    def core(self):
        m = self.be.model
        return m.base_model.model if hasattr(m, "base_model") else m

    def _text_model(self):
        """The decoder stack (.model.language_model or .model) holding the layers."""
        c = self.core
        m = getattr(c, "model", c)
        return getattr(m, "language_model", m)

    def attach_lora(self):
        import torch
        from peft import LoraConfig, get_peft_model
        m = self.be.model
        if self.quantized:
            from peft import prepare_model_for_kbit_training
            m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
        else:
            for p in m.parameters():
                p.requires_grad = False
        cfg = LoraConfig(task_type="CAUSAL_LM", target_modules=LORA_TARGETS, **LORA_KW)
        self.be.model = get_peft_model(m, cfg)
        if self.quantized:
            # CRITICAL: load() leaves the model in .eval(); HF only runs gradient
            # checkpointing when self.training is True. Without train mode the 27B keeps
            # ALL layer activations -> OOM on v0.3's long CoT targets. Engage train mode +
            # disable kv-cache (incompatible w/ checkpointing) + non-reentrant ckpt.
            self.be.model.train()
            try: self.be.model.config.use_cache = False
            except Exception: pass
            self.be.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.be.model.enable_input_require_grads()
        trainable = [p for p in self.be.model.parameters() if p.requires_grad]
        if not self.quantized:
            for p in trainable:
                p.data = p.data.float()
        return trainable

    def enable_eager(self):
        self.core.config._attn_implementation = "eager"
        tm = self._text_model()
        for layer in tm.layers:
            layer.self_attn._attn_implementation = "eager"

    def _build(self, row):
        import torch
        img = self._img(row)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": row["prompt"]}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        proc = self.proc(text=[text], images=[img], return_tensors="pt")
        prompt_ids = proc["input_ids"][0]
        ans_ids = self.proc.tokenizer(row["target"], return_tensors="pt",
                                      add_special_tokens=False)["input_ids"][0]
        input_ids = torch.cat([prompt_ids, ans_ids]).unsqueeze(0).to(self.device)
        labels = input_ids.clone()
        labels[0, :prompt_ids.shape[0]] = -100
        img_pos = (prompt_ids == self.IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        img_start, img_len = int(img_pos[0]), int(img_pos.numel())
        t, h, w = [int(x) for x in proc["image_grid_thw"][0].tolist()]
        native = (h // self.MERGE, w // self.MERGE)
        extra = {"pixel_values": proc["pixel_values"].to(self.device),
                 "image_grid_thw": proc["image_grid_thw"].to(self.device)}
        return input_ids, labels, extra, img_start, img_len, native, int(ans_ids.shape[0])

    def lm_loss(self, row):
        import torch
        input_ids, labels, extra, *_ = self._build(row)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.core(input_ids=input_ids, labels=labels, return_dict=True, **extra)
        return out.loss

    def attn_loss(self, row, masks, lam):
        input_ids, labels, extra, img_start, img_len, native, ans_len = self._build(row)
        out = self.core(input_ids=input_ids, labels=labels, output_attentions=True,
                        return_dict=True, **extra)
        lm_loss = out.loss
        T = out.attentions[0].shape[-1]
        answer_pos = list(range(T - ans_len, T))
        mask = masks.get(row["eye_id"])
        kl_val, total = None, lm_loss
        if mask is not None and answer_pos and native[0] * native[1] == img_len:
            kl, _, _ = attn_kl.attn_kl_loss(out.attentions, img_start, img_len, answer_pos,
                                            mask, native, compare_hw=COMPARE_GRID)
            if kl is not None:
                total = lm_loss + lam * kl
                kl_val = float(kl.detach())
        return total, float(lm_loss.detach()), kl_val

    def generate(self, image, prompt, max_new=220):
        return self.be.generate(image, prompt, max_new_tokens=max_new)

    def save_adapter(self, path):
        os.makedirs(path, exist_ok=True)
        self.be.model.save_pretrained(path)

    def load_adapter(self, path):
        from peft import PeftModel
        self.be.model = PeftModel.from_pretrained(self.be.model, path).eval()


_REGISTRY = {"tier3": RetinaVLM, "tier2": LLaVAMed, "tier1c": QwenVL}


def get_backbone(tier, device="cuda") -> Backbone:
    if tier not in _REGISTRY:
        raise ValueError(f"unknown tier {tier}; have {list(_REGISTRY)}")
    return _REGISTRY[tier](device=device)
