"""3-tier VLM backends with a common interface.

    Tier 1 (generalist)      : Qwen3-VL-8B-Instruct        (HF transformers)
    Tier 2 (medical-general) : LLaVA-Med                    (SpecialistVLMs/models/llava_med.py)
    Tier 3 (OCT-specialist)  : RetinaVLM-Specialist         (SpecialistVLMs retinavlm_wrapper -> mini_gpt4)

All heavy imports are LAZY (inside .load()) so this module imports without GPUs or
model weights. Common interface:

    backend = get_backend("tier3"); backend.load()
    text   = backend.generate(pil_image, prompt)
    attn   = backend.attention(pil_image, prompt)        # may raise NotImplementedError
    logits = backend.class_logits(pil_image, prompt, ["continue", "stop"])

Tier-3 exposes the full attention tensor (mini_gpt4.attention, L392) which feeds
rollout.py; Tier-1/2 expose attentions via HF output_attentions where available.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

SPECIALIST_ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/SpecialistVLMs"

QWEN_MODEL_ID    = "Qwen/Qwen3.6-27B"    # Tier1c — Qwen3.6 27B VLM (Qwen3_5ForConditionalGeneration)
LLAVA16_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"    # Tier1a — Mistral-7B, no medical FT
LLAVA16_LLAMA_ID = "llava-hf/llava-v1.6-vicuna-13b-hf"    # Tier1b — LLaMA-2-13B, no medical FT
RETINAVLM_REPO   = "RobbieHolland/RetinaVLM"
RETINAVLM_SUBFOLDER = "RetinaVLM-Specialist"
# Pre-built dequantized (fp16) checkpoint produced by
# SpecialistVLMs/load_method1_save_dequantized.py (~16GB). RetinaVLM's HF repo
# stores int8-quantized weights in a subfolder, so the standard
# RetinaVLM.from_pretrained(repo) path does NOT work; we build an empty shell
# (base Llama3 + ResNet encoder) and load this dequantized state_dict instead.
RETINAVLM_DEQUANT_CKPT = os.path.join(
    SPECIALIST_ROOT, "saved_models", "RetinaVLM-Specialist-Dequantized", "model.pt"
)


class VLMBackend(ABC):
    tier: str = ""
    name: str = ""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> "VLMBackend":
        ...

    @abstractmethod
    def generate(self, image, prompt: str, max_new_tokens: int = 200) -> str:
        ...

    def attention(self, image, prompt: str):
        raise NotImplementedError(f"{self.name}: attention not implemented")

    def class_logits(self, image, prompt: str, class_words):
        raise NotImplementedError(f"{self.name}: class_logits not implemented")


# ---------------------------------------------------------------------------
# Tier 1a — LLaVA-v1.6-mistral-7b  (generalist, SAME Mistral-7B backbone as LLaVA-Med)
# This creates a PAIRED comparison with Tier2 to isolate catastrophic forgetting:
#   Tier1a (generalist Mistral) → Tier2 (medical-FT Mistral) = pure forgetting measurement
# ---------------------------------------------------------------------------
class LLaVA16Backend(VLMBackend):
    tier, name = "tier1a", "LLaVA-v1.6-mistral-7b"

    def load(self):
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self.processor = LlavaNextProcessor.from_pretrained(LLAVA16_MODEL_ID)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            LLAVA16_MODEL_ID, torch_dtype=torch.float16, device_map=self.device
        ).eval()
        return self

    def generate(self, image, prompt: str, max_new_tokens: int = 200) -> str:
        import torch

        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Tier 1b — Qwen3-VL-8B-Instruct  (generalist, SOTA instruction follower)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tier 1b — LLaVA-v1.6-vicuna-13b  (generalist, LLaMA-2-13B backbone)
# Matched backbone with RetinaVLM's LLaMA family — isolates OCT FT forgetting.
# ---------------------------------------------------------------------------
class LLaVA16LlamaBackend(VLMBackend):
    tier, name = "tier1b", "LLaVA-v1.6-vicuna-13b"

    def load(self):
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        self.processor = LlavaNextProcessor.from_pretrained(LLAVA16_LLAMA_ID)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            LLAVA16_LLAMA_ID, torch_dtype=torch.float16, device_map=self.device
        ).eval()
        return self

    def generate(self, image, prompt: str, max_new_tokens: int = 200) -> str:
        import torch
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Tier 1c — Qwen3.6-27B (generalist, Qwen MoE, SOTA large model)
# ---------------------------------------------------------------------------
class QwenVLBackend(VLMBackend):
    tier, name = "tier1c", "Qwen3.6-27B-Instruct"

    def load(self):
        import torch
        from transformers import AutoProcessor

        # Qwen3.6-27B uses Qwen3_5ForConditionalGeneration architecture.
        # Try specific class first, then fall back to AutoModel.
        try:
            from transformers import Qwen3_5ForConditionalGeneration as _Model
        except ImportError:
            try:
                from transformers import Qwen3VLForConditionalGeneration as _Model
            except ImportError:
                from transformers import AutoModelForImageTextToText as _Model

        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        # attn_implementation must be set at load time — post-hoc patching does NOT work.
        # Use 'sdpa' for normal inference (fast), 'eager' for attention extraction (E3c).
        attn_impl = getattr(self, '_attn_impl', 'sdpa')
        self.model = _Model.from_pretrained(
            QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map=self.device,
            attn_implementation=attn_impl,
        ).eval()
        return self

    def load_for_attention(self) -> "QwenVLBackend":
        """Load with eager attention for output_attentions=True support (E3c/E3d XAI)."""
        self._attn_impl = "eager"
        return self.load()

    def _messages(self, image, prompt):
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

    def generate(self, image, prompt: str, max_new_tokens: int = 200) -> str:
        import torch

        msgs = self._messages(image, prompt)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Tier 2 - LLaVA-Med v1.5 (microsoft/llava-med-v1.5-mistral-7b)
# ---------------------------------------------------------------------------
LLAVAMED_SOURCE = os.path.join(SPECIALIST_ROOT, "LLaVA-Med")
LLAVAMED_WEIGHTS = os.path.join(
    SPECIALIST_ROOT, "saved_models", "LLaVA-Med-v1.5"
)

class LLaVAMedBackend(VLMBackend):
    tier, name = "tier2", "LLaVA-Med-v1.5"

    def load(self):
        import torch

        # LLaVA-Med v1.5 uses LlavaMistralForCausalLM (custom arch in the cloned source).
        # We load it directly without the pytorch-lightning LLavaMed wrapper.
        if LLAVAMED_SOURCE not in sys.path:
            sys.path.insert(0, LLAVAMED_SOURCE)
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM  # noqa
        from transformers import AutoTokenizer, CLIPImageProcessor

        self.model = LlavaMistralForCausalLM.from_pretrained(
            LLAVAMED_WEIGHTS, torch_dtype=torch.float16, device_map=self.device
        ).eval()
        # tokenizer.json exported via oct_llm env (older transformers) to avoid
        # transformers 5.9 TikTokenConverter misfiring on the SentencePiece model.
        self.tokenizer = AutoTokenizer.from_pretrained(LLAVAMED_WEIGHTS)
        mm_vision_tower = getattr(self.model.config, "mm_vision_tower",
                                  "openai/clip-vit-large-patch14-336")
        self.image_processor = CLIPImageProcessor.from_pretrained(mm_vision_tower)

        # Initialise vision tower (required for LLaVA-Med to process images)
        if hasattr(self.model, "get_model"):
            vt = self.model.get_model().vision_tower
            if isinstance(vt, (list, tuple)):
                vt = vt[0]
            if hasattr(vt, "load_model") and not getattr(vt, "is_loaded", True):
                vt.load_model()
            vt.to(device=self.device, dtype=torch.float16)
        return self

    def _preprocess(self, image):
        """PIL image → CLIP pixel_values tensor [1, C, H, W] float16."""
        import torch
        import numpy as np
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        pv = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        return pv.to(self.device, dtype=torch.float16)

    def generate(self, image, prompt: str, max_new_tokens: int = 200) -> str:
        import torch
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        pixel_values = self._preprocess(image)

        # Build prompt with image token placeholder
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids,
                images=pixel_values,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        # LLaVA-Med's fork generates from inputs_embeds, so recent transformers return
        # ONLY the newly generated tokens (out_ids shorter than the text input_ids).
        # Slicing by input length would then drop everything -> decode the full output.
        gen = out_ids if out_ids.shape[1] <= input_ids.shape[1] else out_ids[:, input_ids.shape[1]:]
        return self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Tier 3 - RetinaVLM-Specialist
# ---------------------------------------------------------------------------
class RetinaVLMBackend(VLMBackend):
    tier, name = "tier3", "RetinaVLM-Specialist"

    def load(self):
        import importlib
        import torch

        # SpecialistVLMs imports are relative (`from run...`, `from models...`) and
        # its hydra config is resolved relative to SPECIALIST_ROOT, so we run with
        # that as cwd + at the FRONT of sys.path. CAUTION: this project also has a
        # top-level module named `models` (this file). It is already imported as the
        # `models` entry in sys.modules, which would shadow SpecialistVLMs' `models/`
        # PACKAGE. We temporarily evict the colliding cached modules and force
        # SPECIALIST_ROOT first, then restore everything in `finally`.
        cwd = os.getcwd()
        saved_path = list(sys.path)
        saved_modules = {
            k: sys.modules[k]
            for k in list(sys.modules)
            if k == "models" or k.startswith("models.") or k == "run" or k.startswith("run.")
        }
        os.chdir(SPECIALIST_ROOT)
        sys.path.insert(0, SPECIALIST_ROOT)
        for k in saved_modules:
            del sys.modules[k]
        try:
            import importlib.util
            import types
            from omegaconf import OmegaConf  # noqa: F401
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            # SpecialistVLMs' `models/` and `run/` have NO __init__.py (implicit
            # namespace packages). With this project's `models.py` having occupied
            # the `models` name, importlib's namespace resolution misfires
            # ("'models' is not a package"). Register them explicitly as namespace
            # packages rooted at SPECIALIST_ROOT so internal `from run...`/
            # `from models...` imports resolve.
            for pkg in ("models", "run"):
                pkg_dir = os.path.join(SPECIALIST_ROOT, pkg)
                m = types.ModuleType(pkg)
                m.__path__ = [pkg_dir]
                m.__package__ = pkg
                sys.modules[pkg] = m

            spec = importlib.util.spec_from_file_location(
                "models.retinavlm_wrapper",
                os.path.join(SPECIALIST_ROOT, "models", "retinavlm_wrapper.py"),
            )
            rvlm_mod = importlib.util.module_from_spec(spec)
            sys.modules["models.retinavlm_wrapper"] = rvlm_mod
            spec.loader.exec_module(rvlm_mod)
            RetinaVLM, RetinaVLMConfig = rvlm_mod.RetinaVLM, rvlm_mod.RetinaVLMConfig

            config_dir = os.path.join(SPECIALIST_ROOT, "configs")
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                config = compose(config_name="default")

            rvlm_config = RetinaVLMConfig.from_pretrained(
                RETINAVLM_REPO, subfolder=RETINAVLM_SUBFOLDER
            )
            rvlm_config.update(config)
            rvlm_config.model.checkpoint_path = None  # weights come from dequant ckpt

            # Build the empty shell (loads base Llama3 + blank encoder/adapter)...
            self.model = RetinaVLM(rvlm_config)
            # ...then overwrite with the pre-dequantized fp16 specialist weights.
            state_dict = torch.load(RETINAVLM_DEQUANT_CKPT, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            del state_dict
            if missing:
                print(f"[RetinaVLM] load_state_dict missing={len(missing)} "
                      f"unexpected={len(unexpected)}")
            self.model = self.model.to(self.device).eval()
        finally:
            os.chdir(cwd)
            sys.path[:] = saved_path
            # Drop SpecialistVLMs' `models`/`run` packages and restore this project's.
            for k in [k for k in list(sys.modules)
                      if k == "models" or k.startswith("models.")
                      or k == "run" or k.startswith("run.")]:
                del sys.modules[k]
            sys.modules.update(saved_modules)

        self._inner = getattr(self.model, "model", self.model)  # MiniGPT4 module
        return self

    @staticmethod
    def _to_gray(image):
        """RetinaVLM's ResNet vision encoder expects a SINGLE-channel (grayscale)
        OCT B-scan; data.representative_pre_bscan returns RGB PIL. Convert to a 2D
        grayscale numpy array (matching the reference notebook's
        np.array(Image.open(..).convert('L')))."""
        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            return np.array(image.convert("L"))
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):  # H,W,C -> luminosity
            arr = (arr[..., :3] @ np.array([0.299, 0.587, 0.114])).astype(arr.dtype)
        return arr

    def generate(self, image, prompt: str, max_new_tokens: int = 200) -> str:
        # RetinaVLM.forward() handles np -> normalized 192x192 tensor + dtype/device
        # match, then calls the inner MiniGPT4.query(). Image must be single-channel.
        import torch

        img = self._to_gray(image)
        with torch.no_grad():
            out = self.model.forward([img], [prompt], max_new_tokens=max_new_tokens)
        return out[0] if isinstance(out, (list, tuple)) else str(out)

    def _img_tensor(self, image):
        """Convert a single PIL/np image to the model's normalized device tensor
        (single-channel, as the ResNet encoder requires)."""
        import torch

        param = next(self.model.model.parameters())
        t = self.model.convert_any_image_to_normalized_tensor(self._to_gray(image))
        return torch.stack([t], dim=0).to(device=param.device, dtype=param.dtype)

    def attention(self, image, prompt: str):
        """Full attention bundle from mini_gpt4.attention (L392):
        (samples, inputs_tokens, subsequence_indices, sequence_attentions, image_attention)."""
        return self._inner.attention(self._img_tensor(image), [prompt])

    def class_logits(self, image, prompt: str, class_words):
        """Token probabilities for keyword-conditioned saliency / CI logit
        (mini_gpt4.softmax_logits, L331)."""
        return self._inner.softmax_logits(self._img_tensor(image), texts=[prompt])


# ---------------------------------------------------------------------------
_REGISTRY = {
    "tier1a": LLaVA16Backend,         # generalist, Mistral-7B  (paired with LLaVA-Med → forgetting)
    "tier1b": LLaVA16LlamaBackend,    # generalist, LLaMA-2-13B (LLaMA family, paired with RetinaVLM)
    "tier1c": QwenVLBackend,          # generalist, Qwen3.6-27B MoE (SOTA)
    "tier1":  QwenVLBackend,          # alias
    "tier2":  LLaVAMedBackend,        # medical-general, Mistral-7B
    "tier3":  RetinaVLMBackend,       # OCT-specialist, LLaMA-3-8B
}


def get_backend(tier: str, device: str = "cuda") -> VLMBackend:
    if tier not in _REGISTRY:
        raise ValueError(f"unknown tier {tier!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[tier](device=device)


if __name__ == "__main__":
    # Import-only smoke test (no weights loaded).
    for t in _REGISTRY:
        b = get_backend(t, device="cpu")
        print(f"{t}: {b.name} (loaded={b.model is not None})")
