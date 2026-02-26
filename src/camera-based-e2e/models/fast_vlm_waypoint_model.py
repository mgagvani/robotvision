"""
FastVLM-based waypoint prediction model.
Uses Apple's FastVLM (0.5B) via LlavaQwen2 - works with Python 3.9 and current transformers.
Same interface as FullVLMWaypointModel: full VLM forward, last-token pooling, MLP head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

# Apple's FastVLM uses llava_qwen2 (trust_remote_code); no transformers upgrade needed
DEFAULT_FASTVLM = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # Apple's placeholder token id

# LoRA target modules for Qwen2 decoder (attention only by default)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class FastVLMWaypointModel(nn.Module):
    """
    Waypoint prediction using Apple FastVLM (0.5B).
    Loads via AutoModelForCausalLM + trust_remote_code (LlavaQwen2).
    Optional LoRA on the language model decoder (use_lora=True).
    """

    # Indices for front-facing cameras (FRONT_LEFT, FRONT, FRONT_RIGHT) in Waymo E2E image list
    FRONT_CAMERA_INDICES = (0, 1, 2)
    FRONT_CAMERA_NAMES = ("front-left", "front", "front-right")  # for camera-aware prompts

    def __init__(
        self,
        model_name=DEFAULT_FASTVLM,
        in_dim=96,
        out_dim=40,
        hidden_dim=512,
        prompt_mode="fixed",
        fixed_prompt="Predict the driving trajectory for the vehicle.",
        use_lora=False,
        lora_r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        use_all_front_cameras=False,
    ):
        super().__init__()
        print(f"Loading FastVLM: {model_name}")
        print(f"Prompt mode: {prompt_mode}")
        self.use_lora = use_lora
        self.use_all_front_cameras = use_all_front_cameras
        if use_all_front_cameras:
            print("Using all front cameras (FRONT_LEFT, FRONT, FRONT_RIGHT) for VLM.")

        self.prompt_mode = prompt_mode
        self.fixed_prompt = fixed_prompt
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.vlm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "LoRA requires the 'peft' package. Install it in the robotvision env: pip install peft"
                ) from None
            if hasattr(self.vlm, "enable_input_require_grads"):
                self.vlm.enable_input_require_grads()

            # Restrict LoRA to last 8 decoder layers. PEFT regex is .*\.{layers_pattern}\.(\d+)\.
            # LlavaQwen2 has decoder at model.model.layers.<i>, so pattern "layers" (no dot) matches.
            num_layers = getattr(self.vlm.config, "num_hidden_layers", None)
            layers_to_transform = None
            layers_pattern = None
            if num_layers is not None and num_layers >= 8:
                last_n = 8
                layers_to_transform = list(range(num_layers - last_n, num_layers))
                layers_pattern = "layers"

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
                layers_to_transform=layers_to_transform,
                layers_pattern=layers_pattern,
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
        else:
            for param in self.vlm.parameters():
                param.requires_grad = False
            self.vlm.eval()

        # Apple FastVLM-0.5B hidden_size from config
        self.vlm_dim = getattr(self.vlm.config, "hidden_size", 896)
        print(f"VLM LLM hidden size: {self.vlm_dim}")

        num_cameras = 3 if use_all_front_cameras else 1
        vlm_feat_dim = num_cameras * self.vlm_dim
        if prompt_mode == "dynamic":
            input_dim = vlm_feat_dim + in_dim
        else:
            input_dim = vlm_feat_dim + in_dim + 3
        print(f"MLP input dimension: {input_dim} ({num_cameras} camera(s))")

        self.waypoint_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        vlm_params = sum(p.numel() for p in self.vlm.parameters())
        head_params = sum(p.numel() for p in self.waypoint_head.parameters())
        print(f"Model parameters:")
        if use_lora:
            print(f"  - VLM (base frozen, LoRA trainable): {vlm_params:,}")
            print(f"  - Trainable (LoRA + head): {trainable:,}")
        else:
            print(f"  - Frozen VLM: {vlm_params:,}")
            print(f"  - Trainable head: {head_params:,}")
        print(f"  - Total: {total:,}")
        print(f"  - Trainable %: {100 * trainable / total:.4f}%")

    def _get_prompt_strings(self, intents, batch_size, camera_name=None):
        """Prompt text (no image placeholder; we add <<image>> when building input).
        If camera_name is set (e.g. 'front-left', 'front', 'front-right'), prepend camera context
        so the VLM does not treat all views the same."""
        if self.prompt_mode == "dynamic":
            intent_map = {1: "left", 2: "right", 3: "straight"}
            base = [
                f"The driver intends to go {intent_map.get(int(i.item()), 'straight')}. Predict the driving trajectory."
                for i in intents
            ]
        else:
            base = [self.fixed_prompt] * batch_size
        if camera_name is not None:
            prefix = f"This is the {camera_name} camera view. "
            base = [prefix + s for s in base]
        return base

    def _batch_tensor_to_pil(self, images):
        """(B, 3, H, W) -> list of PIL Images."""
        if images.dtype == torch.uint8:
            images_np = images.cpu().numpy()
        else:
            images_np = (images.cpu().numpy() * 255).astype("uint8")
        return [Image.fromarray(img.transpose(1, 2, 0)) for img in images_np]

    def _build_input_ids_and_pixel_values(self, pil_images, prompt_strings, device):
        """
        Apple FastVLM: prompt has <<image>>, we split and splice IMAGE_TOKEN_INDEX.
        Batch: same prompt string for all, or list of prompts; images are list of PIL.
        """
        image_processor = self.vlm.get_vision_tower().image_processor
        pixel_values_list = []
        input_ids_list = []

        for i, (pil_img, text) in enumerate(zip(pil_images, prompt_strings)):
            # User message with image placeholder
            content = f"<<image>>\n{text}"
            messages = [{"role": "user", "content": content}]
            rendered = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # Rendered may use <image> or <<image>>; try both
            if "<<image>>" in rendered:
                pre, post = rendered.split("<<image>>", 1)
            elif "<image>" in rendered:
                pre, post = rendered.split("<image>", 1)
            else:
                raise ValueError("No image placeholder in chat template output")

            pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids[0]
            post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids[0]
            img_tok = torch.tensor([IMAGE_TOKEN_INDEX], dtype=pre_ids.dtype)
            ids = torch.cat([pre_ids, img_tok, post_ids], dim=0)
            input_ids_list.append(ids)

            px = image_processor(images=pil_img, return_tensors="pt")["pixel_values"]
            pixel_values_list.append(px)

        # Pad input_ids to same length
        max_len = max(ids.size(0) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded = []
        attention_mask = []
        for ids in input_ids_list:
            pad_len = max_len - ids.size(0)
            padded.append(F.pad(ids, (0, pad_len), value=pad_id))
            attn = torch.cat([torch.ones(ids.size(0), dtype=ids.dtype), torch.zeros(pad_len, dtype=ids.dtype)])
            attention_mask.append(attn)
        input_ids = torch.stack(padded).to(device)
        attention_mask = torch.stack(attention_mask).to(device)

        # Pixel values: (B, C, H, W) - stack if same size else pad/collate as model expects
        if all(p.shape == pixel_values_list[0].shape for p in pixel_values_list):
            pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)
        else:
            pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)

        return input_ids, attention_mask, pixel_values

    def extract_features(self, images, intents, camera_name=None):
        """
        Run FastVLM on image + prompt and return last-token hidden state.
        images: (B, 3, H, W), intents: (B,)
        camera_name: optional str (e.g. 'front-left', 'front', 'front-right') for camera-aware prompts.
        Returns: (B, vlm_dim) float32
        """
        batch_size = images.shape[0]
        device = images.device

        pil_images = self._batch_tensor_to_pil(images)
        prompt_strings = self._get_prompt_strings(intents, batch_size, camera_name=camera_name)

        input_ids, attention_mask, pixel_values = self._build_input_ids_and_pixel_values(
            pil_images, prompt_strings, device
        )

        # When using LoRA we need gradients; when frozen use no_grad
        # Apple FastVLM (LlavaQwen2) forward expects images=, not pixel_values=
        if self.use_lora:
            outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.vlm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
        last_hidden = outputs.hidden_states[-1]
        features = last_hidden[:, -1, :]

        return features.float()

    def forward(self, x: dict):
        past = x["PAST"]
        images = x["IMAGES"]
        intent = x["INTENT"]
        if self.use_all_front_cameras:
            # Run VLM on each front camera with camera-aware prompts; concatenate features (B, 3 * vlm_dim)
            feats = [
                self.extract_features(images[i], intent, camera_name=self.FRONT_CAMERA_NAMES[j])
                for j, i in enumerate(self.FRONT_CAMERA_INDICES)
            ]
            vlm_feat = torch.cat(feats, dim=1)
        else:
            front_cam = images[1]
            vlm_feat = self.extract_features(front_cam, intent)
        past_flat = past.reshape(past.size(0), -1)

        if self.prompt_mode == "dynamic":
            combined = torch.cat([vlm_feat, past_flat], dim=1)
        else:
            intent_onehot = F.one_hot((intent - 1).long(), num_classes=3).float()
            combined = torch.cat([vlm_feat, past_flat, intent_onehot], dim=1)

        return self.waypoint_head(combined)
