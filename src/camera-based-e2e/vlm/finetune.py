import os, json
import numpy as np
import torch
from datetime import datetime
from PIL import Image

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
import wandb

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from loader import WaymoE2E

USE_LORA = True

os.environ["HF_HOME"] = "/scratch/gilbreth/$USER/hfcache/"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

model_id = "nvidia/Cosmos-Reason2-2B"  # IMPORTANT: use a real Qwen3-VL checkpoint
OUT_DIR = f"/scratch/gilbreth/$USER/robotvision_scratch/{model_id.split('/')[-1]}_{datetime.now().strftime('%m-%d-%H-%M')}"

wandb.init(project="robotvision", name=OUT_DIR[47:])


processor = AutoProcessor.from_pretrained(model_id)

if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

def chw_to_pil(chw_uint8: torch.Tensor) -> Image.Image:
    hwc = chw_uint8.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(hwc)

def waypoints_to_json_cm(future_xy: np.ndarray) -> str:
    wp_cm = np.rint(future_xy * 100.0).astype(int).tolist()
    return json.dumps({"hz": 4, "waypoints_cm": wp_cm}, separators=(",", ":"))

def collate_waymo(examples):
    texts_full, texts_prompt, images_batch = [], [], []

    for ex in examples:
        img_list = ex["IMAGES"]
        if img_list is None or len(img_list) == 0:
            raise ValueError("No images in example; set images=True in dataset.")
        front_img = chw_to_pil(img_list[1])

        intent = int(ex["INTENT"])
        past = ex["PAST"]
        future = ex["FUTURE"]

        # Ensure numpy arrays
        if isinstance(past, torch.Tensor):
            past = past.cpu().numpy()
        if isinstance(future, torch.Tensor):
            future = future.cpu().numpy()

        target = waypoints_to_json_cm(future)

        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You are a driving planner. Given the front camera image, past ego states, "
                    "and high-level intent, predict the future 5s trajectory as 20 waypoints "
                    "at 4Hz in JSON."
                )},
                {"type": "image"},
                {"type": "text", "text": f"intent={intent}\npast_states={past.tolist()}\nReturn JSON only."},
            ],
        }
        assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": target}]}

        prompt_text = processor.apply_chat_template([user_msg], add_generation_prompt=True)
        full_text = processor.apply_chat_template([user_msg, assistant_msg], add_generation_prompt=False)

        texts_prompt.append(prompt_text.strip())
        texts_full.append(full_text.strip())
        images_batch.append([front_img])

    batch = processor(text=texts_full, images=images_batch, return_tensors="pt", padding=True)

    # prompt lengths for masking
    prompt_tok = processor(text=texts_prompt, images=images_batch, return_tensors="pt", padding=True)
    prompt_lens = prompt_tok["attention_mask"].sum(dim=1)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for i, L in enumerate(prompt_lens.tolist()):
        labels[i, :L] = -100  # only train on assistant tokens

    batch["labels"] = labels
    if (batch["labels"] != -100).sum().item() == 0:
        raise RuntimeError("All labels are -100 (nothing to learn). Check prompt/label masking.")
    return batch

# ---- load model ----
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    _attn_implementation="sdpa",  # make it flash_attention_2 for a100/h100
)
model.enable_input_require_grads()

# Freeze vision encoder initially (recommended for first run)
# unfreeze it :)
# for p in model.model.visual.parameters():
#     p.requires_grad = True

if USE_LORA:
    lora_config = LoraConfig(
        r=128,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

# ---- dataset ----
DATA_DIR = "/scratch/gilbreth/$USER/wod/waymo_open_dataset_end_to_end_camera_v_1_0_0/"
train_dataset = WaymoE2E(
    indexFile="index_train.pkl",
    data_dir=DATA_DIR,
    images=True,
    n_items=2500, 
    seed=42,
)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataloader_num_workers=12,
    dataloader_pin_memory=True,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_waymo,
)

print("Any trainable params?", any(p.requires_grad for p in model.parameters()))
trainer.train()
trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)
