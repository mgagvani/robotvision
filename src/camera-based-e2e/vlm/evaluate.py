import os, json
import numpy as np
import torch
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader


from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from loader import WaymoE2E
os.environ["HF_HOME"] = "/scratch/gilbreth/mgagvani/hfcache/"


def predict_waypoints_json(model, front_img: Image.Image, intent: int, past: np.ndarray):
    # todo: make it so we don't have to copy paste this between finetune/evaluate.
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

    prompt = processor.apply_chat_template([user_msg], add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=[[front_img]],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=512,
        )

    # Decode only the newly generated part (strip the prompt)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # input(text + "\n\nPress Enter to continue...")

    # Parse JSON (be strict; driving eval should fail loudly)
    pred = json.loads(text)
    wp_cm = np.array(pred["waypoints_cm"], dtype=np.float32)  # [20,2] in cm
    wp_m = wp_cm / 100.0
    print(wp_m)
    return wp_m

def ade_fde(pred_xy: np.ndarray, gt_xy: np.ndarray):
    # shapes: [T,2]
    d = np.linalg.norm(pred_xy - gt_xy, axis=1)
    ade = float(d.mean())
    fde = float(d[-1])
    return ade, fde

def eval_model(model, dataset, num_samples=1000):
    loader = DataLoader(dataset, batch_size=1, num_workers=4)
    ades, fdes = [], []

    for i, ex in enumerate(loader):
        if i >= num_samples:
            break

        # ex is batched dict (batch=1); unwrap
        intent = int(ex["INTENT"][0])
        past = ex["PAST"][0].numpy()
        gt_future = ex["FUTURE"][0].numpy()

        # images: list inside batch; depending on your dataset/collation,
        # you may need to unwrap differently.
        # If your dataset returns IMAGES as list[Tensor CHW], DataLoader will nest it.
        img_list = ex["IMAGES"]           # may be a list-like
        front_chw = img_list[1][0]            # confirm front index; drop batch dim
        front_img = Image.fromarray(front_chw.permute(1,2,0).numpy())

        try:
            pred_future = predict_waypoints_json(model, front_img, intent, past)
            ade, fde = ade_fde(pred_future, gt_future)

            ades.append(ade)
            fdes.append(fde)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    return {
        "ADE": float(np.mean(ades)),
        "FDE": float(np.mean(fdes)),
        "N": len(ades),
    }

if __name__ == "__main__":
    # 3 evals
    # Base 1 - Qwen/Qwen3-VL-2B-Instruct
    # Base 2 - nvidia/Cosmos-Reason2-2B
    # Finetune (on cosmos reason 2 2b)

    # base 1/2 cannot even generate valid json :(

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "nvidia/Cosmos-Reason2-2B",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        _attn_implementation="sdpa",
    )
    base_model.eval()

    ADAPTER_DIR = "/scratch/gilbreth/mgagvani/robotvision_scratch/Cosmos-Reason2-2B_12-27-19-12"
    lora_model = PeftModel.from_pretrained(
        base_model,  # start from base
        ADAPTER_DIR,  # load LoRA weights
        is_trainable=False,
    )
    lora_model.eval()

    processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")

    DATA_DIR = "/scratch/gilbreth/mgagvani/wod/waymo_open_dataset_end_to_end_camera_v_1_0_0/" 
    dataset = WaymoE2E(
        indexFile="index_val.pkl",
        data_dir=DATA_DIR,
        n_items=50, 
        seed=42,
    )

    results = eval_model(lora_model, dataset, num_samples=500)
    print(results)
