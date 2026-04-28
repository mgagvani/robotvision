'''
NOTE: This must be run in a GPU env with sglang installed.

Command for Illinois NCSA:

```
export MODEL_ROOT=/work/nvme/bgxf/mgagvani/hfcache
  export HF_HOME=$MODEL_ROOT/huggingface
  export HF_HUB_CACHE=$HF_HOME/hub
  export TRANSFORMERS_CACHE=$HF_HUB_CACHE
  export PATH=/work/nvme/bgxf/mgagvani/conda/envs/robotvision/bin:$PATH
  CUDA_VISIBLE_DEVICES=0 /work/nvme/bgxf/mgagvani/conda/envs/robotvision/bin/python -m sglang.launch_server \
    --model-path "$MODEL_ROOT/checkpoints/Qwen3.5-35B-A3B-GPTQ-Int4" \
    --served-model-name qwen3.5-35b-a3b \
    --host 127.0.0.1 \
    --port 8000 \
    --tp-size 1 \
    --mem-fraction-static 0.80 \
    --context-length 8192 \
    --dtype float16 \
    --disable-cuda-graph \
    --reasoning-parser qwen3
```

Note that --tp-size 1 is for 1 GPU, if N gpus set to N. 
'''


from transformers import pipeline
from ultralytics import YOLO
from loader import WaymoE2E
import argparse
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

import os
import dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from diffusers import QwenImageEditPipeline
import torch


PROMPT = '''
Write a prompt for an image editing model to edit one of three things: 
1) The color of traffic lights in the image
2) The presence of stop signs in the image
3) The presence of pedestrians in the image.

E.g., if there are no traffic lights, do not write a prompt involving traffic lights. 
If there is, ask to change the color of the traffic light. 
If there is no stop sign, ask to add one, or vice versa. 
If there are no pedestrians, add some, and vice versa. 
If no condition is met, output NO CHANGE. Your prompt should be two sentences.
'''

def load_yolo(model_path: str = "yolo26x.pt"):
    # Load a model
    model = YOLO(model_path)
    return model

def inference_yolo(model, images):
    # Predict with the model
    results = model(images, verbose=False, device="cuda:0")  # predict on an image

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

    return results

def generate_gemini(scene_description: str):
    # Load Gemini API key from .env file
    dotenv.load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or export it before running.")
    genai_client = genai.Client(api_key=api_key)

    # Generate prompt for image editing model
    response = genai_client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=scene_description)],
            ),
        ],
        config=types.GenerateContentConfig(
            system_instruction=PROMPT,
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        ),
    )

    return response.text

def generate_sglang(scene_description: str):
    client = OpenAI(
        base_url=os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("SGLANG_API_KEY", "EMPTY"),
    )
    response = client.chat.completions.create(
        model=os.getenv("SGLANG_MODEL", "qwen3.5-35b-a3b"),
        messages=[
            {"role": "system", "content": PROMPT.strip()},
            {"role": "user", "content": scene_description},
        ],
        max_tokens=512,
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content.strip()

def load_qwen_image_edit(id: str = "Qwen/Qwen-Image-Edit"):
    pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
    print("qwen image edit pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)


def generate_qwen_image(pipeline, image: Image.Image, prompt: str):
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ", # TODO: add negative prompt!
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]

    return output_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Waymo directory")
    parser.add_argument(
        "--generator",
        type=str,
        choices=["sglang", "gemini"],
        default="sglang",
        help="Prompt generator backend. Assumes SGLang is already running for --generator sglang.",
    )
    args = parser.parse_args()

    model = load_yolo()

    test_dataset = WaymoE2E(
        indexFile="index_val.pkl", data_dir=args.data_dir, n_items=5_000
    )

    # Qwen Image Edit
    if not os.environ.get("HF_HOME") and os.environ.get("HF_HOME").startswith("/work/nvme/bgxf"):
        raise RuntimeError("Set HF_HOME to not home bruh")
    qwen_pipeline = load_qwen_image_edit()

    images = []
    for i in range(10):
        sample = test_dataset[i]
        jpeg = sample["IMAGES_JPEG"][1] # front cam
        if hasattr(jpeg, 'numpy'):
            jpeg = jpeg.numpy().tobytes()
        elif hasattr(jpeg, 'tobytes'):
            jpeg = jpeg.tobytes()
        image = Image.open(io.BytesIO(jpeg))
        images.append(image)
    
    
    results = inference_yolo(model, images)

    for i, result in enumerate(results):
        # Generate string describing scene
        # e.g "Car at (x1, y1) -> (x2, y2)\nPedestrian at (x1, y1) -> (x2, y2)\n..."
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        confs = result.boxes.conf
        xyxy = result.boxes.xyxy
        scene_description = ""
        for j, (name, conf) in enumerate(zip(names, confs)):
            scene_description += f"{name} at {xyxy[j].tolist()}\n"
        print(scene_description)

        # Save the image that corresponds to this YOLO result.
        images[i].save(f"image_{i}.jpg")
        if args.generator == "sglang":
            prompt = generate_sglang(scene_description)
        else:
            prompt = generate_gemini(scene_description)
        print(f"Prompt for image {i}: {prompt}")

        # Generate edited image with prompt
        edited_image = generate_qwen_image(qwen_pipeline, images[i], prompt)
        edited_image.save(f"edited_image_{i}.jpg")
