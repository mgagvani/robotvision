'''
NOTE: This must be run in a GPU env with sglang installed.

Command for Illinois NCSA Delta GPU:
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

NCSA Delta AI:
```
export SGLANG_MAMBA_CONV_DTYPE=float16
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODEL_ROOT=/work/nvme/bgxf/mgagvani/hfcache
export HF_HOME=$MODEL_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
CUDA_VISIBLE_DEVICES=0 /u/mgagvani/robotvision/.venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL_ROOT/checkpoints/Qwen3.5-35B-A3B-GPTQ-Int4" \
      --served-model-name qwen3.5-35b-a3b \
      --host 127.0.0.1 \
      --port 8000 \
      --tp-size 1 \
      --mem-fraction-static 0.80 \
      --context-length 8192 \
      --dtype float16 \
      --mamba-ssm-dtype float16 \
      --linear-attn-prefill-backend triton \
      --linear-attn-backend flashinfer \
      --disable-cuda-graph \
      --reasoning-parser qwen3
```

```
export VLLM_DEEP_GEMM_WARMUP=skip
export MODEL_ROOT=/work/nvme/bgxf/mgagvani/hfcache
export HF_HOME=$MODEL_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
CUDA_VISIBLE_DEVICES=0 /u/mgagvani/robotvision/.venv/bin/vllm serve \
  "$MODEL_ROOT/checkpoints/Qwen3.5-4B" \
  --served-model-name qwen3.5-4b \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 64 \
  --max-model-len 8192 \
  --dtype float16 \
  --limit-mm-per-prompt '{"image": 1}'
  --reasoning-parser qwen3
```

Note that --tp-size 1 is for 1 GPU, if N gpus set to N. 
'''


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
import base64

from diffusers import QwenImageEditPipeline, Flux2KleinPipeline, QwenImageEditPlusPipeline
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


PROMPT = '''
Write a prompt for an image editing model to edit one of three things: 
1) The color of traffic lights in the image
2) The presence of stop signs in the image
3) The presence of pedestrians in the image.

E.g., if there are no traffic lights, do not write a prompt involving traffic lights. 
If there is, ask to change the color of the traffic light following the rule "RED -> GREEN, GREEN -> RED, YELLOW -> GREEN".
If there is no stop sign, ask to add one, or vice versa, only if at an intersection.
If there are no pedestrians, add some, and vice versa, only if there is a sidewalk or crosswalk visible.
If no condition is met, output NO CHANGE. Your prompt should be visually descriptive, at most 3 sentences.
Do not include unnecessary detail about tone/style, as we want the new image to be as similar to the original as possible aside from the specified edit.
Do not mention bounding box coordinates in the prompt, but you can reference relative positions of objects.
Make sure to mention "Edit the image minimally. Preserve original composition, lighting, texture, and all unrelated details."
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

def image_to_url(image: Image.Image) -> str:
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def generate_local(scene_description: str, image: Image.Image | None = None):
    client = OpenAI(
        base_url=os.getenv("BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("API_KEY", "EMPTY"),
    )
    user_content = [{"type": "text", "text": scene_description}]
    if image is not None:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_url(image),
            },
        })
    messages = [
        {
            "role": "system",
            "content": PROMPT.strip(),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    response = client.chat.completions.create(
        model=os.getenv("MODEL", "qwen3.5-4b"),
        messages=messages,
        max_tokens=512,
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content.strip()

def load_qwen_image_edit():
    pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)

    return pipeline

def generate_image_edit(pipeline, image, prompt):
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]

    return output_image

def load_flux2klein():
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9b-kv",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=None)
    return pipe

def generate_flux2klein_edit(pipe, image, prompt):
    out = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=4,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]
    return out

def load_firered():
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "FireRedTeam/FireRed-Image-Edit-1.1",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=None)
    return pipe

def generate_firered_edit(pipe, image, prompt):
    # image must be a list
    if not isinstance(image, list):
        image = [image]
    out = pipe(
        prompt=prompt,
        negative_prompt=" ", # TODO: negative prompt
        image=image,
        num_inference_steps=40,
        true_cfg_scale=3.0,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Waymo directory")
    parser.add_argument(
        "--generator",
        type=str,
        choices=["local", "gemini"],
        default="local",
        help="Prompt generator backend. Assumes local LLM is already running for --generator local.",
    )
    args = parser.parse_args()

    model = load_yolo()

    test_dataset = WaymoE2E(
        indexFile="index_val.pkl", data_dir=args.data_dir, n_items=5_000
    )

    # Load  image edit
    if not os.getenv("HF_HOME") and not os.getenv("HF_HOME").startswith("/work/nvme/bgxf"):
        raise RuntimeError("Set HF_HOME to correct dir")
    pipeline = load_firered()

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

        # Generate image edit prompt
        if args.generator == "gemini":
            prompt = generate_gemini(scene_description)
        elif args.generator == "local":
            prompt = generate_local(scene_description, images[i])
        else:
            prompt = generate_gemini(scene_description)
        print(f"Prompt for image {i}: {prompt}")

        
        # Generate edited image
        if prompt != "NO CHANGE":
            edited_image = generate_firered_edit(pipeline, images[i], prompt)
            edited_image.save(f"edited_image_{i}.jpg")
        else: 
            print(f"No change for image {i}, skipping edit.")