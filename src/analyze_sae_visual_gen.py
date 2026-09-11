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


import argparse
import base64
import io
import json
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline
from openai import OpenAI
from PIL import Image
from ultralytics import YOLO

from loader import WaymoE2E
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


PROMPT = '''
Choose one minimal counterfactual edit for an image editing model from one of three things:
1) The color of traffic lights in the image
2) The presence of stop signs in the image
3) The presence of pedestrians in the image.

E.g., if there are no traffic lights, do not write a prompt involving traffic lights. 
If there is, ask to change the color of the traffic light following the rule "RED -> GREEN, GREEN -> RED, YELLOW -> GREEN".
If there is no stop sign, ask to add one, or vice versa, only if at an intersection.
If there are no pedestrians, add some, and vice versa, only if there is a sidewalk or crosswalk visible.
If no condition is met, set edit_type and edit_direction to "no_change" and prompt to "NO CHANGE".
The prompt should be visually descriptive, at most 3 sentences.
Do not include unnecessary detail about tone/style, as we want the new image to be as similar to the original as possible aside from the specified edit.
Do not mention bounding box coordinates in the prompt, but you can reference relative positions of objects.
Make sure to mention "Edit the image minimally. Preserve original composition, lighting, texture, and all unrelated details."

Return strict JSON only, with no markdown or extra text:
{
  "edit_type": "traffic_light_color | traffic_light_presence | pedestrian_presence | stop_sign_presence_location | no_change",
  "edit_direction": "red_to_green | green_to_red | yellow_to_green | add_traffic_light | remove_traffic_light | add_pedestrian | remove_pedestrian | add_stop_sign | remove_stop_sign | move_stop_sign | no_change",
  "prompt": "image editing prompt or NO CHANGE"
}
'''

VALID_EDIT_TYPES = {
    "traffic_light_color",
    "traffic_light_presence",
    "pedestrian_presence",
    "stop_sign_presence_location",
    "no_change",
}

VALID_EDIT_DIRECTIONS = {
    "red_to_green",
    "green_to_red",
    "yellow_to_green",
    "add_traffic_light",
    "remove_traffic_light",
    "add_pedestrian",
    "remove_pedestrian",
    "add_stop_sign",
    "remove_stop_sign",
    "move_stop_sign",
    "no_change",
}

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

def parse_edit_plan(raw_output: str) -> dict:
    raw = (raw_output or "").strip()
    if raw == "NO CHANGE":
        return {
            "edit_type": "no_change",
            "edit_direction": "no_change",
            "prompt": "NO CHANGE",
            "raw_generator_output": raw_output,
        }

    cleaned = raw
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if not isinstance(data, dict):
        return {
            "edit_type": "no_change",
            "edit_direction": "no_change",
            "prompt": "NO CHANGE",
            "raw_generator_output": raw_output,
            "parse_error": "generator_output_was_not_valid_json",
        }

    edit_type = str(data.get("edit_type", "")).strip()
    edit_direction = str(data.get("edit_direction", "")).strip()
    prompt = str(data.get("prompt", "")).strip()

    if edit_type not in VALID_EDIT_TYPES or edit_direction not in VALID_EDIT_DIRECTIONS:
        return {
            "edit_type": "no_change",
            "edit_direction": "no_change",
            "prompt": "NO CHANGE",
            "raw_generator_output": raw_output,
            "parse_error": "invalid_edit_type_or_direction",
        }

    if edit_type == "no_change" or edit_direction == "no_change" or prompt == "NO CHANGE":
        edit_type = "no_change"
        edit_direction = "no_change"
        prompt = "NO CHANGE"
    elif "Edit the image minimally" not in prompt:
        prompt = (
            f"{prompt} Edit the image minimally. Preserve original composition, "
            "lighting, texture, and all unrelated details."
        )

    return {
        "edit_type": edit_type,
        "edit_direction": edit_direction,
        "prompt": prompt,
        "raw_generator_output": raw_output,
    }

def jpeg_tensor_to_image(jpeg) -> Image.Image:
    if hasattr(jpeg, "numpy"):
        jpeg = jpeg.numpy().tobytes()
    elif hasattr(jpeg, "tobytes"):
        jpeg = jpeg.tobytes()
    return Image.open(io.BytesIO(jpeg)).convert("RGB")

def yolo_scene_description(result) -> str:
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
    confs = result.boxes.conf
    xyxy = result.boxes.xyxy
    lines = []
    for j, (name, conf) in enumerate(zip(names, confs)):
        lines.append(f"{name} conf={float(conf.item()):.3f} at {xyxy[j].tolist()}")
    return "\n".join(lines)

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

def load_editor(name: str):
    if name == "firered":
        return load_firered()
    if name == "qwen":
        return load_qwen_image_edit()
    if name == "flux":
        return load_flux2klein()
    raise ValueError(f"Unknown editor: {name}")

def generate_edit_with_editor(editor_name: str, pipeline, image: Image.Image, prompt: str) -> Image.Image:
    if editor_name == "firered":
        return generate_firered_edit(pipeline, image, prompt)
    if editor_name == "qwen":
        return generate_image_edit(pipeline, image, prompt)
    if editor_name == "flux":
        return generate_flux2klein_edit(pipeline, image, prompt)
    raise ValueError(f"Unknown editor: {editor_name}")

RESUME_CONFIG_KEYS = ("index_file", "n_items", "camera_idx", "generator", "editor")

def load_resume_state(manifest_path: Path, output_dir: Path, args) -> tuple[list[dict], set[int]]:
    """Read an existing manifest and report which items need not be redone.

    Returns the rows worth keeping and the dataset indices they cover. Rows are
    dropped when they cannot be trusted: a truncated final line from a job that
    died mid-write, or an "edited" row whose image is no longer on disk.
    """
    if not manifest_path.exists():
        return [], set()

    kept: list[dict] = []
    completed: set[int] = set()
    with manifest_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A killed run leaves a partially written final line.
                print(f"WARNING: discarding malformed manifest line {lineno}; that item will be redone")
                continue

            for key in RESUME_CONFIG_KEYS:
                want = getattr(args, key)
                if row.get(key) != want:
                    raise ValueError(
                        f"{manifest_path} was built with {key}={row.get(key)!r}, but this run "
                        f"uses {want!r}. Resuming would mix incompatible settings; pass "
                        f"--no-resume to start over, or use a different --output_dir."
                    )

            if row.get("status") == "edited":
                edited_path = row.get("edited_path")
                if not edited_path or not (output_dir / edited_path).exists():
                    print(
                        f"WARNING: dataset_idx={row.get('dataset_idx')} is marked edited but its "
                        f"image is missing; that item will be redone"
                    )
                    continue

            kept.append(row)
            completed.add(int(row["dataset_idx"]))
    return kept, completed

def run_generate_edits(args) -> None:
    output_dir = Path(args.output_dir)
    edited_dir = output_dir / "edited"
    edited_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    yolo_model = load_yolo(args.yolo_model)
    dataset = WaymoE2E(
        indexFile=args.index_file,
        data_dir=args.data_dir,
        n_items=args.n_items,
    )

    end_idx = min(args.start_idx + args.max_items, len(dataset))
    dataset_indices = list(range(args.start_idx, end_idx))

    resume_rows, completed = load_resume_state(manifest_path, output_dir, args) if args.resume else ([], set())
    pending = [dataset_idx for dataset_idx in dataset_indices if dataset_idx not in completed]
    if completed:
        print(f"Resuming: {len(completed)} items already done, {len(pending)} to go")
    if not pending:
        print(f"Nothing to do; manifest at {manifest_path} is already complete.")
        return

    # Rewrite the manifest from the rows worth keeping, so a truncated line or a
    # row whose image vanished does not survive. Build it beside the original and
    # move it into place, so an interruption here cannot lose the earlier work.
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    with tmp_path.open("w") as f:
        for row in resume_rows:
            f.write(json.dumps(row) + "\n")
    tmp_path.replace(manifest_path)

    pipeline = None
    bs = 16

    # Decode, detect, and edit one batch at a time. Holding every frame and every
    # YOLO Results for the whole run costs ~6 GB per 1k items (Results keeps its
    # own copy of the source frame), which OOM-killed a 5k-item run at 64 GB.
    with manifest_path.open("a") as f:
        for start in range(0, len(pending), bs):
            batch_indices = pending[start : start + bs]
            batch_names, batch_images = [], []
            for dataset_idx in batch_indices:
                sample = dataset[dataset_idx]
                batch_names.append(sample["NAME"])
                batch_images.append(jpeg_tensor_to_image(sample["IMAGES_JPEG"][args.camera_idx]))

            # Reduce each Results to its description before the next batch is read,
            # so the frames they carry can be freed.
            batch_descriptions = [
                yolo_scene_description(result)
                for result in inference_yolo(yolo_model, batch_images)
            ]

            for dataset_idx, name, image, scene_description in zip(
                batch_indices, batch_names, batch_images, batch_descriptions
            ):
                raw_prompt = generate_local(scene_description, image)
                edit_plan = parse_edit_plan(raw_prompt)
                prompt = edit_plan["prompt"]

                row = {
                    "dataset_idx": dataset_idx,
                    "name": name,
                    "camera_idx": args.camera_idx,
                    "index_file": args.index_file,
                    "n_items": args.n_items,
                    "status": "no_change",
                    "edit_type": edit_plan["edit_type"],
                    "edit_direction": edit_plan["edit_direction"],
                    "prompt": prompt,
                    "scene_description": scene_description,
                    "generator": args.generator,
                    "editor": args.editor,
                    "edited_path": None,
                    "raw_generator_output": edit_plan.get("raw_generator_output"),
                }
                if "parse_error" in edit_plan:
                    row["parse_error"] = edit_plan["parse_error"]

                print(f"dataset_idx={dataset_idx} edit_type={row['edit_type']} direction={row['edit_direction']}")
                print(prompt)

                if row["edit_type"] != "no_change":
                    if pipeline is None:
                        hf_home = os.getenv("HF_HOME")
                        if hf_home is None:
                            print("WARNING: HF_HOME is not set; diffusers will use its default cache.")
                        pipeline = load_editor(args.editor)
                    edited_image = generate_edit_with_editor(args.editor, pipeline, image, prompt)
                    edited_path = edited_dir / f"{dataset_idx}.jpg"
                    edited_image.save(edited_path)
                    row["status"] = "edited"
                    row["edited_path"] = str(edited_path.relative_to(output_dir))
                else:
                    print(f"No change for dataset_idx={dataset_idx}, skipping edit.")

                f.write(json.dumps(row) + "\n")
                f.flush()

    print(f"Saved manifest to {manifest_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-edits")
    gen.add_argument("--data_dir", type=str, required=True, help="Path to Waymo directory")
    gen.add_argument("--index_file", type=str, default="index_val.pkl")
    gen.add_argument("--output_dir", type=str, required=True)
    gen.add_argument("--n_items", type=int, default=5_000)
    gen.add_argument("--start_idx", type=int, default=0)
    gen.add_argument("--max_items", type=int, default=10)
    gen.add_argument(
        "--camera_idx",
        type=int,
        choices=[1],
        default=1,
        help="Camera to edit. Only the front camera (1), which is consumed by the "
             "planner, is supported.",
    )
    gen.add_argument("--yolo_model", type=str, default="yolo26x.pt")
    gen.add_argument(
        "--editor",
        type=str,
        choices=["firered", "qwen", "flux"],
        default="firered",
    )
    gen.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep completed items from an existing manifest in --output_dir and only "
             "generate the rest. --no-resume regenerates everything from scratch.",
    )
    gen.set_defaults(func=run_generate_edits, generator="local")
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
