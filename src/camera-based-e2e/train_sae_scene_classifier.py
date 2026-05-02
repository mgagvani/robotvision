"""
Train an SAE-hidden-state scene classifier from scenes.json labels.

The frozen driving model produces hooked activations for each frame, the trained
SAE maps those to sparse hidden activations, and this script summarizes all
frames from a scene into one scene-level feature vector before training an MLP
classifier on top.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from loader import WaymoE2E
from models.base_model import collate_with_images
from sae_utils import compute_hidden_activations, load_model_and_sae, set_eval_mode


DEFAULT_SEED = 42
DEFAULT_MLP_LAYER_COUNTS = (1, 2, 3, 4)
INDEX_FILES = {
    "train": "index_train.pkl",
    "val": "index_val.pkl",
    "test": "index_test.pkl",
}
DEFAULT_CHECKPOINT_NAME = "sae_scene_classifier.pt"
FRAME_SUFFIX_RE = re.compile(r"^(?P<context>.+)-\d+$")


def default_num_workers() -> int:
    candidates: list[int] = []
    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        env_value = os.environ.get(env_name)
        if env_value is not None:
            try:
                candidates.append(int(env_value))
            except ValueError:
                pass
    if hasattr(os, "sched_getaffinity"):
        candidates.append(len(os.sched_getaffinity(0)))
    candidates.append(os.cpu_count() or 1)
    return max(1, min(candidates))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_index_file(split: str, index_file: str | None) -> str:
    if index_file is not None:
        return index_file
    return (Path(__file__).resolve().parent / INDEX_FILES[split]).as_posix()


def load_scene_labels(path: str | Path, label_key: str) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    labels: dict[str, str] = {}
    for context_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Expected a dict for scenes entry '{context_name}'")
        label = entry.get(label_key)
        if label is None:
            raise ValueError(
                f"Scene '{context_name}' is missing label key '{label_key}'"
            )
        labels[canonical_scene_name(str(context_name))] = str(label)
    if not labels:
        raise ValueError(f"No labels were loaded from {path}")
    return labels


def canonical_scene_name(name: str) -> str:
    """Use segment-level IDs when Waymo frame names include a trailing -NNN."""
    match = FRAME_SUFFIX_RE.match(name)
    return match.group("context") if match is not None else name


def class_maps(scene_labels: dict[str, str]) -> tuple[list[str], dict[str, int]]:
    classes = sorted(set(scene_labels.values()))
    return classes, {label: idx for idx, label in enumerate(classes)}


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def context_in_requested_fold(
    context_name: str,
    fold: str,
    val_fraction: float,
    seed: int,
) -> bool:
    if val_fraction <= 0:
        return True
    is_val = stable_fraction(context_name, seed) < val_fraction
    return is_val if fold == "val" else not is_val


def build_loader(
    *,
    split: str,
    index_file: str | None,
    data_dir: str,
    n_items: int | None,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    dataset = WaymoE2E(
        indexFile=resolve_index_file(split, index_file),
        data_dir=data_dir,
        n_items=n_items,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
        shuffle=shuffle,
    )


class SceneMLPClassifier(nn.Module):
    def __init__(
        self,
        sae_dim: int,
        num_classes: int,
        *,
        num_layers: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        resolved_hidden_dim = sae_dim if hidden_dim is None else hidden_dim
        if resolved_hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

        layers: list[nn.Module] = []
        in_dim = sae_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, resolved_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = resolved_hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.input_dim = sae_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = resolved_hidden_dim

    def forward(self, sae_hidden: torch.Tensor) -> torch.Tensor:
        return self.network(sae_hidden)


def one_hot_targets(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets, num_classes=num_classes).to(dtype=torch.float32)


@dataclass
class SceneAccumulator:
    target_id: int
    count: int
    hidden_sum: torch.Tensor
    hidden_sumsq: torch.Tensor
    hidden_min: torch.Tensor
    hidden_max: torch.Tensor
    example_jpeg: torch.Tensor | None = None

    @classmethod
    def from_hidden(
        cls,
        hidden: torch.Tensor,
        *,
        target_id: int,
        example_jpeg: torch.Tensor | None = None,
    ) -> "SceneAccumulator":
        hidden = hidden.detach().cpu().to(torch.float64)
        return cls(
            target_id=target_id,
            count=1,
            hidden_sum=hidden.clone(),
            hidden_sumsq=hidden.square(),
            hidden_min=hidden.clone(),
            hidden_max=hidden.clone(),
            example_jpeg=(
                example_jpeg.detach().cpu().clone()
                if isinstance(example_jpeg, torch.Tensor)
                else None
            ),
        )

    def update(self, hidden: torch.Tensor, *, example_jpeg: torch.Tensor | None = None) -> None:
        hidden = hidden.detach().cpu().to(torch.float64)
        self.count += 1
        self.hidden_sum += hidden
        self.hidden_sumsq += hidden.square()
        self.hidden_min = torch.minimum(self.hidden_min, hidden)
        self.hidden_max = torch.maximum(self.hidden_max, hidden)
        if self.example_jpeg is None and isinstance(example_jpeg, torch.Tensor):
            self.example_jpeg = example_jpeg.detach().cpu().clone()


@dataclass
class SceneFeatureDataset:
    features: torch.Tensor
    targets: torch.Tensor
    context_names: list[str]
    example_jpegs: list[torch.Tensor | None]


def summarize_scene_activation(
    scene: SceneAccumulator,
    *,
    stats_type: str,
) -> torch.Tensor:
    mean = scene.hidden_sum / max(scene.count, 1)
    if stats_type == "mean_std":
        variance = torch.clamp(scene.hidden_sumsq / max(scene.count, 1) - mean.square(), min=0.0)
        std = torch.sqrt(variance)
        summary = torch.cat((mean, std), dim=0)
    elif stats_type == "min_max":
        summary = torch.cat((scene.hidden_min, scene.hidden_max), dim=0)
    else:
        raise ValueError(f"Unsupported stats_type: {stats_type}")
    return summary.to(torch.float32)


def classifier_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_type: str,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if loss_type == "cross_entropy":
        return F.cross_entropy(logits, targets, weight=class_weights)
    if loss_type == "mse":
        per_class_error = F.mse_loss(
            logits,
            one_hot_targets(targets, num_classes=logits.size(-1)),
            reduction="none",
        ).mean(dim=-1)
        if class_weights is not None:
            per_class_error = per_class_error * class_weights[targets]
        return per_class_error.mean()
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def make_class_weights(
    scene_labels: dict[str, str],
    class_to_idx: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(scene_labels.values())
    num_classes = len(class_to_idx)
    total = sum(counts.values())
    weights = torch.ones(num_classes, dtype=torch.float32)
    for label, idx in class_to_idx.items():
        weights[idx] = total / max(num_classes * counts[label], 1)
    return weights.to(device)


def batch_labels(
    names: list[str],
    scene_labels: dict[str, str],
    class_to_idx: dict[str, int],
    *,
    fold: str,
    context_val_fraction: float,
    seed: int,
) -> tuple[list[int], torch.Tensor, list[str]]:
    keep_indices: list[int] = []
    target_ids: list[int] = []
    kept_names: list[str] = []

    for sample_idx, name in enumerate(names):
        raw_name = str(name)
        context_name = canonical_scene_name(raw_name)
        label = scene_labels.get(context_name)
        if label is None:
            label = scene_labels.get(raw_name)
            context_name = raw_name
        if label is None:
            continue
        if not context_in_requested_fold(
            context_name,
            fold=fold,
            val_fraction=context_val_fraction,
            seed=seed,
        ):
            continue
        keep_indices.append(sample_idx)
        target_ids.append(class_to_idx[label])
        kept_names.append(context_name)

    return keep_indices, torch.as_tensor(target_ids, dtype=torch.long), kept_names


def extract_labeled_hidden(
    batch: dict[str, Any],
    *,
    model: nn.Module,
    sae: nn.Module,
    device: torch.device,
    scene_labels: dict[str, str],
    class_to_idx: dict[str, int],
    fold: str,
    context_val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str], list[int]]:
    keep_indices, targets, kept_names = batch_labels(
        batch["NAME"],
        scene_labels,
        class_to_idx,
        fold=fold,
        context_val_fraction=context_val_fraction,
        seed=seed,
    )
    if not keep_indices:
        return None, None, [], []

    past = batch["PAST"].to(device)
    intent = batch["INTENT"].to(device)
    images = model.decode_batch_jpeg(batch["IMAGES_JPEG"], device=device)
    model_inputs = {"PAST": past, "IMAGES": images, "INTENT": intent}

    sae.internal_acts = None
    _ = model(model_inputs)
    if sae.internal_acts is None:
        raise RuntimeError("No SAE activations were captured from the registered hook")

    hidden = compute_hidden_activations(sae, sae.internal_acts)
    indices = torch.as_tensor(keep_indices, dtype=torch.long, device=hidden.device)
    return (
        hidden.index_select(0, indices).detach(),
        targets.to(device),
        kept_names,
        keep_indices,
    )


def get_example_jpeg(
    images_jpeg: list[list[torch.Tensor]],
    *,
    sample_idx: int,
    camera_idx: int,
) -> torch.Tensor | None:
    if sample_idx >= len(images_jpeg) or camera_idx >= len(images_jpeg[sample_idx]):
        return None
    jpeg_tensor = images_jpeg[sample_idx][camera_idx]
    if not isinstance(jpeg_tensor, torch.Tensor) or jpeg_tensor.numel() == 0:
        return None
    return jpeg_tensor


def build_scene_feature_dataset(
    *,
    loader: DataLoader,
    model: nn.Module,
    sae: nn.Module,
    device: torch.device,
    scene_labels: dict[str, str],
    class_to_idx: dict[str, int],
    fold: str,
    context_val_fraction: float,
    seed: int,
    stats_type: str,
    example_camera_idx: int,
) -> SceneFeatureDataset:
    scenes: dict[str, SceneAccumulator] = {}
    ordered_contexts: list[str] = []

    pbar = tqdm(loader, desc=f"{fold}_scene_features", leave=False)
    for batch in pbar:
        with torch.no_grad():
            hidden, targets, kept_names, keep_indices = extract_labeled_hidden(
                batch,
                model=model,
                sae=sae,
                device=device,
                scene_labels=scene_labels,
                class_to_idx=class_to_idx,
                fold=fold,
                context_val_fraction=context_val_fraction,
                seed=seed,
            )

        if hidden is None or targets is None:
            continue

        hidden_cpu = hidden.detach().cpu()
        target_ids = targets.detach().cpu().tolist()
        for row_idx, (context_name, target_id, sample_idx) in enumerate(
            zip(kept_names, target_ids, keep_indices)
        ):
            example_jpeg = get_example_jpeg(
                batch["IMAGES_JPEG"],
                sample_idx=sample_idx,
                camera_idx=example_camera_idx,
            )
            scene = scenes.get(context_name)
            if scene is None:
                scenes[context_name] = SceneAccumulator.from_hidden(
                    hidden_cpu[row_idx],
                    target_id=int(target_id),
                    example_jpeg=example_jpeg,
                )
                ordered_contexts.append(context_name)
            else:
                if scene.target_id != int(target_id):
                    raise ValueError(
                        f"Context '{context_name}' maps to multiple target ids: "
                        f"{scene.target_id} vs {int(target_id)}"
                    )
                scene.update(hidden_cpu[row_idx], example_jpeg=example_jpeg)

        pbar.set_postfix(contexts=len(scenes))

    if not ordered_contexts:
        return SceneFeatureDataset(
            features=torch.empty(0, 0, dtype=torch.float32),
            targets=torch.empty(0, dtype=torch.long),
            context_names=[],
            example_jpegs=[],
        )

    features = torch.stack(
        [
            summarize_scene_activation(scenes[context_name], stats_type=stats_type)
            for context_name in ordered_contexts
        ],
        dim=0,
    )
    targets = torch.as_tensor(
        [scenes[context_name].target_id for context_name in ordered_contexts],
        dtype=torch.long,
    )
    example_jpegs = [scenes[context_name].example_jpeg for context_name in ordered_contexts]
    return SceneFeatureDataset(
        features=features,
        targets=targets,
        context_names=ordered_contexts,
        example_jpegs=example_jpegs,
    )


def empty_metrics() -> dict[str, Any]:
    return {
        "loss": 0.0,
        "accuracy": 0.0,
        "macro_accuracy": 0.0,
        "samples": 0,
        "contexts": 0,
        "per_class": {},
    }


def finalize_metrics(
    *,
    total_loss: float,
    total_seen: int,
    total_correct: int,
    class_seen: dict[int, int],
    class_correct: dict[int, int],
    classes: list[str],
    contexts: set[str],
) -> dict[str, Any]:
    if total_seen == 0:
        return empty_metrics()

    per_class = {}
    class_acc_values = []
    for idx, label in enumerate(classes):
        seen = class_seen.get(idx, 0)
        correct = class_correct.get(idx, 0)
        acc = correct / seen if seen else 0.0
        if seen:
            class_acc_values.append(acc)
        per_class[label] = {
            "accuracy": acc,
            "correct": correct,
            "samples": seen,
        }

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "macro_accuracy": (
            sum(class_acc_values) / len(class_acc_values) if class_acc_values else 0.0
        ),
        "samples": total_seen,
        "contexts": len(contexts),
        "per_class": per_class,
    }


def run_epoch(
    *,
    dataset: SceneFeatureDataset,
    classifier: SceneMLPClassifier,
    optimizer: torch.optim.Optimizer | None,
    training: bool,
    device: torch.device,
    classes: list[str],
    fold: str,
    batch_size: int,
    loss_type: str,
    class_weights: torch.Tensor | None,
) -> dict[str, Any]:
    if dataset.targets.numel() == 0:
        return empty_metrics()

    is_train = training
    classifier.train(is_train)

    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    class_seen: dict[int, int] = defaultdict(int)
    class_correct: dict[int, int] = defaultdict(int)
    contexts = set(dataset.context_names)
    index_tensor = torch.arange(dataset.targets.numel(), dtype=torch.long)
    scene_loader = DataLoader(
        TensorDataset(dataset.features, dataset.targets, index_tensor),
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
    )

    pbar = tqdm(scene_loader, desc=fold, leave=False)
    for hidden, targets, _ in pbar:
        hidden = hidden.to(device=device, dtype=torch.float32)
        targets = targets.to(device)

        if is_train:
            if optimizer is None:
                raise RuntimeError("Training requested without an optimizer")
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(hidden)
            loss = classifier_loss(
                logits,
                targets,
                loss_type=loss_type,
                class_weights=class_weights,
            )
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = classifier(hidden)
                loss = classifier_loss(
                    logits,
                    targets,
                    loss_type=loss_type,
                    class_weights=class_weights,
                )

        preds = logits.argmax(dim=-1)
        correct_mask = preds.eq(targets)
        current_batch_size = targets.numel()
        total_seen += current_batch_size
        total_correct += int(correct_mask.sum().item())
        total_loss += float(loss.item()) * current_batch_size

        for target_id in targets.detach().cpu().tolist():
            class_seen[int(target_id)] += 1
        for target_id, is_correct in zip(
            targets.detach().cpu().tolist(),
            correct_mask.detach().cpu().tolist(),
        ):
            if is_correct:
                class_correct[int(target_id)] += 1

        pbar.set_postfix(
            loss=f"{total_loss / max(total_seen, 1):.4f}",
            acc=f"{total_correct / max(total_seen, 1):.3f}",
            labeled=total_seen,
        )

    metrics = finalize_metrics(
        total_loss=total_loss,
        total_seen=total_seen,
        total_correct=total_correct,
        class_seen=class_seen,
        class_correct=class_correct,
        classes=classes,
        contexts=contexts,
    )
    return metrics


def save_example_jpeg(
    jpeg_tensor: torch.Tensor | None,
    *,
    image_path: Path,
    label_record: dict[str, Any] | None = None,
) -> str | None:
    if not isinstance(jpeg_tensor, torch.Tensor) or jpeg_tensor.numel() == 0:
        return None

    image_path.parent.mkdir(parents=True, exist_ok=True)
    jpeg_bytes = jpeg_tensor.detach().cpu().numpy().tobytes()
    if label_record is None:
        image_path.write_bytes(jpeg_bytes)
    else:
        write_labeled_jpeg(jpeg_bytes, image_path=image_path, label_record=label_record)
    return image_path.as_posix()


def write_labeled_jpeg(
    jpeg_bytes: bytes,
    *,
    image_path: Path,
    label_record: dict[str, Any],
) -> None:
    try:
        from io import BytesIO

        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        image_path.write_bytes(jpeg_bytes)
        return

    image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    width, height = image.size
    font_size = max(18, min(34, width // 32))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
        small_font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, font_size - 6))
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    status = "OK" if label_record["correct"] else "MISS"
    lines = [
        f"{status}  true: {label_record['true_label']}",
        (
            f"pred: {label_record['predicted_label']}  "
            f"conf: {label_record['confidence']:.3f}"
        ),
        f"context: {label_record['context_name'][:16]}...",
    ]
    fonts = [font, font, small_font]
    padding = max(10, width // 100)
    line_gap = max(4, font_size // 5)
    text_width = 0
    text_height = 0
    bboxes = []
    for line, line_font in zip(lines, fonts):
        bbox = draw.textbbox((0, 0), line, font=line_font)
        bboxes.append(bbox)
        text_width = max(text_width, bbox[2] - bbox[0])
        text_height += bbox[3] - bbox[1]
    text_height += line_gap * (len(lines) - 1)

    box_w = min(width - 2 * padding, text_width + 2 * padding)
    box_h = text_height + 2 * padding
    box_h = min(height, box_h)
    fill = (20, 20, 20, 210) if label_record["correct"] else (90, 20, 20, 220)
    accent = (38, 166, 91, 235) if label_record["correct"] else (220, 64, 64, 235)
    draw.rectangle((0, 0, box_w, box_h), fill=fill)
    draw.rectangle((0, 0, max(6, padding // 2), box_h), fill=accent)

    y = padding
    for line, line_font, bbox in zip(lines, fonts, bboxes):
        draw.text((padding + max(6, padding // 2), y), line, fill=(255, 255, 255, 255), font=line_font)
        y += (bbox[3] - bbox[1]) + line_gap

    image.save(image_path, format="JPEG", quality=92)


def prediction_record(
    *,
    context_name: str,
    target_id: int,
    logits: torch.Tensor,
    classes: list[str],
    image_path: str | None,
    top_k: int = 3,
) -> dict[str, Any]:
    probabilities = torch.softmax(logits.detach().cpu(), dim=-1)
    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(top_k, probabilities.numel()),
    )
    predicted_id = int(top_indices[0].item())
    true_label = classes[target_id]
    predicted_label = classes[predicted_id]
    return {
        "context_name": context_name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": predicted_label == true_label,
        "confidence": float(top_probs[0].item()),
        "image_path": image_path,
        "top_predictions": [
            {
                "label": classes[int(idx.item())],
                "probability": float(prob.item()),
                "logit": float(logits.detach().cpu()[int(idx.item())].item()),
            }
            for prob, idx in zip(top_probs, top_indices)
        ],
    }


def collect_prediction_examples(
    *,
    dataset: SceneFeatureDataset,
    classifier: SceneMLPClassifier,
    device: torch.device,
    classes: list[str],
    fold: str,
    batch_size: int,
    loss_type: str,
    class_weights: torch.Tensor | None,
    num_examples: int,
    image_dir: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if dataset.targets.numel() == 0:
        return empty_metrics(), []

    classifier.eval()

    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    class_seen: dict[int, int] = defaultdict(int)
    class_correct: dict[int, int] = defaultdict(int)
    examples: list[dict[str, Any]] = []

    contexts = set(dataset.context_names)
    index_tensor = torch.arange(dataset.targets.numel(), dtype=torch.long)
    scene_loader = DataLoader(
        TensorDataset(dataset.features, dataset.targets, index_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    pbar = tqdm(scene_loader, desc=f"{fold}_examples", leave=False)
    for hidden, targets, indices in pbar:
        with torch.no_grad():
            logits = classifier(hidden.to(device=device, dtype=torch.float32))
            loss = classifier_loss(
                logits,
                targets.to(device),
                loss_type=loss_type,
                class_weights=class_weights,
            )

        preds = logits.argmax(dim=-1)
        correct_mask = preds.eq(targets.to(device))
        current_batch_size = targets.numel()
        total_seen += current_batch_size
        total_correct += int(correct_mask.sum().item())
        total_loss += float(loss.item()) * current_batch_size

        target_ids = targets.detach().cpu().tolist()
        for target_id in target_ids:
            class_seen[int(target_id)] += 1
        for target_id, is_correct in zip(target_ids, correct_mask.detach().cpu().tolist()):
            if is_correct:
                class_correct[int(target_id)] += 1

        if len(examples) < num_examples:
            for row_idx, (dataset_idx, target_id) in enumerate(
                zip(indices.detach().cpu().tolist(), target_ids)
            ):
                if len(examples) >= num_examples:
                    break
                context_name = dataset.context_names[int(dataset_idx)]
                image_path = None
                image_path_obj = (
                    image_dir / f"example_{len(examples) + 1:03d}.jpg"
                    if image_dir is not None
                    else None
                )
                if image_path_obj is not None:
                    image_path = image_path_obj.as_posix()
                record = prediction_record(
                    context_name=context_name,
                    target_id=int(target_id),
                    logits=logits[row_idx],
                    classes=classes,
                    image_path=image_path,
                )
                if image_path_obj is not None:
                    save_example_jpeg(
                        dataset.example_jpegs[int(dataset_idx)],
                        image_path=image_path_obj,
                        label_record=record,
                    )
                examples.append(record)

        pbar.set_postfix(
            acc=f"{total_correct / max(total_seen, 1):.3f}",
            examples=len(examples),
        )

    metrics = finalize_metrics(
        total_loss=total_loss,
        total_seen=total_seen,
        total_correct=total_correct,
        class_seen=class_seen,
        class_correct=class_correct,
        classes=classes,
        contexts=contexts,
    )
    return metrics, examples


def save_prediction_examples(
    *,
    examples_path: Path,
    checkpoint_path: Path,
    metrics: dict[str, Any],
    examples: list[dict[str, Any]],
    classes: list[str],
    args: argparse.Namespace,
) -> None:
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": checkpoint_path.as_posix(),
        "classes": classes,
        "metrics": metrics,
        "num_examples_requested": args.num_examples,
        "num_examples_saved": len(examples),
        "examples": examples,
    }
    examples_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes_path",
        type=str,
        default=(repo_root() / "scenes.json").as_posix(),
        help="Path to scenes.json",
    )
    parser.add_argument("--label_key", type=str, default="scenario_cluster")
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--sae_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="val", choices=INDEX_FILES)
    parser.add_argument("--val_split", type=str, default="val", choices=INDEX_FILES)
    parser.add_argument("--train_index_file", type=str, default=None)
    parser.add_argument("--val_index_file", type=str, default=None)
    parser.add_argument("--train_items", type=int, default=100_000)
    parser.add_argument("--val_items", type=int, default=20_2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=13)
    parser.add_argument("--block_idx", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        choices=("cross_entropy", "mse"),
        help="Use mse for a true one-hot regression style scene classifier.",
    )
    parser.add_argument(
        "--scene_feature_stats",
        type=str,
        default="mean_std",
        choices=("mean_std", "min_max"),
        help=(
            "How to summarize frame-level SAE activations into one scene-level "
            "descriptor before the classifier."
        ),
    )
    parser.add_argument(
        "--mlp_layer_counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_MLP_LAYER_COUNTS),
        help="MLP depths to train, where each depth counts all Linear layers.",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=None,
        help="Hidden width for MLP runs with 2+ layers. Defaults to the SAE feature size.",
    )
    parser.add_argument("--balanced_loss", action="store_true")
    parser.add_argument(
        "--context_val_fraction",
        type=float,
        default=0.2,
        help=(
            "Optional deterministic context-level val split applied inside each "
            "loader. The repository scenes.json labels are validation contexts, "
            "so the default uses index_val.pkl for both loaders and holds out "
            "20%% of labeled contexts for validation. Set this to 0 when using "
            "separate labeled train/val index files."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metrics_path", type=str, default=None)
    parser.add_argument(
        "--examples_path",
        type=str,
        default=None,
        help="Optional JSON path for stored scene prediction examples.",
    )
    parser.add_argument("--num_examples", type=int, default=16)
    parser.add_argument(
        "--example_image_dir",
        type=str,
        default=None,
        help="Optional directory for representative front-camera JPEG examples.",
    )
    parser.add_argument("--example_camera_idx", type=int, default=1)
    return parser


def save_checkpoint(
    *,
    output_path: Path,
    classifier: SceneMLPClassifier,
    classes: list[str],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
    best_epoch: int,
    best_metric: float,
    history: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "sae_dim": classifier.input_dim,
            "num_classes": classifier.num_classes,
            "mlp_num_layers": classifier.num_layers,
            "mlp_hidden_dim": classifier.hidden_dim,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "idx_to_class": {idx: label for label, idx in class_to_idx.items()},
            "best_epoch": best_epoch,
            "best_val_accuracy": best_metric,
            "history": history,
            "args": vars(args),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        output_path,
    )


def resolve_output_path(raw_output_path: str) -> Path:
    output_path = Path(raw_output_path)
    if raw_output_path.endswith(os.sep) or (
        output_path.exists() and output_path.is_dir()
    ):
        return output_path / DEFAULT_CHECKPOINT_NAME
    return output_path


def tagged_path(path: Path, tag: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.{tag}{path.suffix}")
    return path.with_name(f"{path.name}.{tag}")


def normalize_layer_counts(raw_counts: list[int]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for count in raw_counts:
        if count < 1:
            raise ValueError("--mlp_layer_counts values must be at least 1")
        if count not in seen:
            normalized.append(count)
            seen.add(count)
    if not normalized:
        raise ValueError("--mlp_layer_counts must include at least one value")
    return normalized


def train_classifier_for_depth(
    *,
    num_layers: int,
    train_scene_dataset: SceneFeatureDataset,
    val_scene_dataset: SceneFeatureDataset,
    device: torch.device,
    classes: list[str],
    batch_size: int,
    loss_type: str,
    class_weights: torch.Tensor | None,
    lr: float,
    weight_decay: float,
    epochs: int,
    hidden_dim: int | None,
) -> tuple[SceneMLPClassifier, list[dict[str, Any]], float, int]:
    classifier = SceneMLPClassifier(
        sae_dim=train_scene_dataset.features.size(-1),
        num_classes=len(classes),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        train_metrics = run_epoch(
            dataset=train_scene_dataset,
            classifier=classifier,
            optimizer=optimizer,
            training=True,
            device=device,
            classes=classes,
            fold=f"train_{num_layers}layer",
            batch_size=batch_size,
            loss_type=loss_type,
            class_weights=class_weights,
        )
        val_metrics = run_epoch(
            dataset=val_scene_dataset,
            classifier=classifier,
            optimizer=None,
            training=False,
            device=device,
            classes=classes,
            fold=f"val_{num_layers}layer",
            batch_size=batch_size,
            loss_type=loss_type,
            class_weights=class_weights,
        )

        monitor_metric = (
            val_metrics["accuracy"]
            if val_metrics["samples"] > 0
            else train_metrics["accuracy"]
        )
        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }

        row = {
            "epoch": epoch,
            "mlp_num_layers": num_layers,
            "train": train_metrics,
            "val": val_metrics,
            "monitor_accuracy": monitor_metric,
        }
        history.append(row)
        print(
            f"layers={num_layers} "
            f"epoch={epoch} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"train_samples={train_metrics['samples']} "
            f"val_samples={val_metrics['samples']}"
        )

    if best_state is not None:
        classifier.load_state_dict(best_state, strict=True)

    return classifier, history, best_metric, best_epoch


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0")
    if not 0.0 <= args.context_val_fraction < 1.0:
        raise ValueError("--context_val_fraction must be in [0, 1)")
    if args.mlp_hidden_dim is not None and args.mlp_hidden_dim <= 0:
        raise ValueError("--mlp_hidden_dim must be greater than 0")
    args.mlp_layer_counts = normalize_layer_counts(args.mlp_layer_counts)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device
    )

    scene_labels = load_scene_labels(args.scenes_path, args.label_key)
    classes, class_to_idx = class_maps(scene_labels)
    class_weights = (
        make_class_weights(scene_labels, class_to_idx, device)
        if args.balanced_loss
        else None
    )

    train_loader = build_loader(
        split=args.train_split,
        index_file=args.train_index_file,
        data_dir=args.data_dir,
        n_items=args.train_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=True,
    )
    val_loader = build_loader(
        split=args.val_split,
        index_file=args.val_index_file,
        data_dir=args.data_dir,
        n_items=args.val_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        shuffle=False,
    )

    model, sae, hook_handle = load_model_and_sae(
        model_checkpoint_path=args.model_checkpoint_path,
        sae_checkpoint_path=args.sae_checkpoint_path,
        block_idx=args.block_idx,
        device=device,
    )
    set_eval_mode(model, sae)

    try:
        train_scene_dataset = build_scene_feature_dataset(
            loader=train_loader,
            model=model,
            sae=sae,
            device=device,
            scene_labels=scene_labels,
            class_to_idx=class_to_idx,
            fold="train",
            context_val_fraction=args.context_val_fraction,
            seed=args.seed,
            stats_type=args.scene_feature_stats,
            example_camera_idx=args.example_camera_idx,
        )
        val_scene_dataset = build_scene_feature_dataset(
            loader=val_loader,
            model=model,
            sae=sae,
            device=device,
            scene_labels=scene_labels,
            class_to_idx=class_to_idx,
            fold="val",
            context_val_fraction=args.context_val_fraction,
            seed=args.seed,
            stats_type=args.scene_feature_stats,
            example_camera_idx=args.example_camera_idx,
        )
        print(
            f"Built scene-level features using {args.scene_feature_stats}: "
            f"train_contexts={len(train_scene_dataset.context_names)} "
            f"val_contexts={len(val_scene_dataset.context_names)}"
        )
        if train_scene_dataset.targets.numel() == 0:
            raise RuntimeError(
                "No labeled training scenes were found for "
                f"train_split={args.train_split!r}, "
                f"train_index_file={resolve_index_file(args.train_split, args.train_index_file)!r}, "
                f"train_items={args.train_items!r}. The bundled scenes.json "
                "contains validation context IDs, so use index_val.pkl for "
                "training or provide a scenes.json that labels the selected "
                "training index."
            )

        base_output_path = resolve_output_path(args.output_path)
        summary_path = tagged_path(base_output_path, "sweep_summary").with_suffix(".json")
        sweep_results: list[dict[str, Any]] = []

        for num_layers in args.mlp_layer_counts:
            run_tag = f"{num_layers}layer"
            print(f"Starting scene-classifier run for mlp_layers={num_layers}")

            classifier, history, best_metric, best_epoch = train_classifier_for_depth(
                num_layers=num_layers,
                train_scene_dataset=train_scene_dataset,
                val_scene_dataset=val_scene_dataset,
                device=device,
                classes=classes,
                batch_size=args.batch_size,
                loss_type=args.loss_type,
                class_weights=class_weights,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                hidden_dim=args.mlp_hidden_dim,
            )

            output_path = tagged_path(base_output_path, run_tag)
            save_checkpoint(
                output_path=output_path,
                classifier=classifier,
                classes=classes,
                class_to_idx=class_to_idx,
                args=args,
                best_epoch=best_epoch,
                best_metric=best_metric,
                history=history,
            )

            metrics_path = (
                tagged_path(Path(args.metrics_path), run_tag)
                if args.metrics_path is not None
                else output_path.with_suffix(output_path.suffix + ".metrics.json")
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            print(f"Saved SAE scene classifier to {output_path}")
            print(f"Saved metrics to {metrics_path}")
            print(
                f"Best monitored accuracy for {num_layers} layer(s): "
                f"{best_metric:.4f} at epoch {best_epoch}"
            )
            if history and history[-1]["val"]["samples"] == 0:
                print("Warning: no labeled validation samples were found.")

            examples_path_str: str | None = None
            if args.examples_path is not None:
                image_dir = (
                    Path(args.example_image_dir) / run_tag
                    if args.example_image_dir is not None
                    else None
                )
                example_metrics, examples = collect_prediction_examples(
                    dataset=val_scene_dataset,
                    classifier=classifier,
                    device=device,
                    classes=classes,
                    fold=f"val_{run_tag}",
                    batch_size=args.batch_size,
                    loss_type=args.loss_type,
                    class_weights=class_weights,
                    num_examples=args.num_examples,
                    image_dir=image_dir,
                )
                examples_path = tagged_path(Path(args.examples_path), run_tag)
                save_prediction_examples(
                    examples_path=examples_path,
                    checkpoint_path=output_path,
                    metrics=example_metrics,
                    examples=examples,
                    classes=classes,
                    args=args,
                )
                examples_path_str = examples_path.as_posix()
                print(
                    f"Saved {len(examples)} scene prediction example(s) to "
                    f"{examples_path}"
                )

            sweep_results.append(
                {
                    "mlp_num_layers": num_layers,
                    "mlp_hidden_dim": classifier.hidden_dim,
                    "checkpoint_path": output_path.as_posix(),
                    "metrics_path": metrics_path.as_posix(),
                    "examples_path": examples_path_str,
                    "best_epoch": best_epoch,
                    "best_monitor_accuracy": best_metric,
                    "final_train_accuracy": history[-1]["train"]["accuracy"] if history else 0.0,
                    "final_val_accuracy": history[-1]["val"]["accuracy"] if history else 0.0,
                    "val_samples": history[-1]["val"]["samples"] if history else 0,
                }
            )

        best_run = max(sweep_results, key=lambda row: row["best_monitor_accuracy"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "layer_counts": args.mlp_layer_counts,
                    "results": sweep_results,
                    "best_run": best_run,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved sweep summary to {summary_path}")
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    main()
