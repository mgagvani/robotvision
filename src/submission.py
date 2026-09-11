from protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import numpy as np
import torch
from torch.utils.data import DataLoader
from loader import WaymoE2E
from models.base_model import collate_with_images
from models.checkpoint_loader import load_model_from_checkpoint
import os
import argparse
import resource
import shutil
import tarfile
from typing import List
import math
from tqdm import tqdm


def relax_fd_limits() -> None:
    """Raise the soft open-file limit toward the hard limit.

    Each test frame carries 8 JPEG tensors, so a prefetching multi-worker
    DataLoader keeps thousands of shared-memory descriptors alive at once.
    Combined with the file_system sharing strategy below this avoids the
    `RuntimeError: received 0 items of ancdata` crash.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def selected_trajectory(output: dict[str, torch.Tensor], horizon: int = 20) -> torch.Tensor:
    traj = output["trajectory"]
    scores = output.get("scores")
    if traj.ndim != 2:
        raise ValueError(f"Expected flat trajectory output, got {traj.shape}")

    batch_size = traj.size(0)
    traj = traj.view(batch_size, -1, horizon, 2)
    if scores is None or traj.size(1) == 1:
        return traj[:, 0]

    selected = scores.argmin(dim=1)
    return traj[torch.arange(batch_size, device=traj.device), selected]


def generate_submission_data(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device = None,
) -> List[wod_e2ed_submission_pb2.FrameTrajectoryPredictions]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Generating submission predictions"):
            past = batch["PAST"].to(device, non_blocking=True)
            intent = batch["INTENT"].to(device, non_blocking=True)
            names = batch["NAME"]
            images = [img.to(device, non_blocking=True) if img is not None else None for img in batch["IMAGES"]]

            output = model({"PAST": past, "IMAGES": images, "INTENT": intent})
            pred_future = selected_trajectory(output, horizon=20).cpu().numpy()

            for i, name in enumerate(names):
                pos_x = pred_future[i, :, 0].astype(np.float32)
                pos_y = pred_future[i, :, 1].astype(np.float32)

                trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(
                    pos_x=pos_x,
                    pos_y=pos_y,
                )
                frame_prediction = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(
                    frame_name=name,
                    trajectory=trajectory,
                )
                predictions.append(frame_prediction)

    return predictions


def serialize_and_save_submission(
    predictions: List[wod_e2ed_submission_pb2.FrameTrajectoryPredictions],
    output_file: str,
    num_shards: int = 8,
    *,
    unique_method_name: str = "TrajScorer",
    description: str = "ViT features -> cross-attention -> scorer + multiple trajectory heads",
    num_model_parameters: str = "36m",
):
    submission_file_base = output_file
    os.makedirs(submission_file_base, exist_ok=True)
    sub_file_names = [
        os.path.join(
            submission_file_base,
            f"submission.binproto-{i:05d}-of-{num_shards:05d}",
        )
        for i in range(num_shards)
    ]
    submissions = []
    num_predictions_per_shard = math.ceil(len(predictions) / num_shards)
    for i in range(num_shards):
        start = i * num_predictions_per_shard
        end = (i + 1) * num_predictions_per_shard
        submissions.append(
            wod_e2ed_submission_pb2.E2EDChallengeSubmission(predictions=predictions[start:end])
        )

    for i, shard in enumerate(submissions):
        shard.submission_type = (
            wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
        )
        shard.authors[:] = ["Manav Gagvani"]
        shard.affiliation = "Purdue University"
        shard.account_name = "manavgagvani@gmail.com"
        shard.unique_method_name = unique_method_name
        shard.method_link = ""
        shard.description = description
        shard.uses_public_model_pretraining = True
        shard.public_model_names.extend(["Perception Encoder Small"])
        shard.num_model_parameters = num_model_parameters
        with open(sub_file_names[i], "wb") as fp:
            fp.write(shard.SerializeToString())


def package_submission_tarball(submission_dir: str, tarball_path: str) -> str:
    os.makedirs(os.path.dirname(tarball_path) or ".", exist_ok=True)
    shard_names = sorted(
        f
        for f in os.listdir(submission_dir)
        if f.startswith("submission.binproto-")
    )
    if not shard_names:
        raise FileNotFoundError(f"No submission shards found in {submission_dir}")

    with tarfile.open(tarball_path, "w:gz") as tar:
        for name in shard_names:
            tar.add(os.path.join(submission_dir, name), arcname=name)
    return tarball_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Waymo E2E data directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Directory to save submission shard files",
    )
    parser.add_argument(
        "--tarball",
        type=str,
        default=None,
        help="Optional path for submission .tar.gz (defaults to output_file.tar.gz)",
    )
    parser.add_argument(
        "--copy_to",
        type=str,
        default=None,
        help="Optional destination path for the tarball (e.g. ~/robotvision/gtrs_n250.tar.gz)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "deepmonocular", "gtrs", "drivor"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers; keep at or below --cpus-per-task",
    )
    parser.add_argument("--unique_method_name", type=str, default=None)
    args = parser.parse_args()

    relax_fd_limits()
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

    test_dataset = WaymoE2E(
        indexFile="index_test.pkl", data_dir=args.data_dir, n_items=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_with_images,
        persistent_workers=False,
        pin_memory=False,
    )

    model = load_model_from_checkpoint(args.checkpoint, model_type=args.model_type)
    device = next(model.parameters()).device

    predictions = generate_submission_data(model, test_loader, device=device)
    method_name = args.unique_method_name or f"{args.model_type}_waymo"
    serialize_and_save_submission(
        predictions,
        args.output_file,
        unique_method_name=method_name,
        description=f"Waymo E2E submission from {args.model_type} checkpoint",
    )

    tarball_path = args.tarball or f"{args.output_file.rstrip('/')}.tar.gz"
    package_submission_tarball(args.output_file, tarball_path)
    print(f"Wrote submission tarball: {tarball_path}")

    if args.copy_to:
        copy_to = os.path.expanduser(args.copy_to)
        os.makedirs(os.path.dirname(copy_to) or ".", exist_ok=True)
        shutil.copy2(tarball_path, copy_to)
        print(f"Copied submission tarball to: {copy_to}")
