from protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loader import WaymoE2E
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures
from models.base_model import LitModel, collate_with_images
import os
import argparse
from typing import List
import math
from tqdm import tqdm
import torchvision


def decode_batch_jpeg(images_jpeg: list[list[torch.Tensor]], device: torch.device) -> list[torch.Tensor]:
    """Decode batched JPEG bytes into per-camera tensors on device."""
    flat_encoded = []
    cam_sizes = []
    for cam in images_jpeg:
        cam_sizes.append(len(cam))
        flat_encoded.extend(
            jpg if isinstance(jpg, torch.Tensor) else torch.frombuffer(memoryview(jpg), dtype=torch.uint8)
            for jpg in cam
        )

    flat_decoded = torchvision.io.decode_jpeg(
        flat_encoded,
        mode=torchvision.io.ImageReadMode.UNCHANGED,
        device=device,
    )

    out = []
    idx = 0
    for n in cam_sizes:
        cam_list = flat_decoded[idx:idx + n]
        idx += n
        out.append(torch.stack(cam_list, dim=0))
    return out

def load_model(checkpoint_path: str, device: torch.device = None) -> DeepMonocularModel:
    """Load a trained MonocularModel from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.ckpt file)
        device: Device to load the model on. Defaults to CUDA if available.
    
    Returns:
        The loaded MonocularModel in eval mode.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Not running on CUDA")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device("cuda" if device is None else device)
    
    out_dim = 20 * 2  # Future: (B, 20, 2)
    model = DeepMonocularModel(out_dim=out_dim, 
                               feature_extractor=SAMFeatures(model_name="timm/vit_pe_spatial_small_patch16_512.fb"),
                               n_blocks=4,
                               n_proposals=50)
    lit_model = LitModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        lr=1e-4,
        map_location="cpu",
        weights_only=False,
    )
    model = lit_model.model
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def generate_submission_data(
    model: DeepMonocularModel, 
    data_loader: DataLoader,
    device: torch.device = None
) -> List[wod_e2ed_submission_pb2.FrameTrajectoryPredictions]:
    """Generate submission predictions for all frames in the data loader.
    
    Args:
        model: The trained MonocularModel.
        data_loader: DataLoader yielding batches with PAST, IMAGES, INTENT, and NAME.
        device: Device to run inference on. Defaults to CUDA if available.
    
    Returns:
        List of FrameTrajectoryPredictions for submission.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predictions = []

    model = torch.compile(model, mode="default")
    
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Generating submission predictions"):
            past = batch['PAST'].to(device, non_blocking=True)
            if "IMAGES_JPEG" in batch:
                images = decode_batch_jpeg(batch["IMAGES_JPEG"], device)
            elif "IMAGES" in batch:
                images = [img.to(device, non_blocking=True) for img in batch["IMAGES"]]
            else:
                raise KeyError("Batch must contain either 'IMAGES_JPEG' or 'IMAGES'.")
            intent = batch['INTENT'].to(device, non_blocking=True)
            names = batch['NAME']  # List of strings, keep on CPU
            
            # Model forward pass
            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}
            pred_future = model(model_inputs)  # (B, 40)

            pred_futures, pred_depth, pred_scores = pred_future["trajectory"], pred_future.get("depth", None), pred_future.get("scores", None)

            t2 = 20 * 2
            k_modes = getattr(model, "n_proposals", 1)
            if pred_futures.ndim == 2 and pred_futures.shape[1] == k_modes * t2:
                pred_futures = pred_futures.view(pred_futures.size(0), k_modes, t2)

            if pred_scores is None:
                raise ValueError("DeepMonocularModel must output scores")
            else:
                best_idx = pred_scores.argmin(dim=1)  # (B,)

            pred_future = pred_futures[torch.arange(pred_futures.size(0)), best_idx, :]  # (B, 40)

            pred_future = pred_future.view(-1, 20, 2).cpu().numpy()  # (B, 20, 2)
            
            # Create submission entries for each sample in the batch
            for i, name in enumerate(names):
                pos_x = pred_future[i, :, 0].astype(np.float32)
                pos_y = pred_future[i, :, 1].astype(np.float32)
                
                trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(
                    pos_x=pos_x,
                    pos_y=pos_y
                )
                frame_prediction = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(
                    frame_name=name,
                    trajectory=trajectory
                )
                predictions.append(frame_prediction)
    
    return predictions

def serialize_and_save_submission(predictions: List[wod_e2ed_submission_pb2.FrameTrajectoryPredictions], output_file: str, num_shards: int = 8):
    # Pack for submission.
    num_submission_shards = num_shards
    submission_file_base = output_file
    if not os.path.exists(submission_file_base):
      os.makedirs(submission_file_base)
    # Use Waymo's expected naming format: submission.binproto-XXXXX-of-YYYYY
    sub_file_names = [
        os.path.join(submission_file_base, f'submission.binproto-{i:05d}-of-{num_submission_shards:05d}')
        for i in range(num_submission_shards)
    ]
    # As the submission file may be large, we shard them into different chunks.
    submissions = []
    num_predictions_per_shard =  math.ceil(len(predictions) / num_submission_shards)
    for i in range(num_submission_shards):
      start = i * num_predictions_per_shard
      end = (i + 1) * num_predictions_per_shard
      submissions.append(
          wod_e2ed_submission_pb2.E2EDChallengeSubmission(
              predictions=predictions[start:end]))
    
    for i, shard in enumerate(submissions):
        shard.submission_type  =  wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
        shard.authors[:] = ['Manav Gagvani']  
        shard.affiliation = 'Purdue University'
        shard.account_name = 'manavgagvani@gmail.com' 
        shard.unique_method_name = 'TrajScorer' 
        shard.method_link = ''  # TODO: make project page + add link
        shard.description = 'ViT features -> cross-attention -> scorer + multiple trajectory heads'
        shard.uses_public_model_pretraining = True
        shard.public_model_names.extend(['Perception Encoder Small'])
        shard.num_model_parameters = "36m" # TODO: don't hardcode, can be found with parameters()
        with open(sub_file_names[i], 'wb') as fp:
            fp.write(shard.SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save submission file')
    args = parser.parse_args()

    # Data 
    test_dataset = WaymoE2E(indexFile='index_test.pkl', data_dir=args.data_dir, images=True, n_items=None)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=16, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)

    # Load model
    model = load_model(args.checkpoint)
    
    # Generate submission data
    predictions = generate_submission_data(model, test_loader)

    # Serialize and save submission
    serialize_and_save_submission(predictions, args.output_file)


