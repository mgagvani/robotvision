from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import numpy as np
import torch
from torch.utils.data import DataLoader
from loader import WaymoE2E
from models.monocular import MonocularModel, SAMFeatures
from models.base_model import collate_with_images
import os
import argparse
from typing import List
import math
from tqdm import tqdm

def load_model(checkpoint_path: str, device: torch.device = None) -> MonocularModel:
    """Load a trained MonocularModel from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.ckpt file)
        device: Device to load the model on. Defaults to CUDA if available.
    
    Returns:
        The loaded MonocularModel in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_dim = 16 * 6  # Past: (B, 16, 6)
    out_dim = 20 * 2  # Future: (B, 20, 2)
    model = MonocularModel(in_dim=in_dim, out_dim=out_dim, feature_extractor=SAMFeatures())

    # Handle torch.compile'd model checkpoints (same fix as viz.py)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    mapped = {}
    for k, v in state.items():
        k = k.replace("model._orig_mod.", "").replace("model.", "")
        if k.startswith("features.sam_pos_embed"):
            k = k.replace("features.sam_pos_embed", "features.sam_model.model.pos_embed")
        elif k.startswith("features.sam_pos_embed_window"):
            k = k.replace("features.sam_pos_embed_window", "features.sam_model.model.pos_embed_window")
        elif k.startswith("features.sam_patch_embed"):
            k = k.replace("features.sam_patch_embed", "features.sam_model.model.patch_embed")
        elif k.startswith("features.sam_blocks"):
            k = k.replace("features.sam_blocks", "features.sam_model.model.blocks")
        mapped[k] = v

    model.load_state_dict(mapped, strict=True)
    model.to(device)
    model.eval()
    
    return model

def generate_submission_data(
    model: MonocularModel, 
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

    model.compile(mode="reduce-overhead")
    
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Generating submission predictions"):
            past = batch['PAST'].to(device, non_blocking=True)
            images = [img.to(device, non_blocking=True) for img in batch['IMAGES']]
            intent = batch['INTENT'].to(device, non_blocking=True)
            names = batch['NAME']  # List of strings, keep on CPU
            
            # Model forward pass
            model_inputs = {'PAST': past, 'IMAGES': images, 'INTENT': intent}
            pred_future = model(model_inputs)  # (B, 40)
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
        shard.unique_method_name = 'TrajSAM' 
        shard.method_link = ''  # TODO: make project page + add link
        shard.description = 'SAM features -> cross-attention -> regress points'
        shard.uses_public_model_pretraining = True
        shard.public_model_names.extend(['SAM 2.1'])
        shard.num_model_parameters = "30m" # TODO: don't hardcode, can be found with parameters()
        with open(sub_file_names[i], 'wb') as fp:
            fp.write(shard.SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save submission file')
    args = parser.parse_args()

    # Data 
    test_dataset = WaymoE2E(batch_size=16, indexFile='index_test.pkl', data_dir=args.data_dir, images=True, n_items=None)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=32, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)

    # Load model
    model = load_model(args.checkpoint)
    
    # Generate submission data
    predictions = generate_submission_data(model, test_loader)

    # Serialize and save submission
    serialize_and_save_submission(predictions, args.output_file)


