"""
Trajectory visualization with camera projection.
==================================================

Produces an animated video of model-predicted driving trajectories overlaid on
the front-3 camera panoramic view (FRONT_LEFT | FRONT | FRONT_RIGHT) from the
Waymo Open Dataset End-to-End Camera challenge.

Each output frame shows:
  - The stitched panoramic image from three front-facing cameras.
  - **Red** trajectory: the model's best (lowest-scoring) proposal.
  - **Light-red** trajectories: the remaining top-K proposals.
  - **Green** trajectory: the ground-truth future path.
  - A blue HUD overlay with ADE, FDE, and the driving intent
    (STRAIGHT / LEFT / RIGHT / UNKNOWN).

Quick-start
-----------
Generate a short clip (50 frames, single driving segment)::

    python viz_camera_projection.py \\
        --data_root /anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0 \\
        --model_path /anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/checkpoints/camera-e2e-epoch=08-val_loss=4.71.ckpt \\
        --num_samples 50 \\
        --output_dir ./visualizations

Generate a ~5-minute video (multiple segments auto-selected)::

    python viz_camera_projection.py \\
        --data_root /anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0 \\
        --model_path /anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/checkpoints/camera-e2e-epoch=08-val_loss=4.71.ckpt \\
        --num_samples 1200 \\
        --output_dir ./visualizations

CLI arguments
-------------
--data_root     Path to the directory containing tfrecord files.
--model_path    Path to a PyTorch Lightning ``.ckpt`` checkpoint.
--index_file    Path to the byte-offset index pickle (default: ``index_val.pkl``).
--num_samples   Total number of frames to render.  When > 230 (the max segment
                length in the validation set), multiple segments are
                automatically concatenated.
--top_k         Number of top trajectory proposals to draw (default: 10).
--output_dir    Directory for saved video / gif (default: ``./visualizations``).
--fps           Video framerate (default: 4).  The Waymo E2E validation set has
                one frame every ~0.5 s, so 4 fps gives roughly 2x real-time.
--format        ``mp4`` (default), ``gif``, or ``both``.

Pipeline overview
-----------------
1. **Segment index** (``_build_segment_index``):
   The Waymo E2E tfrecord files interleave one frame per driving segment per
   file -- consecutive byte offsets in a single file belong to *different*
   segments.  To play back a coherent driving sequence we must know which byte
   ranges belong to which segment.

   On the first run, every entry in ``index_val.pkl`` is scanned (~2 min, ~107 k
   entries at ~900 entries/s).  The resulting mapping
   ``{segment_hash -> [(frame_idx, filename, offset, length), ...]}``
   is cached to ``index_val.segments.pkl`` so subsequent runs load instantly.

2. **Frame selection** (``WaymoE2ESequential``):
   Given the segment index, the dataset class selects one or more driving
   segments and yields their frames in temporal order.  Two modes:

   * ``total_frames=N`` (no ``frames_per_sequence``): auto-select enough of the
     longest available segments (shuffled by seed for variety) until at least N
     frames are collected.  Each segment is used in its entirety.
   * ``frames_per_sequence=N``: pick exactly N contiguous frames per segment,
     with a random starting offset.

3. **Model inference** (``generate_viz_frames``):
   Each frame is fed to ``DeepMonocularModel`` which outputs ``K`` trajectory
   proposals (each ``T`` waypoints at ``dt=0.25 s``) and per-proposal scores
   (predicted ADE; lower = better).

4. **Projection** (``project_trajectory_to_image``):
   Trajectory waypoints are in the *vehicle frame* (Waymo convention:
   ``+x`` forward, ``+y`` left, ``+z`` up).  Projection to pixels:

   a. Transform vehicle -> Waymo camera frame via ``inv(extrinsic)``.
   b. Convert Waymo camera (``x`` fwd, ``y`` left, ``z`` up) to OpenCV camera
      (``x`` right, ``y`` down, ``z`` fwd):
      ``opencv = [-waymo_y, -waymo_z, waymo_x]``.
   c. Apply intrinsics + distortion with ``cv2.projectPoints``.
   d. Discard points behind the camera or outside the image bounds.

5. **Rendering** (``_stitch_front3``):
   Trajectories are drawn on each of the three front cameras independently
   (so left-turning trajectories are visible in FRONT_LEFT, etc.), then the
   three panels are resized to a common height and concatenated horizontally.

6. **Output** (``create_video`` / ``create_animated_gif``):
   Frames are written to an MP4 (via OpenCV ``VideoWriter``) or animated GIF
   (via matplotlib + pillow).

Key files
---------
index_val.pkl
    Pre-built byte-offset index: list of ``(filename, start_byte, byte_length)``
    tuples, one per frame across all tfrecord files.  Created by the dataset
    export pipeline (see ``dataset/export_dataset.py``).

index_val.segments.pkl
    Cached segment-level index built by ``_build_segment_index``.  Maps each
    segment hash to its temporally-sorted list of
    ``(frame_idx, filename, byte_offset, byte_length)`` tuples.  Auto-generated
    on first run; delete to force a rebuild.

protos/e2e_pb2.py
    Compiled protobuf definitions for ``E2EDFrame`` (wraps camera images,
    past/future ego states, intent, and the inner ``Frame`` with calibrations).

models/monocular.py
    ``DeepMonocularModel`` -- the trajectory prediction network that consumes
    multi-camera images + past ego states and outputs scored proposals.

Notes for further development
-----------------------------
- **Adding cameras**: Extend ``FRONT3_CAMERAS`` and ``_stitch_front3`` to
  include ``SIDE_LEFT`` (4) and ``SIDE_RIGHT`` (5) for a full surround view.
  You will need a different stitching layout (e.g. 2-row grid).

- **Changing the model**: ``load_model`` handles both older (pickled model
  object in ``hyper_parameters``) and newer (state-dict-only) Lightning
  checkpoints.  If the architecture changes, update the reconstruction branch
  (the ``else`` block) and make sure ``n_proposals`` / ``horizon`` are set.

- **Segment selection seed**: Use ``--seed`` to get different segment
  combinations while keeping runs reproducible.

- **Smaller files**: Use ``--video_scale`` (for example ``0.9``) to downscale
  output frames and reduce MP4 filesize.

- **Training set visualization**: Pass ``--index_file index_train.pkl``.  The
  segment index cache is keyed by the index filename, so
  ``index_train.segments.pkl`` will be built automatically.

- **Higher framerate / real-time playback**: The dataset provides frames at
  ~2 Hz.  For smoother video, increase ``--fps`` (e.g. 10) -- this speeds up
  playback rather than interpolating new frames.

- **Trajectory representation**: The model predicts ``(x, y)`` waypoints on the
  ground plane (``z=0``).  If you add a height/elevation model, pass 3D
  trajectories ``(T, 3)`` to ``project_trajectory_to_image`` -- it already
  handles both 2D and 3D inputs.

- **Batch inference**: Currently frames are processed one at a time.  For speed,
  batch multiple frames (requires padding images to a common size).

Dependencies
------------
torch, opencv-python, numpy, Pillow, matplotlib, tqdm, protobuf
"""

import argparse
import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from protos import e2e_pb2
from models.gtrs import GTRSModel
from models.monocular import DeepMonocularModel
from models.feature_extractors import SAMFeatures


# Camera name enums from proto (dataset.proto CameraName)
CAMERA_FRONT = 1
CAMERA_FRONT_LEFT = 2
CAMERA_FRONT_RIGHT = 3

# The three front-facing cameras, in left-to-right visual order
FRONT3_CAMERAS = [CAMERA_FRONT_LEFT, CAMERA_FRONT, CAMERA_FRONT_RIGHT]


class LegacyGTRSModel(nn.Module):
    """
    GTRS checkpoint-compatible architecture variant used by older runs.

    Key differences vs current models.gtrs.GTRSModel:
      - no positional encoding on visual tokens
      - status encoder consumes only flattened past state (no intent one-hot)
      - score head is 2-layer MLP: d_model -> d_ffn -> 1
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        d_model: int = 256,
        d_ffn: int = 2048,
        n_head: int = 8,
        n_layers: int = 4,
        n_past: int = 16,
    ):
        super().__init__()
        self.features = feature_extractor
        self.d_features = sum(self.features.dims)
        h, w = self.features.data_config["input_size"][1:]
        self.n_img_tokens = (h // self.features.patch_size) * (w // self.features.patch_size)

        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(Path(__file__).parent / "vocab.npy")),
            requires_grad=False,
        )
        self.n_proposals = int(self.vocab.shape[0])
        self.horizon = int(self.vocab.shape[1])
        self.dt = 0.25
        self.max_accel = 8.0
        self.max_omega = 1.0

        self.down_conv = nn.Conv1d(self.d_features, d_model, 1, 1)
        self.vocab_embed = nn.Sequential(
            nn.Linear(self.vocab[0].numel(), d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ffn,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=n_layers,
        )

        # Keep parameterized layers at indices 1 and 3 to match legacy checkpoints.
        self.status_encoding = nn.Sequential(
            nn.Identity(),
            nn.Linear(n_past * 6, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        self.heads = nn.ModuleDict(
            {
                "scores": nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.GELU(),
                    nn.Linear(d_ffn, 1),
                )
            }
        )

    def _extract_tokens(self, front_cam: torch.Tensor) -> torch.Tensor:
        bsz, n_cam, c, h, w = front_cam.shape
        cam_inputs = front_cam.reshape(bsz * n_cam, c, h, w)
        feats_vit = self.features(cam_inputs)
        if isinstance(feats_vit, (list, tuple)):
            feats_vit = torch.cat([f.flatten(2) for f in feats_vit], dim=1)
        else:
            feats_vit = feats_vit.flatten(2)
        tokens = feats_vit.permute(0, 2, 1)
        n_tokens = tokens.shape[1]
        return tokens.reshape(bsz, n_cam, n_tokens, self.d_features)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        past = x["PAST"]
        images = x["IMAGES"]
        bsz = past.size(0)

        visual_tokens = self._extract_tokens(images[1][:, None, ...])
        visual_tokens = visual_tokens.flatten(start_dim=1, end_dim=2)
        tokens = self.down_conv(visual_tokens.transpose(1, 2)).transpose(1, 2)

        out: Dict[str, torch.Tensor] = {}
        vocab = self.vocab
        out["trajectory"] = vocab.unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz, -1)

        vocab_flat = vocab.reshape(vocab.shape[0], -1)
        embedded_vocab = self.vocab_embed(vocab_flat)
        tr_out = self.transformer(embedded_vocab.unsqueeze(0).expand(bsz, -1, -1), tokens)

        status = self.status_encoding(past.reshape(bsz, -1))
        dist_status = tr_out + status.unsqueeze(1)
        out["scores"] = self.heads["scores"](dist_status).squeeze(-1)
        return out


def _infer_model_type(hparams: dict, mapped_state: Dict[str, torch.Tensor]) -> str:
    """Infer architecture from checkpoint metadata/state keys."""
    model_name = str(hparams.get("model_name", "")).lower()
    if "gtrs" in model_name:
        return "gtrs"
    if "deepmonocular" in model_name:
        return "deepmonocular"

    keys = list(mapped_state.keys())
    if any(k.startswith("vocab_embed.") or k.startswith("status_encoding.") for k in keys):
        return "gtrs"
    return "deepmonocular"


def load_model(
    model_path: str,
    device: torch.device,
    model_type: str = "auto",
    feature_model: str = "timm/vit_pe_spatial_small_patch16_512.fb",
) -> nn.Module:
    """Load model from checkpoint (auto/GTRS/DeepMonocular)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("state_dict", {}) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {model_path}")

    # Remove "model." prefix from Lightning checkpoint
    mapped_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[6:]
        mapped_state[k] = v

    resolved_model_type = model_type
    if resolved_model_type == "auto":
        resolved_model_type = _infer_model_type(hparams, mapped_state)
    print(f"Detected model type: {resolved_model_type}")

    # Older checkpoints may store pickled model directly.
    if resolved_model_type == "deepmonocular" and "model" in hparams:
        model = hparams["model"]
    elif resolved_model_type == "gtrs":
        # Legacy GTRS checkpoints have no positional_encoding and different
        # status/score head wiring.
        use_legacy_gtrs = "positional_encoding" not in mapped_state and "status_encoding.1.weight" in mapped_state
        if use_legacy_gtrs:
            cfg = hparams.get("model_cfg", {}) if isinstance(hparams, dict) else {}
            model = LegacyGTRSModel(
                feature_extractor=SAMFeatures(
                    model_name=feature_model,
                    frozen=True,
                ),
                d_model=int(cfg.get("d_model", 256)),
                d_ffn=int(cfg.get("d_ffn", 2048)),
                n_head=int(cfg.get("n_head", 8)),
                n_layers=int(cfg.get("n_layers", 4)),
                n_past=int(cfg.get("n_past", 16)),
            )
        else:
            model = GTRSModel(
                feature_extractor=SAMFeatures(
                    model_name=feature_model,
                    frozen=True,
                ),
                out_dim=20 * 2,
            )
    elif resolved_model_type == "deepmonocular":
        out_dim = 20 * 2
        model = DeepMonocularModel(
            feature_extractor=SAMFeatures(
                model_name=feature_model,
                frozen=True,
            ),
            out_dim=out_dim,
            n_blocks=4,
        )
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    model.load_state_dict(mapped_state, strict=True)

    # Patch attributes that may be missing in older pickled model objects
    if not hasattr(model, "n_proposals"):
        if hasattr(model, "vocab"):
            model.n_proposals = int(model.vocab.shape[0])
        else:
            raise AttributeError("Model is missing n_proposals and has no vocab to infer it")
    if not hasattr(model, "horizon"):
        if hasattr(model, "traj_decoder"):
            out_features = model.traj_decoder[-1].out_features
            model.horizon = out_features // (model.n_proposals * 2)
        elif hasattr(model, "vocab"):
            model.horizon = int(model.vocab.shape[1])
        else:
            raise AttributeError("Model is missing horizon and cannot be inferred")
    if not hasattr(model, "dt"):
        model.dt = 0.25
    if not hasattr(model, "max_accel"):
        model.max_accel = 8.0
    if not hasattr(model, "max_omega"):
        model.max_omega = 1.0

    model.to(device)
    model.eval()
    return model


def decode_image(img_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to numpy array (H, W, 3)."""
    img = Image.open(BytesIO(img_bytes))
    return np.array(img)


def get_camera_calibration(
    calibrations, camera_name: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Extract camera intrinsics and extrinsics for a given camera.

    Returns:
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) camera-to-vehicle transform matrix
        dist_coeffs: (5,) distortion coefficients
        width, height: image dimensions
    """
    for calib in calibrations:
        if calib.name == camera_name:
            # Intrinsic: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
            intr = list(calib.intrinsic)
            f_u, f_v, c_u, c_v = intr[0], intr[1], intr[2], intr[3]
            k1, k2, p1, p2, k3 = (
                intr[4],
                intr[5],
                intr[6],
                intr[7],
                intr[8] if len(intr) > 8 else 0,
            )

            intrinsic = np.array(
                [[f_u, 0, c_u], [0, f_v, c_v], [0, 0, 1]], dtype=np.float64
            )

            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            # Extrinsic: 4x4 row-major transform (camera frame to vehicle frame)
            extr = np.array(list(calib.extrinsic.transform)).reshape(4, 4)

            return intrinsic, extr, dist_coeffs, calib.width, calib.height

    raise ValueError(f"Camera {camera_name} not found in calibrations")


def project_trajectory_to_image(
    trajectory: np.ndarray,  # (T, 2) or (T, 3) in vehicle frame
    intrinsic: np.ndarray,  # (3, 3)
    extrinsic: np.ndarray,  # (4, 4) camera-to-vehicle transform
    dist_coeffs: np.ndarray,  # (5,) distortion coefficients
    width: int,
    height: int,
) -> np.ndarray:
    """
    Project 3D trajectory points from vehicle frame to 2D image pixels.

    Vehicle frame (Waymo): +x = forward, +y = left, +z = up
    Waymo camera frame: Also aligned with vehicle (+x = forward, +y = left, +z = up)
                        based on nearly-identity rotation in extrinsic

    For projection, we need to convert to OpenCV camera convention:
        OpenCV camera: +x = right, +y = down, +z = forward (optical axis)

    Mapping from Waymo camera to OpenCV camera:
        OpenCV_x = -Waymo_y (left -> right)
        OpenCV_y = -Waymo_z (up -> down)
        OpenCV_z = Waymo_x (forward -> depth)

    Returns:
        pixels: (T, 2) pixel coordinates, NaN for points behind camera or outside image
    """
    T = trajectory.shape[0]

    # Add z=0 if only (x, y) provided (ground plane assumption)
    if trajectory.shape[1] == 2:
        trajectory_3d = np.hstack([trajectory, np.zeros((T, 1))])
    else:
        trajectory_3d = trajectory

    # Convert to homogeneous coordinates
    pts_vehicle = np.hstack([trajectory_3d, np.ones((T, 1))])  # (T, 4)

    # Transform from vehicle frame to Waymo camera frame
    # extrinsic is camera-to-vehicle, so we need vehicle-to-camera (inverse)
    vehicle_to_camera = np.linalg.inv(extrinsic)
    pts_waymo_cam = (vehicle_to_camera @ pts_vehicle.T).T[
        :, :3
    ]  # (T, 3) in Waymo camera frame

    # Convert from Waymo camera convention to OpenCV camera convention
    # Waymo camera: x=forward, y=left, z=up
    # OpenCV camera: x=right, y=down, z=forward
    pts_opencv_cam = np.zeros_like(pts_waymo_cam)
    pts_opencv_cam[:, 0] = -pts_waymo_cam[:, 1]  # OpenCV x = -Waymo y (left -> right)
    pts_opencv_cam[:, 1] = -pts_waymo_cam[:, 2]  # OpenCV y = -Waymo z (up -> down)
    pts_opencv_cam[:, 2] = pts_waymo_cam[:, 0]  # OpenCV z = Waymo x (forward -> depth)

    # Filter points behind camera (z <= 0 in OpenCV convention)
    valid_depth = pts_opencv_cam[:, 2] > 0.1  # at least 10cm in front

    pixels = np.full((T, 2), np.nan)

    if np.any(valid_depth):
        valid_pts = pts_opencv_cam[valid_depth]

        # Use OpenCV projectPoints with distortion
        # Since we're already in OpenCV camera frame, use identity rotation and zero translation
        rvec = np.zeros(3)
        tvec = np.zeros(3)

        pts_2d, _ = cv2.projectPoints(
            valid_pts.reshape(-1, 1, 3), rvec, tvec, intrinsic, dist_coeffs
        )
        pts_2d = pts_2d.reshape(-1, 2)

        # Check if points are within image bounds
        in_bounds = (
            (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 0] < width)
            & (pts_2d[:, 1] >= 0)
            & (pts_2d[:, 1] < height)
        )

        # Fill in valid pixels
        valid_indices = np.where(valid_depth)[0]
        for i, (idx, pt, valid) in enumerate(zip(valid_indices, pts_2d, in_bounds)):
            if valid:
                pixels[idx] = pt

    return pixels


def draw_trajectory_on_image(
    image: np.ndarray,
    trajectory_pixels: np.ndarray,  # (T, 2)
    color: Tuple[int, int, int],
    thickness: int = 3,
    point_radius: int = 5,
    alpha: float = 0.8,
) -> np.ndarray:
    """Draw trajectory line and points on image."""
    img = image.copy()

    # Filter out NaN points
    valid = ~np.isnan(trajectory_pixels).any(axis=1)
    valid_pts = trajectory_pixels[valid].astype(np.int32)

    if len(valid_pts) < 2:
        return img

    # Create overlay for alpha blending
    overlay = img.copy()

    # Draw lines connecting consecutive valid points
    for i in range(len(valid_pts) - 1):
        pt1 = tuple(valid_pts[i])
        pt2 = tuple(valid_pts[i + 1])
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw points
    for i, pt in enumerate(valid_pts):
        # Vary point size based on time (larger = later)
        r = point_radius + i // 4
        cv2.circle(overlay, tuple(pt), r, color, -1, cv2.LINE_AA)

    # Alpha blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img


def _build_segment_index(
    index_file: str,
    data_dir: str,
) -> Dict[str, List[Tuple[int, str, int, int]]]:
    """
    Build an index mapping each driving segment to its frames.

    Scans every entry in the dataset to extract the context name, which
    encodes ``{segment_hash}-{frame_idx}``.  Results are cached next to
    *index_file* so subsequent runs are instant.

    Returns:
        mapping  segment_hash -> sorted list of
            (frame_idx, filename, byte_offset, byte_length)
    """
    cache_path = Path(index_file).with_suffix(".segments.pkl")
    if cache_path.exists():
        print(f"Loading cached segment index from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Building segment index (one-time, takes ~2 min) ...")
    with open(index_file, "rb") as f:
        indexes = pickle.load(f)

    by_segment: Dict[str, List[Tuple[int, str, int, int]]] = defaultdict(list)
    current_file = None
    current_fn: Optional[str] = None

    for fn, start, length in tqdm(indexes, desc="Scanning segments"):
        if fn != current_fn:
            if current_file:
                current_file.close()
            current_file = open(os.path.join(data_dir, fn), "rb")
            current_fn = fn
        current_file.seek(start)
        data = current_file.read(length)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(data)
        ctx = frame.frame.context.name
        parts = ctx.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            seg, idx = parts[0], int(parts[1])
        else:
            seg, idx = ctx, 0
        by_segment[seg].append((idx, fn, start, length))

    if current_file:
        current_file.close()

    # Sort each segment's frames by frame index
    for seg in by_segment:
        by_segment[seg].sort()

    with open(cache_path, "wb") as f:
        pickle.dump(dict(by_segment), f)
    print(f"Segment index cached to {cache_path} ({len(by_segment)} segments)")
    return dict(by_segment)


class WaymoE2ESequential:
    """Dataset that yields temporally-ordered frames from one or more driving segments."""

    def __init__(
        self,
        index_file: str,
        data_dir: str,
        num_sequences: int = 1,
        frames_per_sequence: Optional[int] = None,
        total_frames: Optional[int] = None,
        min_segment_length: int = 50,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        num_sequences : int
            Number of driving segments to concatenate.
        frames_per_sequence : int, optional
            If given, take exactly this many frames per segment.
        total_frames : int, optional
            If given (and frames_per_sequence is None), automatically choose
            enough of the longest segments to reach *total_frames* frames,
            using each segment's full length.
        min_segment_length : int
            Ignore segments shorter than this.
        """
        self.data_dir = data_dir

        seg_index = _build_segment_index(index_file, data_dir)

        import random
        rng = random.Random(seed)

        if total_frames is not None and frames_per_sequence is None:
            # Auto-select: pick longest segments until we have enough frames
            valid = [
                (seg, frames)
                for seg, frames in seg_index.items()
                if len(frames) >= min_segment_length
            ]
            valid.sort(key=lambda x: -len(x[1]))
            if not valid:
                raise ValueError(
                    f"No segments with >= {min_segment_length} frames "
                    f"(max = {max(len(v) for v in seg_index.values())})"
                )
            # Shuffle the top candidates so we get variety across runs/seeds
            # but still pick long ones first
            rng.shuffle(valid)
            chosen_segs = []
            collected = 0
            for seg, frames in valid:
                chosen_segs.append((seg, frames))
                collected += len(frames)
                if collected >= total_frames:
                    break
            print(
                f"Selected {len(chosen_segs)} segments "
                f"({collected} frames, ~{collected / 4 / 60:.1f} min at 4fps)"
            )
        else:
            # Original behaviour: pick N segments with at least fps frames each
            fps = frames_per_sequence or 50
            valid_segments = [
                seg
                for seg, frames in seg_index.items()
                if len(frames) >= fps
            ]
            if not valid_segments:
                raise ValueError(
                    f"No segments with >= {fps} frames "
                    f"(max = {max(len(v) for v in seg_index.values())})"
                )
            chosen_keys = rng.sample(
                valid_segments, min(num_sequences, len(valid_segments))
            )
            chosen_segs = [(seg, seg_index[seg]) for seg in chosen_keys]

        # Build the flat entry list, taking a contiguous window per segment
        self.selected_entries: List[Tuple[str, int, int]] = []
        for seg, frames in chosen_segs:
            # frames is already sorted by frame_idx
            if frames_per_sequence is not None:
                max_start = max(0, len(frames) - frames_per_sequence)
                start_idx = rng.randint(0, max_start)
                window = frames[start_idx : start_idx + frames_per_sequence]
            else:
                window = frames  # use full segment
            self.selected_entries.extend((fn, s, l) for _, fn, s, l in window)

        # Keep exact requested duration when total_frames is provided.
        if total_frames is not None and len(self.selected_entries) > total_frames:
            self.selected_entries = self.selected_entries[:total_frames]

    def __len__(self):
        return len(self.selected_entries)

    def __iter__(self):
        current_file = None
        current_filename = None

        for filename, start_byte, byte_length in self.selected_entries:
            if filename != current_filename:
                if current_file:
                    current_file.close()
                current_file = open(os.path.join(self.data_dir, filename), "rb")
                current_filename = filename

            current_file.seek(start_byte)
            protobuf = current_file.read(byte_length)

            frame = e2e_pb2.E2EDFrame()
            frame.ParseFromString(protobuf)

            # Extract past/future states
            past = np.stack(
                [
                    frame.past_states.pos_x,
                    frame.past_states.pos_y,
                    frame.past_states.vel_x,
                    frame.past_states.vel_y,
                    frame.past_states.accel_x,
                    frame.past_states.accel_y,
                ],
                axis=-1,
            ).astype(np.float32)

            future = np.stack(
                [
                    frame.future_states.pos_x,
                    frame.future_states.pos_y,
                ],
                axis=-1,
            ).astype(np.float32)

            # Build image tensors for the model (all cameras, proto order)
            # and decoded numpy arrays keyed by camera name for visualization
            images = []
            cam_images: Dict[int, np.ndarray] = {}
            for img in frame.frame.images:
                img_array = decode_image(img.image)
                images.append(torch.from_numpy(img_array).permute(2, 0, 1))  # (3, H, W)
                if img.name in FRONT3_CAMERAS:
                    cam_images[img.name] = img_array

            # Get calibrations for the three front cameras
            calibrations: Dict[int, dict] = {}
            for cam_id in FRONT3_CAMERAS:
                try:
                    intr, extr, dist, w, h = get_camera_calibration(
                        frame.frame.context.camera_calibrations, cam_id
                    )
                    calibrations[cam_id] = {
                        "intrinsic": intr,
                        "extrinsic": extr,
                        "dist_coeffs": dist,
                        "width": w,
                        "height": h,
                    }
                except ValueError:
                    pass  # camera not present in this frame

            yield {
                "PAST": past,
                "FUTURE": future,
                "IMAGES": images,
                "CAM_IMAGES": cam_images,
                "INTENT": frame.intent,
                "NAME": frame.frame.context.name,
                "CALIBRATIONS": calibrations,
            }

        if current_file:
            current_file.close()


def _draw_trajectories_on_cam(
    cam_img: np.ndarray,
    calib: dict,
    best_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    top_k_trajectories: np.ndarray,
) -> np.ndarray:
    """Project and draw predicted + GT trajectories onto a single camera image."""
    intrinsic = calib["intrinsic"]
    extrinsic = calib["extrinsic"]
    dist_coeffs = calib["dist_coeffs"]
    width = calib["width"]
    height = calib["height"]

    img = cam_img.copy()

    # Lighter colours for non-best top-k proposals
    colors_topk = [
        (255, 150, 150),
        (255, 170, 170),
        (255, 180, 180),
        (255, 190, 190),
        (255, 200, 200),
        (255, 210, 210),
        (255, 220, 220),
        (255, 230, 230),
        (255, 240, 240),
        (255, 245, 245),
    ]

    # Draw non-best top-k proposals
    for i, traj in enumerate(top_k_trajectories[1:]):
        color_idx = min(i, len(colors_topk) - 1)
        pixels = project_trajectory_to_image(
            traj, intrinsic, extrinsic, dist_coeffs, width, height
        )
        img = draw_trajectory_on_image(
            img, pixels, colors_topk[color_idx],
            thickness=2, point_radius=3, alpha=0.5,
        )

    # Draw best prediction (red)
    best_pixels = project_trajectory_to_image(
        best_trajectory, intrinsic, extrinsic, dist_coeffs, width, height
    )
    img = draw_trajectory_on_image(
        img, best_pixels, (255, 0, 0), thickness=4, point_radius=6, alpha=0.9,
    )

    # Draw ground truth (green)
    gt_pixels = project_trajectory_to_image(
        gt_trajectory, intrinsic, extrinsic, dist_coeffs, width, height
    )
    img = draw_trajectory_on_image(
        img, gt_pixels, (0, 255, 0), thickness=4, point_radius=6, alpha=0.9,
    )

    return img


def _stitch_front3(
    cam_images: Dict[int, np.ndarray],
    calibrations: Dict[int, dict],
    best_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    top_k_trajectories: np.ndarray,
) -> np.ndarray:
    """
    Draw trajectories on each of the three front cameras and stitch them
    horizontally in left-to-right order.  All panels are resized to match
    the height of the FRONT camera so they tile cleanly.
    """
    panels = []
    # Target height = FRONT camera (always present)
    target_h = cam_images[CAMERA_FRONT].shape[0]

    for cam_id in FRONT3_CAMERAS:
        if cam_id not in cam_images or cam_id not in calibrations:
            continue
        panel = _draw_trajectories_on_cam(
            cam_images[cam_id],
            calibrations[cam_id],
            best_trajectory,
            gt_trajectory,
            top_k_trajectories,
        )
        h, w = panel.shape[:2]
        if h != target_h:
            scale = target_h / h
            panel = cv2.resize(
                panel, (int(w * scale), target_h), interpolation=cv2.INTER_LINEAR
            )
        panels.append(panel)

    if not panels:
        raise RuntimeError("No front camera images available")

    return np.concatenate(panels, axis=1)


def generate_viz_frames(
    model: nn.Module,
    data_root: str,
    index_file: str,
    num_samples: int,
    device: torch.device,
    top_k: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate visualization frames with model predictions projected onto all
    three front cameras (FRONT_LEFT | FRONT | FRONT_RIGHT), stitched into a
    panoramic view.

    Returns list of dicts containing:
        - image: np.ndarray (H, W, 3) panoramic image with trajectories drawn
        - ade, fde: error metrics
    """
    dataset = WaymoE2ESequential(
        index_file=index_file,
        data_dir=data_root,
        total_frames=num_samples,
        seed=seed,
    )

    model.eval()
    viz_frames = []

    with torch.no_grad():
        for sample in tqdm(
            dataset, desc="Generating visualizations", total=len(dataset)
        ):
            past = torch.from_numpy(sample["PAST"]).unsqueeze(0).to(device)
            future = torch.from_numpy(sample["FUTURE"])
            images = [img.unsqueeze(0).to(device) for img in sample["IMAGES"]]
            intent = torch.tensor([sample["INTENT"]]).to(device)

            # Get model prediction
            output = model({"PAST": past, "IMAGES": images, "INTENT": intent})
            traj_pred = output["trajectory"]  # (1, K*T*2)
            scores = output["scores"]  # (1, K)

            # Reshape predictions
            n_proposals = model.n_proposals
            horizon = model.horizon
            traj_pred = (
                traj_pred.view(1, n_proposals, horizon, 2).squeeze(0).cpu().numpy()
            )  # (K, T, 2)
            scores = scores.squeeze(0).cpu().numpy()  # (K,)

            # Get top-k proposals by score (lower score = better, since scorer predicts ADE)
            top_k_indices = np.argsort(scores)[: min(top_k, len(scores))]
            top_k_trajectories = traj_pred[top_k_indices]  # (top_k, T, 2)

            best_idx = top_k_indices[0]
            best_trajectory = traj_pred[best_idx]  # (T, 2)
            gt_trajectory = future.numpy()  # (T, 2)

            # Stitch front-3 panoramic image with trajectories drawn
            pano = _stitch_front3(
                sample["CAM_IMAGES"],
                sample["CALIBRATIONS"],
                best_trajectory,
                gt_trajectory,
                top_k_trajectories,
            )

            # Calculate metrics
            ade = np.mean(np.linalg.norm(best_trajectory - gt_trajectory, axis=1))
            fde = np.linalg.norm(best_trajectory[-1] - gt_trajectory[-1])

            # Decode intent enum
            _INTENT_NAMES = {0: "UNKNOWN", 1: "STRAIGHT", 2: "LEFT", 3: "RIGHT"}
            intent_str = _INTENT_NAMES.get(sample["INTENT"], f"?({sample['INTENT']})")

            # Add legend / metrics text (blue in BGR = (255, 180, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 180, 0)  # light blue in BGR
            pano_bgr = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)
            cv2.putText(
                pano_bgr,
                f"ADE: {ade:.2f}m | FDE: {fde:.2f}m | Intent: {intent_str}",
                (10, 30),
                font, 0.8, text_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                pano_bgr,
                "Red: Predicted | Green: Ground Truth",
                (10, 60),
                font, 0.7, text_color, 2, cv2.LINE_AA,
            )
            pano = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

            viz_frames.append(
                {
                    "image": pano,
                    "ade": ade,
                    "fde": fde,
                    "name": sample["NAME"],
                }
            )

    return viz_frames


def create_video(
    viz_frames: List[Dict],
    output_path: str,
    fps: int = 4,
    scale: float = 1.0,
):
    """Create MP4 video from visualization frames."""
    if not viz_frames:
        print("No frames to save")
        return
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    # Get frame dimensions from first frame
    h, w = viz_frames[0]["image"].shape[:2]
    h_out = max(1, int(round(h * scale)))
    w_out = max(1, int(round(w * scale)))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))

    for frame_data in viz_frames:
        img_rgb = frame_data["image"]
        if scale != 1.0:
            img_rgb = cv2.resize(img_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

    out.release()
    print(f"Video saved to {output_path}")


def create_animated_gif(
    viz_frames: List[Dict],
    output_path: str,
    fps: int = 4,
):
    """Create animated GIF from visualization frames."""
    if not viz_frames:
        print("No frames to save")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    def init():
        ax.clear()
        ax.axis("off")
        return []

    def animate(frame_idx):
        ax.clear()
        ax.axis("off")
        frame_data = viz_frames[frame_idx]
        ax.imshow(frame_data["image"])
        ax.set_title(
            f"Frame {frame_idx + 1}/{len(viz_frames)} | "
            f"ADE: {frame_data['ade']:.2f}m | FDE: {frame_data['fde']:.2f}m"
        )
        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(viz_frames),
        interval=1000 // fps,
        blit=False,
    )

    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"GIF saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trajectory predictions projected onto front camera images"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Waymo E2E dataset directory containing tfrecord files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["auto", "deepmonocular", "gtrs"],
        default="auto",
        help="Model architecture to load. Use auto to infer from checkpoint.",
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        default="timm/vit_pe_spatial_small_patch16_512.fb",
        help="Backbone model name used when reconstructing checkpoints.",
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default="index_val.pkl",
        help="Path to index pickle file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of sequential samples to visualize",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top proposals to show",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save output video",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for output video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting/ordering segments",
    )
    parser.add_argument(
        "--video_scale",
        type=float,
        default=1.0,
        help="Uniform output scale for MP4 (e.g., 0.9 to reduce filesize slightly)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mp4", "gif", "both"],
        default="mp4",
        help="Output format",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(
        args.model_path,
        device,
        model_type=args.model_type,
        feature_model=args.feature_model,
    )

    # Generate visualization frames
    print(f"Generating {args.num_samples} sequential visualization frames...")
    viz_frames = generate_viz_frames(
        model=model,
        data_root=args.data_root,
        index_file=args.index_file,
        num_samples=args.num_samples,
        device=device,
        top_k=args.top_k,
        seed=args.seed,
    )

    # Calculate overall metrics
    ades = [f["ade"] for f in viz_frames]
    fdes = [f["fde"] for f in viz_frames]
    print(f"\nMetrics over {len(viz_frames)} samples:")
    print(f"  ADE: {np.mean(ades):.3f} +/- {np.std(ades):.3f} m")
    print(f"  FDE: {np.mean(fdes):.3f} +/- {np.std(fdes):.3f} m")

    # Save outputs
    if args.format in ["mp4", "both"]:
        mp4_path = os.path.join(args.output_dir, f"trajectory_camera_top{args.top_k}.mp4")
        create_video(viz_frames, mp4_path, fps=args.fps, scale=args.video_scale)

    if args.format in ["gif", "both"]:
        gif_path = os.path.join(args.output_dir, f"trajectory_camera_top{args.top_k}.gif")
        create_animated_gif(viz_frames, gif_path, fps=args.fps)


if __name__ == "__main__":
    main()
