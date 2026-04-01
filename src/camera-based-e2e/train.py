import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import SimpleProfiler

from matplotlib import pyplot as plt
import pandas as pd

import torch
from pathlib import Path

from loader import WaymoE2E

# Replace with your model defined in models/ 
from models.base_model import LitModel, collate_with_images
from models.monocular import DeepMonocularModel
from models.proposal_planner import ProposalPlanner
from models.feature_extractors import SAMFeatures, DINOFeatures, ResNetFeatures
from models.debug_callbacks import GradientDebugCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Waymo E2E data directory')
    parser.add_argument('--model_type', type=str, default='deep_monocular', choices=['deep_monocular', 'proposal'],
                        help='Model type: deep_monocular or proposal (proposal-centric planner)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--num_proposals', type=int, default=16, help='Number of proposals (proposal model only)')
    parser.add_argument('--num_refinement_steps', type=int, default=4, help='Refinement iterations (weight-shared, iPad default=4)')
    parser.add_argument('--smoothness_weight', type=float, default=0.01, help='Smoothness (jerk) loss weight (proposal model)')
    parser.add_argument('--collision_weight', type=float, default=0.0, help='Collision penalty weight (proposal model)')
    parser.add_argument('--comfort_weight', type=float, default=0.01, help='Comfort (curvature) loss weight (proposal model)')
    parser.add_argument('--rfs_weight', type=float, default=0.0, help='RFS loss weight (proposal model)')
    parser.add_argument('--diversity_weight', type=float, default=0.0, help='Diversity weight (0 = off, MoN loss handles diversity)')
    parser.add_argument('--prev_weight', type=float, default=0.1, help='Discount λ for intermediate proposal losses')
    parser.add_argument('--score_weight', type=float, default=1.0, help='Score loss weight (BCE is well-scaled, safe at 1.0)')
    parser.add_argument('--score_warmup_epochs', type=int, default=2, help='Epochs before score loss activates (prevents early mode collapse)')
    parser.add_argument('--score_temperature', type=float, default=5.0, help='Temperature τ for quality target exp(-ADE/τ)')
    parser.add_argument(
        '--score_loss_type',
        type=str,
        default='bce',
        choices=['bce', 'ce', 'bce_pairwise', 'listnet'],
        help='Scorer objective: bce (iPad-faithful), ce, bce_pairwise, or listnet',
    )
    parser.add_argument(
        '--score_target_type',
        type=str,
        default='l1',
        choices=['l1', 'navsim', 'rfs'],
        help='Quality target: l1 (exp(-L1/tau)), navsim (EP+Comf), or rfs (long/lat RFS at 3s/5s × optional jerk)',
    )
    parser.add_argument(
        '--no_rfs_target_comfort',
        action='store_true',
        help='For score_target_type=rfs: use pure RFS mean only (no jerk comfort multiplier)',
    )
    parser.add_argument('--score_rank_weight', type=float, default=0.2, help='Aux weight for pairwise ranking term (bce_pairwise only)')
    parser.add_argument('--score_margin', type=float, default=0.2, help='Pairwise ranking margin (bce_pairwise only)')
    parser.add_argument('--score_topk', type=int, default=0, help='Hard negative top-k for pairwise ranking (0=all negatives)')
    parser.add_argument('--comfort_jerk_threshold', type=float, default=5.0, help='Jerk threshold (m/s^3) for NAVSIM comfort sub-metric')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--log_every_n_steps', type=int, default=100, help='How often (steps) to emit trainer logs')
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'dino', 'sam'],
                        help='Backbone: resnet (default, widely available), dino, or sam')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging (use CSV only)')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model with torch.compile')
    parser.add_argument('--profile', action='store_true', help='Whether to run the profiler')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualizations (gradient norms, activation stats, proposal diagnostics)')
    parser.add_argument('--debug_log_every', type=int, default=10, help='How often (steps) to log debug metrics')
    args = parser.parse_args()

    # Data 
    # #region agent log
    try:
        _logpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-aec3fc.log")
        open(_logpath, "a").write(
            '{"sessionId":"aec3fc","hypothesisId":"H1,H2","location":"train.py:datasets","message":"train_val_config","data":{"data_dir":"%s","train_index":"index_train.pkl","val_index":"index_val.pkl","has_val_loader":true}}\n'
            % (str(args.data_dir),)
        )
    except Exception:
        pass
    # #endregion
    train_dataset = WaymoE2E(indexFile='index_train.pkl', data_dir=args.data_dir, n_items=250_000)
    test_dataset = WaymoE2E(indexFile='index_val.pkl', data_dir=args.data_dir, n_items=25_000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_with_images, persistent_workers=False, pin_memory=False)

    # Model
    out_dim = 20 * 2  # Future: (B, 20, 2)

    if args.model_type == 'proposal':
        backbone = (
            ResNetFeatures(frozen=True) if args.backbone == 'resnet'
            else DINOFeatures(frozen=True) if args.backbone == 'dino'
            else SAMFeatures(frozen=True)
        )
        model = ProposalPlanner(
            backbone=backbone,
            d_model=256,
            num_proposals=args.num_proposals,
            num_refinement_steps=args.num_refinement_steps,
            horizon=20,
            num_cams=8,
        )
    else:
        backbone = (
            ResNetFeatures(frozen=True) if args.backbone == 'resnet'
            else DINOFeatures(frozen=True) if args.backbone == 'dino'
            else SAMFeatures(frozen=True)
        )
        model = DeepMonocularModel(feature_extractor=backbone, out_dim=out_dim, n_blocks=8)
    if args.debug:
        model._debug = True

    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    lit_model = LitModel(
        model=model,
        lr=args.lr,
        smoothness_weight=args.smoothness_weight if args.model_type == 'proposal' else 0.0,
        collision_weight=args.collision_weight if args.model_type == 'proposal' else 0.0,
        comfort_weight=args.comfort_weight if args.model_type == 'proposal' else 0.0,
        rfs_weight=args.rfs_weight if args.model_type == 'proposal' else 0.0,
        diversity_weight=args.diversity_weight if args.model_type == 'proposal' else 0.0,
        score_weight=args.score_weight if args.model_type == 'proposal' else 0.0,
        score_warmup_epochs=args.score_warmup_epochs if args.model_type == 'proposal' else 0,
        score_temperature=args.score_temperature if args.model_type == 'proposal' else 5.0,
        score_loss_type=args.score_loss_type if args.model_type == 'proposal' else 'bce',
        score_target_type=args.score_target_type if args.model_type == 'proposal' else 'l1',
        score_rank_weight=args.score_rank_weight if args.model_type == 'proposal' else 0.0,
        score_margin=args.score_margin if args.model_type == 'proposal' else 0.2,
        score_topk=args.score_topk if args.model_type == 'proposal' else 0,
        comfort_jerk_threshold=args.comfort_jerk_threshold if args.model_type == 'proposal' else 5.0,
        prev_weight=args.prev_weight if args.model_type == 'proposal' else 0.1,
        rfs_target_use_comfort=not args.no_rfs_target_comfort if args.model_type == 'proposal' else True,
    )

    # We don't want to save logs or checkpoints in the home directory - it'll fill up fast
    base_path = Path(args.data_dir).parent.as_posix()
    timestamp = f"camera_e2e_{datetime.now().strftime('%Y%m%d_%H%M')}"
    loggers = [CSVLogger(base_path + "/logs", name=timestamp)]
    if not args.no_wandb:
        wandb_logger = WandbLogger(name=timestamp, save_dir=base_path + "/logs", project="robotvision", log_model=True)
        wandb_logger.watch(lit_model, log="all")
        loggers.append(wandb_logger)

    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"
    callbacks = [
        ModelCheckpoint(monitor='val_loss',
                         mode='min', 
                         save_top_k=1, 
                         dirpath=base_path + '/checkpoints',
                         filename='camera-e2e-{epoch:02d}-{val_loss:.2f}'
                        ),
    ]
    if args.debug:
        callbacks.append(GradientDebugCallback(log_every=args.debug_log_every))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=loggers,
        strategy=strategy,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 16,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        gradient_clip_algorithm="norm",
        profiler=SimpleProfiler(extended=True) if args.profile else None,
        callbacks=callbacks,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Export loss graph to visualizations/
    try:
        base_path = Path(base_path)
        run_dir = sorted((base_path / "logs").glob("camera_e2e_*"))[-1]  # newest run
        metrics = pd.read_csv(run_dir / "version_0" / "metrics.csv")
        train = metrics[metrics["train_loss"].notna()]
        val = metrics[metrics["val_loss"].notna()]

        plt.figure()
        plt.plot(train["step"], train["train_loss"], label="train_loss")
        plt.plot(val["step"], val["val_loss"], label="val_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        out = Path("./visualizations")
        out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "loss.png", dpi=200)
    except Exception as e:
        print(f"Could not save loss plot: {e}")

    if args.debug:
        try:
            from debug_viz import find_metrics_csv, load_and_merge
            from debug_viz import (
                plot_gradient_norms_by_module, plot_gradient_norms_refinement,
                plot_gradient_norms_scorer_propinit, plot_gradient_dominance,
                plot_activation_stats, plot_proposal_diagnostics,
                plot_loss_breakdown, plot_ade_regret, plot_param_norms,
                plot_summary_dashboard,
            )
            csv_path = find_metrics_csv(run_dir / "version_0")
            df = load_and_merge(csv_path)
            debug_out = base_path / "debug_plots"
            debug_out.mkdir(parents=True, exist_ok=True)
            print(f"\nGenerating debug plots to {debug_out} ...")
            plot_gradient_norms_by_module(df, debug_out)
            plot_gradient_norms_refinement(df, debug_out)
            plot_gradient_norms_scorer_propinit(df, debug_out)
            plot_gradient_dominance(df, debug_out)
            plot_activation_stats(df, debug_out)
            plot_proposal_diagnostics(df, debug_out)
            plot_loss_breakdown(df, debug_out)
            plot_ade_regret(df, debug_out)
            plot_param_norms(df, debug_out)
            plot_summary_dashboard(df, debug_out)
            print(f"Debug plots saved to {debug_out}")
        except Exception as e:
            print(f"Could not generate debug plots: {e}")





    
