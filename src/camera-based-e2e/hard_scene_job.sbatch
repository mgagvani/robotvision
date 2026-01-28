#!/bin/bash
#SBATCH --job-name=waymo-hard-index
#SBATCH --output=waymo-hard-index-%j.out
#SBATCH --error=waymo-hard-index-%j.err

#SBATCH --account=csso
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --time=012:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load anaconda
source activate /scratch/gilbreth/shar1159/conda_envs/waymo

cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e

srun python build_hard_index.py \
  --data_dir /scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0/ \
  --index_in index_train.pkl \
  --loss_json scene_loss.json \
  --top_k 25000 \
  --index_out index_train_hard_25k.pkl

