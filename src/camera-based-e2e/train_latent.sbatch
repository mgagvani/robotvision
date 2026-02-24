#!/bin/bash
#SBATCH --job-name=waymo-e2e-latent
#SBATCH --output=waymo-e2e-latent-%j.out
#SBATCH --error=waymo-e2e-latent-%j.err

#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=5

# ---- Optional: pass an index file as the first sbatch arg ----
# Example:
#   sbatch train_latent.sbatch index_train_hard_25k.pkl
TRAIN_INDEX="${1:-index_train.pkl}"

# >>> Activate conda environment <<<
module load anaconda
source activate /scratch/gilbreth/shar1159/conda_envs/waymo

# >>> Navigate to your project directory <<<
cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e || exit 1

echo "Using TRAIN_INDEX=${TRAIN_INDEX}"

srun python train_latent.py \
  --data_dir /scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0/ \
  --batch_size 16 \
  --lr 1e-3 \
  --max_epochs 10 \
  --train_index "${TRAIN_INDEX}"

