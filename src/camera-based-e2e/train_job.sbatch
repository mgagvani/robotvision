#!/bin/bash
#SBATCH --job-name=waymo-e2e
#SBATCH --output=waymo-e2e-%j.out
#SBATCH --error=waymo-e2e-%j.err

#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ---- Optional: pass an index file as the first sbatch arg ----
# Example:
#   sbatch train_job.sbatch index_train_hard_25k.pkl
TRAIN_INDEX="${1:-index_train.pkl}"

# >>> Activate conda environment <<<
module load anaconda
source activate /scratch/gilbreth/shar1159/conda_envs/waymo

# >>> Navigate to your project directory <<<
cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e || exit 1

echo "Using TRAIN_INDEX=${TRAIN_INDEX}"

# >>> Run training <<<
srun python train.py \
  --data_dir /scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0/ \
  --batch_size 16 \
  --lr 0.001 \
  --max_epochs 10 \
  --train_index "${TRAIN_INDEX}"
