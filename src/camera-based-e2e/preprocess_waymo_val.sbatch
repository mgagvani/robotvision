#!/bin/bash
#SBATCH --job-name=waymo-preprocess-val
#SBATCH --output=waymo-preprocess-val-%j.out
#SBATCH --error=waymo-preprocess-val-%j.err

#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# Note: --num-workers below must match cpus-per-task

module load anaconda
source activate /scratch/gilbreth/shar1159/conda_envs/lead_ltf

cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e || exit 1

export PYTHONUNBUFFERED=1

srun python dataset/preprocess_dataset.py \
    --index-file index_val.pkl \
    --data-dir /scratch/gilbreth/shar1159/waymo_val_files/ \
    --out-dir /scratch/gilbreth/shar1159/waymo_val_preprocessed/ \
    --cameras FRONT_LEFT FRONT FRONT_RIGHT \
    --num-workers 8 \
    --chunk-size 512
