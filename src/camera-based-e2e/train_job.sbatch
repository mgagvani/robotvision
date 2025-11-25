#!/bin/bash
#SBATCH --job-name=waymo-e2e
#SBATCH --output=waymo-e2e-%j.out
#SBATCH --error=waymo-e2e-%j.err

#SBATCH --account=csso
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00           # 12 hours
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# >>> Activate conda environment <<<
module load anaconda/2023.03
source activate /scratch/gilbreth/shar1159/conda_envs/waymo

# >>> Navigate to your project directory <<<
cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e

# >>> Run training <<<
srun python train.py \
    --data_dir /scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0/ \
    --batch_size 16 \
    --lr 0.001 \
    --max_epochs 10

