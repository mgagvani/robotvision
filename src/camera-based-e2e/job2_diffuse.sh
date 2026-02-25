#!/bin/bash


#SBATCH --time=8:00:00
#SBATCH --partition=a10
#SBATCH --account=csso
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load cuda/12.6.0 anaconda
conda activate /home/bnamikas/.conda/envs/2025.06-py313/python3.10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python train_diffuse2.py \
  --data_dir /scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --score_train_k 3 \
  --score_train_every_n_steps 1 \
  --score_train_sampling_steps 20 \
  --score_rank_batch_size 8
