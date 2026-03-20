#!/bin/bash
#SBATCH --job-name=lead-waymo-e2e
#SBATCH --output=lead-waymo-e2e-%j.out
#SBATCH --error=lead-waymo-e2e-%j.err

#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=5

# >>> Activate conda environment <<<
module load anaconda
source activate /scratch/gilbreth/shar1159/conda_envs/lead_ltf

# >>> Navigate to lead_ltf project directory <<<
# Required — train.py uses package-relative imports (lead.*)
cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e/models/lead_ltf || exit 1

# logdir is required; all other Waymo settings (use_waymo_e2e_data=true,
# data paths, epochs=20, batch_size=64) are already defaults in config_training.py
export LEAD_TRAINING_CONFIG="logdir=outputs/training/waymo_e2e_${SLURM_JOB_ID} batch_size=16 carla_root=data/waymo"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

nvidia-smi

srun python -m lead.training.train
