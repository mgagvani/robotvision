#!/bin/bash
#SBATCH --job-name=scene-analysis
#SBATCH --output=scene-analysis-%j.out
#SBATCH --error=scene-analysis%j.err

#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00           # 12 hours
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# >>> Activate conda environment <<<
module load anaconda
source activate /scratch/gilbreth/shar1159/conda_envs/waymo

# >>> Navigate to your project directory <<<
cd /scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e

# >>> Run analysis on scene_loss.json <<< 
srun python scene_analysis.py
