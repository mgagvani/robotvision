#!/bin/bash
#SBATCH --job-name=sae_circuit_all
#SBATCH --output=log/sae_circuit_all-%j.out
#SBATCH --error=log/sae_circuit_all-%j.err
#
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=10

export BLOCKS="${BLOCKS:-0,1,2,3}"
export TARGET_MODE="${TARGET_MODE:-composite_safety}"
export TARGET_COUNT="${TARGET_COUNT:-50}"
export FEATURE_BUDGET="${FEATURE_BUDGET:-96}"
export TOP_ACTIVE_PER_SCENE="${TOP_ACTIVE_PER_SCENE:-8}"
export MAX_SCENES_PER_FEATURE="${MAX_SCENES_PER_FEATURE:-16}"
export TOP_DOWNSTREAM_FEATURES="${TOP_DOWNSTREAM_FEATURES:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/negishi/mgagvani/robotvision_scratch/sae_repro_20260515_123234/analysis/circuit_tracing_all_blocks}"

bash /home/mgagvani/robotvision/sbatch/analyze_sae_circuit_tracing.sbatch
