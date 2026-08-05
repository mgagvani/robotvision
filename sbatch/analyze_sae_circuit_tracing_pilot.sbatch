#!/bin/bash
#SBATCH --job-name=sae_circuit_pilot
#SBATCH --output=log/sae_circuit_pilot-%j.out
#SBATCH --error=log/sae_circuit_pilot-%j.err
#
#SBATCH --account=csso
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10

export BLOCKS="${BLOCKS:-3}"
export TARGET_MODE="${TARGET_MODE:-composite_safety}"
export TARGET_COUNT="${TARGET_COUNT:-16}"
export FEATURE_BUDGET="${FEATURE_BUDGET:-24}"
export TOP_ACTIVE_PER_SCENE="${TOP_ACTIVE_PER_SCENE:-6}"
export MAX_SCENES_PER_FEATURE="${MAX_SCENES_PER_FEATURE:-8}"
export TOP_DOWNSTREAM_FEATURES="${TOP_DOWNSTREAM_FEATURES:-6}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/negishi/mgagvani/robotvision_scratch/sae_repro_20260515_123234/analysis/circuit_tracing_pilot}"

bash /home/mgagvani/robotvision/sbatch/analyze_sae_circuit_tracing.sbatch
