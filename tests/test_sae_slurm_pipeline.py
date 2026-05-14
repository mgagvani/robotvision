from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_script(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def test_sae_pipeline_uses_dependency_driven_slurm_jobs() -> None:
    pipeline = read_script("sbatch/sae_pipeline.sbatch")
    assert "--parsable" in pipeline
    assert "--dependency=" in pipeline
    assert "extract_tok.sbatch" in pipeline
    assert "train_SAE.sbatch" in pipeline
    assert "sae_analysis_block.sbatch" in pipeline
    assert "sae_gated_ade.sbatch" in pipeline
    assert "sae_ade_intervention.sbatch" in pipeline
    assert "sae_merge_analysis.sbatch" in pipeline
    assert "sae_runs_topk_aux_drvla_v1" in pipeline
    assert 'FEATURES=all' in pipeline


def test_sae_helper_scripts_are_block_and_array_aware() -> None:
    extract_script = read_script("sbatch/extract_tok.sbatch")
    train_script = read_script("sbatch/train_SAE.sbatch")
    gated_script = read_script("sbatch/sae_gated_ade.sbatch")
    ade_script = read_script("sbatch/sae_ade_intervention.sbatch")
    analysis_script = read_script("sbatch/sae_analysis_block.sbatch")
    merge_script = read_script("sbatch/sae_merge_analysis.sbatch")

    assert 'SPLIT="${SPLIT:-train}"' in extract_script
    assert 'RUN_ROOT="${RUN_ROOT:-/scratch/negishi/mgagvani/robotvision_scratch/sae_runs_topk_aux_drvla_v1}"' in extract_script
    assert 'SAE_BLOCK="${SAE_BLOCK:-0}"' in train_script
    assert '--blocks "${SAE_BLOCK}"' in train_script
    assert 'SLURM_ARRAY_TASK_ID' in gated_script
    assert 'SLURM_ARRAY_TASK_ID' in ade_script
    assert 'FEATURES="${FEATURES:-all}"' in ade_script
    assert "analyze_sae_intent.py" in analysis_script
    assert "analyze_sae_error.py" in analysis_script
    assert "analyze_sae_control.py" in analysis_script
    assert "merge_sae_block_analysis.py" in merge_script
