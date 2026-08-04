"""Contract tests for the SAE activation hooks added to DeepMonocularModel.

These pin down the invariants the SAE tooling depends on:

* ``forward`` still emits every key the base planner/training path consumes.
* ``return_block_tokens=True`` additionally emits one query token per block.
* Re-entering the decoder from a captured token reproduces the original output,
  which is what makes latent steering meaningful.

A stub feature extractor is used so the test runs on CPU in seconds without
pulling a ViT backbone from the Hub.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[1] / "src" / "camera-based-e2e"
sys.path.insert(0, str(SRC))

from models.monocular import DeepMonocularModel  # noqa: E402

IMG = 64
PATCH = 16
DIM = 384
OUT_DIM = 40
BATCH = 3

# keys the base planner + LitModel._shared_step rely on
BASE_KEYS = (
    "trajectory",
    "trajectory_flat",
    "query_for_score",
    "scores",
    "depth",
    "controls",
)


class _StubFeatures(nn.Module):
    """Minimal stand-in for SAMFeatures: (B,3,H,W) -> (B,DIM,H/PATCH,W/PATCH)."""

    dims = (DIM,)
    patch_size = PATCH
    data_config = {"input_size": (3, IMG, IMG)}

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, PATCH, stride=PATCH)

    def forward(self, x):
        return self.proj(x)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DeepMonocularModel(feature_extractor=_StubFeatures(), out_dim=OUT_DIM).eval()


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(123)
    return {
        "PAST": torch.randn(BATCH, 16, 6),
        "INTENT": torch.tensor([1, 2, 3]),
        "IMAGES": [None, torch.randn(BATCH, 3, IMG, IMG), None],
    }


def test_forward_emits_base_keys(model, batch):
    with torch.inference_mode():
        out = model(batch)
    assert set(BASE_KEYS).issubset(out), f"missing: {set(BASE_KEYS) - set(out)}"
    assert out["trajectory_flat"].shape == (BATCH, model.n_proposals, OUT_DIM)
    assert out["query_for_score"].shape == (BATCH, model.n_proposals, DIM)
    assert out["scores"].shape == (BATCH, model.n_proposals)


def test_block_tokens_are_opt_in(model, batch):
    with torch.inference_mode():
        plain = model(batch)
        with_blocks = model(batch, return_block_tokens=True)

    assert "planner_query_tok_block_0" not in plain
    for i in range(model.cfg.n_blocks):
        assert with_blocks[f"planner_query_tok_block_{i}"].shape == (BATCH, DIM)

    # the opt-in flag must not perturb the planner output
    for key in BASE_KEYS:
        torch.testing.assert_close(plain[key], with_blocks[key], rtol=0, atol=0)


def test_last_block_token_is_the_planner_query(model, batch):
    with torch.inference_mode():
        out = model(batch, return_block_tokens=True)
    last = out[f"planner_query_tok_block_{model.cfg.n_blocks - 1}"]
    torch.testing.assert_close(out["planner_query_tok"], last, rtol=0, atol=0)


def test_replay_from_planner_query_reproduces_forward(model, batch):
    """Steering relies on this: decoding from the captured token is a no-op edit."""
    with torch.inference_mode():
        out = model(batch)
        replay = model.forward_from_planner_query_tok(out["planner_query_tok"], batch["PAST"])
    for key in ("trajectory", "trajectory_flat", "scores", "controls"):
        torch.testing.assert_close(out[key], replay[key], rtol=0, atol=0)


def test_replay_from_intermediate_block_reproduces_forward(model, batch):
    n_blocks = model.cfg.n_blocks
    with torch.inference_mode():
        out = model(batch, return_block_tokens=True)
        tokens, _ = model.prepare_visual_tokens(batch["IMAGES"])
        for start in range(n_blocks):
            replay = model.forward_from_block_query_tok(
                out[f"planner_query_tok_block_{start}"],
                batch["PAST"],
                tokens,
                start_block=start,
            )
            torch.testing.assert_close(
                out["trajectory"], replay["trajectory"], rtol=1e-5, atol=1e-6
            )


def test_replay_rejects_bad_shapes_and_blocks(model, batch):
    with torch.inference_mode():
        tokens, _ = model.prepare_visual_tokens(batch["IMAGES"])
        bad = torch.randn(BATCH, 2, DIM)
        with pytest.raises(ValueError):
            model.forward_from_planner_query_tok(bad, batch["PAST"])
        with pytest.raises(ValueError):
            model.forward_from_block_query_tok(
                torch.randn(BATCH, DIM), batch["PAST"], tokens, start_block=model.cfg.n_blocks
            )
