from pathlib import Path
import sys

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/camera-based-e2e"))

from models.monocular import DeepMonocularModel  # noqa: E402
from models.sae import SparseAutoencoder, TOPK_AUX_SAE_TYPE  # noqa: E402
from sae_utils import planner_token_key, resolve_token_tensor  # noqa: E402


class DummyFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dims = [8]
        self.patch_size = 1
        self.data_config = {"input_size": (3, 2, 2)}

    def forward(self, x: torch.Tensor):
        batch = x.size(0)
        base = torch.arange(batch * 8 * 2 * 2, dtype=x.dtype, device=x.device)
        return [base.view(batch, 8, 2, 2)]


def test_topk_keeps_exact_k_active_latents() -> None:
    model = SparseAutoencoder(
        input_dim=8,
        latent_dim=8,
        sae_type=TOPK_AUX_SAE_TYPE,
        k=3,
        k_aux=4,
    )
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(8))
    x_pre = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])
    z = model.encode_preprocessed(x_pre)
    assert int((z > 0).sum().item()) == 3
    assert torch.allclose(z[0, :3], torch.tensor([5.0, 4.0, 3.0]))


def test_auxiliary_reconstruction_uses_only_dead_latents() -> None:
    model = SparseAutoencoder(
        input_dim=4,
        latent_dim=4,
        sae_type=TOPK_AUX_SAE_TYPE,
        k=1,
        k_aux=4,
    )
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(4))
        model.decoder.weight.copy_(torch.eye(4))
    residual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    dead_mask = torch.tensor([False, True, False, True])
    aux_hat = model.compute_auxiliary_reconstruction(residual, dead_mask)
    expected = torch.tensor([[0.0, 2.0, 0.0, 4.0]])
    assert torch.allclose(aux_hat, expected)


def test_topk_preprocess_restore_roundtrip() -> None:
    model = SparseAutoencoder(
        input_dim=4,
        latent_dim=4,
        sae_type=TOPK_AUX_SAE_TYPE,
        k=2,
        k_aux=4,
    )
    b_pre = torch.tensor([0.5, -1.0, 2.0, 0.25])
    x = torch.tensor(
        [
            [1.5, 0.0, 4.0, -0.75],
            [-0.5, 2.0, 0.5, 1.25],
        ]
    )
    model.set_preprocessing_state(b_pre=b_pre, cmse=1.0)
    x_pre, stats = model.preprocess_input(x)
    restored = model.restore_input(x_pre, stats)
    assert torch.allclose(restored, x, atol=1e-6)
    assert torch.allclose(x_pre.mean(dim=-1), torch.zeros(x.size(0)), atol=1e-6)
    assert torch.allclose(torch.norm(x_pre, dim=-1), torch.ones(x.size(0)), atol=1e-6)


def test_decoder_projection_and_normalization() -> None:
    model = SparseAutoencoder(
        input_dim=4,
        latent_dim=4,
        sae_type=TOPK_AUX_SAE_TYPE,
        k=2,
        k_aux=4,
    )
    with torch.no_grad():
        model.decoder.weight.copy_(torch.randn_like(model.decoder.weight))
        model.normalize_decoder_columns()
        model.decoder.weight.grad = torch.randn_like(model.decoder.weight)
        model.project_decoder_gradients()
        projected = model.decoder.weight.grad
        dots = (projected * model.decoder.weight).sum(dim=0)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)
    column_norms = torch.norm(model.decoder.weight, dim=0)
    assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-5)


def test_resolve_token_tensor_supports_all_block_keys_and_legacy_alias() -> None:
    blob = {
        "meta": {"n_blocks": 4},
        "planner_query_tok": torch.full((2, 3), 9.0),
    }
    for block_idx in range(4):
        key = planner_token_key(block_idx)
        blob[key] = torch.full((2, 3), float(block_idx))
        tensor, resolved_key = resolve_token_tensor(blob, block_idx)
        assert resolved_key == key
        assert torch.allclose(tensor, blob[key])

    legacy_blob = {
        "meta": {"n_blocks": 4},
        "planner_query_tok": torch.full((2, 3), 7.0),
    }
    tensor, resolved_key = resolve_token_tensor(legacy_blob, 3)
    assert resolved_key == "planner_query_tok"
    assert torch.allclose(tensor, legacy_blob["planner_query_tok"])


def test_block_resume_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = DeepMonocularModel(
        feature_extractor=DummyFeatures(),
        out_dim=4,
        n_blocks=4,
        n_proposals=2,
    )
    model.eval()

    batch_size = 2
    past = torch.randn(batch_size, 16, 6)
    images = [torch.zeros(batch_size, 3, 2, 2), torch.randn(batch_size, 3, 2, 2)]
    intent = torch.tensor([1, 2])

    full_out = model({"PAST": past, "IMAGES": images, "INTENT": intent}, return_block_tokens=True)
    tokens, _ = model.prepare_visual_tokens(images)

    for block_idx in range(model.cfg.n_blocks):
        block_query = full_out[f"planner_query_tok_block_{block_idx}"]
        resumed = model.forward_from_block_query_tok(
            block_query,
            past,
            tokens,
            start_block=block_idx,
        )
        assert torch.allclose(resumed["trajectory"], full_out["trajectory"], atol=1e-5)
        assert torch.allclose(resumed["scores"], full_out["scores"], atol=1e-5)
