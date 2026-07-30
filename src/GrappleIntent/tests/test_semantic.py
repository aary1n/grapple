"""Tests for the semantic path model (VL-Transformer + intent field head)."""

import math

import pytest
import torch

from ..models.semantic.model import IntentField, IntentFieldHead, SemanticModel
from ..models.token_types import MultimodalFrame, TokenType, get_null_token

GRID_H = GRID_W = 64


@pytest.fixture(scope="module")
def model():
    # Tiny untrained backbone: tests exercise shapes/semantics, not accuracy
    m = SemanticModel(
        backbone_name="vit_tiny_patch16_224",
        embed_dim=64,
        cross_attention_heads=4,
        cross_attention_layers=2,
        grid_h=GRID_H,
        grid_w=GRID_W,
        num_intents=8,
        pretrained=False,
    )
    m.eval()
    return m


@pytest.fixture
def batch_frame():
    torch.manual_seed(0)
    return MultimodalFrame(
        image_global=torch.randn(2, 3, 112, 112),
        image_foveal=torch.randn(2, 3, 224, 224),
        gaze_vector=torch.randn(2, 3),
        hand_velocity=torch.randn(2, 3),
    )


class TestSemanticModel:
    def test_forward_output_type(self, model, batch_frame):
        with torch.no_grad():
            out = model(batch_frame)
        assert isinstance(out, IntentField)

    def test_output_shapes(self, model, batch_frame):
        with torch.no_grad():
            out = model(batch_frame)
        assert out.log_prob.shape == (2, GRID_H, GRID_W)
        assert out.mu.shape == (2, 2)
        assert out.sigma.shape == (2, 2, 2)
        assert out.entropy.shape == (2,)
        assert out.gradient.shape == (2, 2)
        assert out.intent_logits.shape == (2, 8)

    def test_mu_in_unit_square(self, model, batch_frame):
        with torch.no_grad():
            out = model(batch_frame)
        assert (out.mu >= 0).all() and (out.mu <= 1).all()

    def test_sigma_positive_definite(self, model, batch_frame):
        """Cholesky parameterization must guarantee PD covariance."""
        with torch.no_grad():
            out = model(batch_frame)
        eigvals = torch.linalg.eigvalsh(out.sigma)
        assert (eigvals > 0).all()

    def test_field_peaks_at_mu(self, model, batch_frame):
        """The intent field is a Gaussian — its argmax must sit at μ."""
        with torch.no_grad():
            out = model(batch_frame)
        for b in range(out.log_prob.shape[0]):
            flat = out.log_prob[b].argmax()
            peak_y, peak_x = divmod(flat.item(), GRID_W)
            assert abs(peak_x / (GRID_W - 1) - out.mu[b, 0].item()) < 0.05
            assert abs(peak_y / (GRID_H - 1) - out.mu[b, 1].item()) < 0.05

    def test_entropy_matches_sigma(self, model, batch_frame):
        """Entropy must equal the closed form 0.5·logdet(2πeΣ)."""
        with torch.no_grad():
            out = model(batch_frame)
        expected = 0.5 * torch.logdet(2 * math.pi * math.e * out.sigma)
        torch.testing.assert_close(out.entropy, expected)

    def test_unbatched_input_with_missing_modalities(self, model):
        """Null tokens substitute for missing modalities (graceful degradation)."""
        with torch.no_grad():
            out = model(MultimodalFrame(image_global=torch.randn(3, 112, 112)))
        assert out.mu.shape == (1, 2)
        assert out.log_prob.shape == (1, GRID_H, GRID_W)

    def test_gradients_flow(self, model, batch_frame):
        model.train()
        try:
            out = model(batch_frame)
            loss = -out.log_prob.mean() + out.intent_logits.sum()
            loss.backward()
            assert model.no_context_token.grad is not None
            assert model.intent_field_head.mu_head[0].weight.grad is not None
            assert model.intent_classifier.weight.grad is not None
        finally:
            model.zero_grad(set_to_none=True)
            model.eval()

    def test_deterministic_in_eval(self, model, batch_frame):
        with torch.no_grad():
            a = model(batch_frame)
            b = model(batch_frame)
        torch.testing.assert_close(a.mu, b.mu)
        torch.testing.assert_close(a.log_prob, b.log_prob)


class TestIntentFieldHead:
    def test_standalone_head(self):
        head = IntentFieldHead(embed_dim=32, grid_h=16, grid_w=16)
        log_prob, mu, sigma = head(torch.randn(3, 32))
        assert log_prob.shape == (3, 16, 16)
        assert mu.shape == (3, 2)
        assert (torch.linalg.eigvalsh(sigma) > 0).all()


class TestSyntheticSemanticData:
    def test_shapes_and_determinism(self):
        from ..data.synthetic_semantic import (
            SemanticIntentDataset,
            SemanticSyntheticConfig,
        )

        ds = SemanticIntentDataset(SemanticSyntheticConfig(num_samples=4, seed=1))
        img_g, img_f, gaze, vel, target, label = ds[0]
        assert img_g.shape == (3, 112, 112)
        assert img_f.shape == (3, 224, 224)
        assert gaze.shape == (3,) and vel.shape == (3,)
        assert target.shape == (2,)
        assert 0 <= label.item() < 4

        # Same (seed, idx) → identical sample
        again = SemanticIntentDataset(SemanticSyntheticConfig(num_samples=4, seed=1))[0]
        torch.testing.assert_close(img_g, again[0])
        torch.testing.assert_close(target, again[4])

    def test_label_matches_blob_quadrant(self):
        from ..data.synthetic_semantic import (
            SemanticIntentDataset,
            SemanticSyntheticConfig,
        )

        ds = SemanticIntentDataset(SemanticSyntheticConfig(num_samples=16, seed=3))
        for i in range(len(ds)):
            img_g, _, gaze, _, _, label = ds[i]
            # Recover blob position from the brightest pixel of the labeled channel
            chan = label.item() % 3
            flat = img_g[chan].argmax()
            py, px = divmod(flat.item(), 112)
            x, y = px / 111, py / 111
            assert label.item() == int(x >= 0.5) + 2 * int(y >= 0.5)
            # Gaze must point toward the blob
            assert gaze[0].item() * (x - 0.5) >= -0.05
            assert gaze[1].item() * (y - 0.5) >= -0.05


class TestGaussianNLL:
    def test_matches_closed_form_identity_sigma(self):
        from ..training.train_semantic import gaussian_nll

        mu = torch.zeros(1, 2)
        sigma = torch.eye(2).unsqueeze(0)
        target = torch.tensor([[3.0, 4.0]])  # |d|² = 25
        expected = 0.5 * (25.0 + 0.0 + 2 * math.log(2 * math.pi))
        assert gaussian_nll(mu, sigma, target).item() == pytest.approx(expected)

    def test_minimized_at_target(self):
        from ..training.train_semantic import gaussian_nll

        sigma = (0.1 * torch.eye(2)).unsqueeze(0)
        target = torch.tensor([[0.5, 0.5]])
        at_target = gaussian_nll(target.clone(), sigma, target)
        off_target = gaussian_nll(target + 0.2, sigma, target)
        assert at_target < off_target


class TestNullTokens:
    def test_gaze_null_is_forward_facing(self):
        null = get_null_token(TokenType.GAZE_VECTOR)
        torch.testing.assert_close(null, torch.tensor([0.0, 0.0, -1.0]))

    def test_image_nulls_are_zeros(self):
        g = get_null_token(TokenType.IMAGE_GLOBAL)
        f = get_null_token(TokenType.IMAGE_FOVEAL)
        assert g.shape == (3, 112, 112) and g.abs().sum() == 0
        assert f.shape == (3, 224, 224) and f.abs().sum() == 0
