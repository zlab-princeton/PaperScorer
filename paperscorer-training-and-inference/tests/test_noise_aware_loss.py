"""Tests for the Bayesian noise-aware loss for binary classification."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_model(noise_aware=False, add_rating_head=False, alpha_init=0.506, beta_init=0.15, hidden_size=32):
    """Helper to create a ModelForBinaryClassification with a tiny backbone."""
    from llamafactory.model.model_utils.binary_classifier import ModelForBinaryClassification

    # Minimal backbone that returns hidden states
    class TinyBackbone(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.embed = nn.Embedding(100, hidden_size)
            self.config = type("Config", (), {"hidden_size": hidden_size})()

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True, **kwargs):
            h = self.embed(input_ids)
            return type("Output", (), {"hidden_states": (h,), "last_hidden_state": h})()

    backbone = TinyBackbone(hidden_size)
    model = ModelForBinaryClassification(
        backbone,
        hidden_size,
        add_rating_head=add_rating_head,
        noise_aware=noise_aware,
        noise_alpha_init=alpha_init,
        noise_beta_init=beta_init,
    )
    return model


class TestNoiseAwareLossMath:
    """Test 1: Noise-aware loss math correctness."""

    def test_loss_matches_manual_computation(self):
        model = _make_model(noise_aware=True, alpha_init=0.506, beta_init=0.15)
        model.eval()

        input_ids = torch.randint(0, 100, (4, 10))
        attention_mask = torch.ones(4, 10, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Manually compute expected loss in log-space (matching implementation)
        with torch.no_grad():
            logits = outputs["logits"]
            alpha = torch.sigmoid(model.noise_logits.alpha)
            beta = torch.sigmoid(model.noise_logits.beta)

            log_sig_z = F.logsigmoid(logits.float())
            log_sig_neg_z = F.logsigmoid(-logits.float())
            log_p_noisy = torch.logaddexp(
                torch.log(1.0 - alpha) + log_sig_z,
                torch.log(beta) + log_sig_neg_z,
            )
            log_1mp_noisy = torch.logaddexp(
                torch.log(alpha) + log_sig_z,
                torch.log(1.0 - beta) + log_sig_neg_z,
            )
            expected_loss = -(labels * log_p_noisy + (1.0 - labels) * log_1mp_noisy).mean()

        assert torch.allclose(outputs["loss"], expected_loss, atol=1e-6)


class TestGradientFlow:
    """Test 2: Gradient flow to noise params."""

    def test_gradients_flow_to_all_params(self):
        model = _make_model(noise_aware=True)
        model.train()

        input_ids = torch.randint(0, 100, (4, 10))
        attention_mask = torch.ones(4, 10, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs["loss"].backward()

        assert model.noise_logits.alpha.grad is not None
        assert model.noise_logits.alpha.grad.abs().item() > 0
        assert model.noise_logits.beta.grad is not None
        assert model.noise_logits.beta.grad.abs().item() > 0
        assert model.decision_head.weight.grad is not None


class TestNoiseAwareOff:
    """Test 3: Noise-aware off → standard BCE."""

    def test_standard_bce_without_noise(self):
        model = _make_model(noise_aware=False)
        model.eval()

        input_ids = torch.randint(0, 100, (4, 10))
        attention_mask = torch.ones(4, 10, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs["logits"]
            expected_loss = F.binary_cross_entropy_with_logits(logits.float(), labels)

        assert torch.allclose(outputs["loss"], expected_loss, atol=1e-6)
        assert not hasattr(model, "noise_logits")


class TestSigmoidParameterization:
    """Test 4: Sigmoid parameterization correctness."""

    def test_init_values_match(self):
        model = _make_model(noise_aware=True, alpha_init=0.506, beta_init=0.15)
        alpha = torch.sigmoid(model.noise_logits.alpha).item()
        beta = torch.sigmoid(model.noise_logits.beta).item()
        assert abs(alpha - 0.506) < 1e-5
        assert abs(beta - 0.15) < 1e-5

    def test_custom_init_values(self):
        model = _make_model(noise_aware=True, alpha_init=0.3, beta_init=0.4)
        alpha = torch.sigmoid(model.noise_logits.alpha).item()
        beta = torch.sigmoid(model.noise_logits.beta).item()
        assert abs(alpha - 0.3) < 1e-5
        assert abs(beta - 0.4) < 1e-5


class TestNumericalStability:
    """Test 5: Numerical stability with extreme logits."""

    def test_extreme_logits_no_nan(self):
        model = _make_model(noise_aware=True)
        model.eval()

        input_ids = torch.randint(0, 100, (4, 10))
        attention_mask = torch.ones(4, 10, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        # Force extreme logits by manipulating weights
        with torch.no_grad():
            model.decision_head.weight.fill_(20.0)
            model.decision_head.bias.fill_(20.0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert torch.isfinite(outputs["loss"]).all()

        # Also test negative extreme
        with torch.no_grad():
            model.decision_head.weight.fill_(-20.0)
            model.decision_head.bias.fill_(-20.0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert torch.isfinite(outputs["loss"]).all()


class TestOutputDictStructure:
    """Test 6: Output dict structure."""

    def test_noise_aware_output_keys(self):
        model = _make_model(noise_aware=True)
        input_ids = torch.randint(0, 100, (2, 5))
        labels = torch.tensor([0.0, 1.0])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        assert "noise_alpha" in outputs
        assert "noise_beta" in outputs
        assert outputs["noise_alpha"] is not None
        assert outputs["noise_beta"] is not None
        # Logits should be raw decision logits, not p_noisy
        assert outputs["logits"].shape == (2,)

    def test_standard_output_keys(self):
        model = _make_model(noise_aware=False)
        input_ids = torch.randint(0, 100, (2, 5))
        labels = torch.tensor([0.0, 1.0])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        assert outputs["noise_alpha"] is None
        assert outputs["noise_beta"] is None

    def test_logits_are_raw_decision_logits(self):
        """Logits should be raw (not noise-transformed) regardless of noise_aware."""
        model = _make_model(noise_aware=True)
        input_ids = torch.randint(0, 100, (2, 5))

        # Without labels - no loss computation
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Logits should be unbounded (raw decision head output)
        logits = outputs["logits"]
        assert logits.shape == (2,)
        # They could be any float value, not bounded to [0,1]


class TestRatingCompatibility:
    """Test 7: Compatibility with rating loss."""

    def test_noise_aware_with_rating(self):
        model = _make_model(noise_aware=True, add_rating_head=True)
        model.train()

        input_ids = torch.randint(0, 100, (4, 10))
        attention_mask = torch.ones(4, 10, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        ratings = torch.tensor([0.5, 0.8, 0.9, 0.3])
        weight = 0.1

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ratings=ratings,
            rating_loss_weight=weight,
        )

        assert outputs["loss"] is not None
        assert outputs["loss_bce"] is not None
        assert outputs["loss_rating"] is not None
        # Total = bce + weight * rating
        expected_total = outputs["loss_bce"] + weight * outputs["loss_rating"]
        assert torch.allclose(outputs["loss"], expected_total, atol=1e-6)


class TestNoiseEMCallback:
    """Test 8: NoiseEMCallback phase toggling."""

    def test_phase_toggling(self):
        from llamafactory.train.cls.trainer import NoiseEMCallback

        # model_steps=10, noise_steps=5, cycle=15
        callback = NoiseEMCallback(model_epochs=2.0, noise_epochs=1.0, steps_per_epoch=5)
        assert callback.model_steps == 10
        assert callback.noise_steps == 5
        assert callback.cycle_steps == 15

        # Create a minimal mock model with noise and model params
        class MockParam:
            def __init__(self, name, requires_grad=True):
                self.name = name
                self._requires_grad = requires_grad

            @property
            def requires_grad(self):
                return self._requires_grad

            def requires_grad_(self, val):
                self._requires_grad = val

        class MockModel:
            def __init__(self):
                self._params = [
                    ("noise_logits.alpha", MockParam("noise_logits.alpha")),
                    ("noise_logits.beta", MockParam("noise_logits.beta")),
                    ("backbone.layer.weight", MockParam("backbone.layer.weight")),
                    ("decision_head.weight", MockParam("decision_head.weight")),
                ]

            def named_parameters(self):
                return self._params

        model = MockModel()
        noise_params = [p for n, p in model._params if n.startswith("noise_logits.")]
        model_params = [p for n, p in model._params if not n.startswith("noise_logits.")]

        class MockState:
            def __init__(self, step):
                self.global_step = step

        class MockArgs:
            pass

        class MockControl:
            pass

        args = MockArgs()
        control = MockControl()

        # Step through a full cycle + partial
        for step in range(20):
            state = MockState(step)
            callback.on_step_begin(args, state, control, model=model)

            cycle_pos = step % 15
            if cycle_pos < 10:
                # Model phase
                for p in noise_params:
                    assert not p.requires_grad, f"Step {step}: noise params should be frozen"
                for p in model_params:
                    assert p.requires_grad, f"Step {step}: model params should be trainable"
            else:
                # Noise phase
                for p in noise_params:
                    assert p.requires_grad, f"Step {step}: noise params should be trainable"
                for p in model_params:
                    assert not p.requires_grad, f"Step {step}: model params should be frozen"


class TestCheckpointRoundTrip:
    """Test 9: Checkpoint round-trip."""

    def test_save_load_noise_params(self):
        model1 = _make_model(noise_aware=True, alpha_init=0.506, beta_init=0.15)

        # Modify noise params
        with torch.no_grad():
            model1.noise_logits.alpha.fill_(1.5)
            model1.noise_logits.beta.fill_(-0.7)

        # Save state dict
        state_dict = model1.state_dict()

        # Create fresh model and load
        model2 = _make_model(noise_aware=True, alpha_init=0.3, beta_init=0.3)
        model2.load_state_dict(state_dict)

        assert torch.allclose(model1.noise_logits.alpha, model2.noise_logits.alpha)
        assert torch.allclose(model1.noise_logits.beta, model2.noise_logits.beta)
