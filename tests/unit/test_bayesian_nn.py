# tests/unit/test_bayesian_nn.py

import pytest
import torch

from cardioscar.engines.bayesian_nn import BayesianNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_model() -> BayesianNN:
    """BayesianNN with default architecture: 4 hidden layers, 128 neurons."""
    return BayesianNN()


@pytest.fixture
def sample_coords() -> torch.Tensor:
    """Batch of 100 normalised 3D coordinates in [0, 1]."""
    return torch.rand(100, 3)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

def test_default_hyperparameters(default_model):
    assert default_model.hidden_size == 128
    assert default_model.n_hidden_layers == 4
    assert default_model.dropout_rate == 0.1


def test_default_parameter_count(default_model):
    # Input:  3 → 128  = 3*128 + 128 = 512
    # Hidden: 128 → 128 (x3) = 3 * (128*128 + 128) = 49,536
    # Output: 128 → 1  = 128 + 1 = 129
    # Total: 512 + 49,536 + 129 = 50,177
    assert default_model.count_parameters() == 50_177


def test_dropout_layer_count(default_model):
    dropout_layers = [
        m for m in default_model.network.modules()
        if isinstance(m, torch.nn.Dropout)
    ]
    # Dropout after every 2 hidden layers: 4 layers → 1 dropout (at i=2)
    assert len(dropout_layers) == 1


def test_output_shape(default_model, sample_coords):
    output = default_model(sample_coords)
    assert output.shape == (100, 1)


def test_output_range(default_model, sample_coords):
    output = default_model(sample_coords)
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Parameterisation tests
# ---------------------------------------------------------------------------

def test_wider_model_parameter_count():
    model = BayesianNN(hidden_size=256, n_hidden_layers=4)
    # Input:  3 → 256  = 3*256 + 256 = 1,024
    # Hidden: 256 → 256 (x3) = 3 * (256*256 + 256) = 197,376
    # Output: 256 → 1  = 256 + 1 = 257
    # Total: 1,024 + 197,376 + 257 = 198,657
    assert model.count_parameters() == 198_657


def test_wider_model_output_shape():
    model = BayesianNN(hidden_size=256, n_hidden_layers=4)
    coords = torch.rand(50, 3)
    assert model(coords).shape == (50, 1)


def test_deeper_model_dropout_count():
    # 6 hidden layers → dropout at i=2 and i=4 → 2 dropout layers
    model = BayesianNN(n_hidden_layers=6)
    dropout_layers = [
        m for m in model.network.modules()
        if isinstance(m, torch.nn.Dropout)
    ]
    assert len(dropout_layers) == 2


# ---------------------------------------------------------------------------
# MC Dropout tests
# ---------------------------------------------------------------------------

def test_enable_dropout_sets_train_mode(default_model):
    default_model.eval()
    default_model.enable_dropout()
    dropout_layers = [
        m for m in default_model.modules()
        if isinstance(m, torch.nn.Dropout)
    ]
    assert all(m.training for m in dropout_layers)


def test_mc_dropout_produces_variance(default_model, sample_coords):
    default_model.eval()
    default_model.enable_dropout()
    samples = torch.stack([default_model(sample_coords) for _ in range(20)])
    std = samples.std(dim=0)
    # With dropout active, predictions should vary across samples
    assert std.mean().item() > 0.0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_invalid_hidden_size():
    with pytest.raises(ValueError):
        BayesianNN(hidden_size=0)


def test_invalid_n_hidden_layers():
    with pytest.raises(ValueError):
        BayesianNN(n_hidden_layers=0)


def test_invalid_dropout_rate():
    with pytest.raises(ValueError):
        BayesianNN(dropout_rate=1.0)