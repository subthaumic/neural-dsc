import json

import pytest
import torch

from neural_dsc.data.gen import sample_quadrant_data
from neural_dsc.utils.probe_selection import softmax_probabilities
from neural_dsc.utils.trained_quadrants import (
    QuadrantTrainingConfig,
    checkpoint_metadata_path,
    load_trained_quadrant_model,
    train_or_load_quadrant_model,
)


def test_softmax_probe_space_is_invariant_to_common_logit_shift():
    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0], [-3.0, 1.0, 2.0, 4.0]])
    shifts = torch.tensor([[100.0], [-37.0]])

    assert torch.allclose(
        softmax_probabilities(logits),
        softmax_probabilities(logits + shifts),
    )


def test_online_quadrant_stream_is_fresh_and_reproducible():
    first_stream = torch.Generator().manual_seed(123)
    first_batch = sample_quadrant_data(16, generator=first_stream)
    second_batch = sample_quadrant_data(16, generator=first_stream)

    replay_stream = torch.Generator().manual_seed(123)
    replay_batch = sample_quadrant_data(16, generator=replay_stream)

    assert torch.equal(first_batch[0], replay_batch[0])
    assert torch.equal(first_batch[1], replay_batch[1])
    assert not torch.equal(first_batch[0], second_batch[0])


def tiny_config():
    return QuadrantTrainingConfig(
        architecture="test",
        layer_sizes=(2, 4),
        train_draws_per_epoch=64,
        train_data_seed=10,
        eval_size=32,
        eval_data_seed=11,
        model_seed=12,
        epochs=1,
        batch_size=16,
        learning_rate=0.01,
    )


def test_trained_checkpoint_records_provenance_and_is_reused(tmp_path):
    checkpoint = tmp_path / "quadrants_test_trained.pth"
    config = tiny_config()

    first = train_or_load_quadrant_model(
        checkpoint,
        config,
        print_every=1,
        min_eval_accuracy=0.0,
    )
    second = train_or_load_quadrant_model(
        checkpoint,
        config,
        print_every=1,
        min_eval_accuracy=0.0,
    )

    assert first.checkpoint_reused is False
    assert second.checkpoint_reused is True
    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert config.total_train_draws == 64

    metadata = json.loads(checkpoint_metadata_path(checkpoint).read_text())
    assert metadata["config"] == config.to_dict()
    assert metadata["config"]["train_sampling"] == "fresh_uniform_draws_per_minibatch"
    assert metadata["checkpoint_sha256"] == first.checkpoint_sha256

    loaded = load_trained_quadrant_model(checkpoint, config.layer_sizes)
    assert all(
        torch.equal(first.model.state_dict()[key], loaded.state_dict()[key])
        for key in first.model.state_dict()
    )


def test_trained_checkpoint_rejects_metadata_checksum_mismatch(tmp_path):
    checkpoint = tmp_path / "quadrants_test_trained.pth"
    config = tiny_config()
    train_or_load_quadrant_model(
        checkpoint,
        config,
        print_every=1,
        min_eval_accuracy=0.0,
    )
    metadata_path = checkpoint_metadata_path(checkpoint)
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_sha256"] = "not-the-checkpoint-hash"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="disagree"):
        load_trained_quadrant_model(checkpoint, config.layer_sizes)
