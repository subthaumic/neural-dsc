"""Reproducible training and checkpointing for trained quadrant classifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neural_dsc.data.gen import generate_quadrant_data, sample_quadrant_data
from neural_dsc.mlp import MLP


CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class QuadrantTrainingConfig:
    """Everything that determines the trained weights and their evaluation."""

    architecture: str
    layer_sizes: tuple[int, ...]
    train_draws_per_epoch: int
    train_data_seed: int
    eval_size: int
    eval_data_seed: int
    model_seed: int
    epochs: int
    batch_size: int
    learning_rate: float

    @property
    def total_train_draws(self) -> int:
        return self.train_draws_per_epoch * self.epochs

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer_sizes"] = list(self.layer_sizes)
        payload["train_sampling"] = "fresh_uniform_draws_per_minibatch"
        return payload


@dataclass
class TrainedQuadrantRun:
    """A model together with the independent data and provenance used for it."""

    model: MLP
    eval_data: tuple[torch.Tensor, torch.Tensor]
    final_epoch_train_loss: float
    eval_accuracy: float
    eval_loss: float
    checkpoint_sha256: str
    checkpoint_reused: bool


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    """Return the JSON provenance path next to a state-dict checkpoint."""
    return checkpoint_path.with_suffix(".json")


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_is_current(
    checkpoint_path: Path,
    config: QuadrantTrainingConfig,
) -> tuple[bool, str]:
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if not checkpoint_path.exists():
        return False, "checkpoint is missing"
    metadata = _load_metadata(metadata_path)
    if metadata is None:
        return False, "checkpoint metadata is missing or unreadable"
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False, "checkpoint metadata schema changed"
    if metadata.get("config") != config.to_dict():
        return False, "training or evaluation configuration changed"
    expected_hash = metadata.get("checkpoint_sha256")
    actual_hash = file_sha256(checkpoint_path)
    if expected_hash != actual_hash:
        return False, "checkpoint checksum does not match its metadata"
    return True, "checkpoint and metadata match"


def _save_checkpoint(
    model: MLP,
    checkpoint_path: Path,
    config: QuadrantTrainingConfig,
    metrics: dict[str, float],
) -> str:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    checkpoint_tmp = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.tmp")

    torch.save(model.state_dict(), checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": config.to_dict(),
        "metrics_at_creation": metrics,
        "checkpoint_sha256": checkpoint_hash,
    }
    with metadata_tmp.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    metadata_tmp.replace(metadata_path)
    return checkpoint_hash


def load_trained_quadrant_model(
    checkpoint_path: Path,
    layer_sizes: tuple[int, ...] | list[int],
) -> MLP:
    """Load a trained model only when its checkpoint provenance is intact."""
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    metadata = _load_metadata(metadata_path)
    if metadata is None:
        raise FileNotFoundError(
            f"Missing trained-checkpoint metadata: {metadata_path}. Run the "
            "corresponding compute_quadrants_*_trained.py producer first."
        )
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint metadata at {metadata_path} uses an obsolete training "
            "protocol. Run the corresponding trained producer to retrain it."
        )
    expected_hash = metadata.get("checkpoint_sha256")
    if not checkpoint_path.exists() or file_sha256(checkpoint_path) != expected_hash:
        raise RuntimeError(
            f"Trained checkpoint and metadata disagree: {checkpoint_path}. Run "
            "the corresponding trained producer to regenerate them together."
        )
    expected_layers = list(layer_sizes)
    if metadata.get("config", {}).get("layer_sizes") != expected_layers:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} was not trained for layers "
            f"{expected_layers}."
        )
    model = MLP(expected_layers)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_or_load_quadrant_model(
    checkpoint_path: Path,
    config: QuadrantTrainingConfig,
    *,
    print_every: int,
    min_eval_accuracy: float,
) -> TrainedQuadrantRun:
    """Load a matching checkpoint or train on fresh uniform draws each batch."""
    eval_data = generate_quadrant_data(
        n_total=config.eval_size,
        seed=config.eval_data_seed,
    )

    current, reason = _checkpoint_is_current(checkpoint_path, config)
    checkpoint_reused = current
    if current:
        print(f"Loading current trained checkpoint from {checkpoint_path}")
        model = load_trained_quadrant_model(checkpoint_path, config.layer_sizes)
        metadata = _load_metadata(checkpoint_metadata_path(checkpoint_path))
        assert metadata is not None
        final_epoch_train_loss = float(
            metadata["metrics_at_creation"]["final_epoch_train_loss"]
        )
    else:
        if checkpoint_path.exists():
            print(f"Retraining {config.architecture} model: {reason}")
        else:
            print(f"Training new {config.architecture} quadrant classifier")
        torch.manual_seed(config.model_seed)
        model = MLP(list(config.layer_sizes))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        training_stream = torch.Generator().manual_seed(config.train_data_seed)
        print(
            "Training on fresh uniform draws "
            f"({config.train_draws_per_epoch:,} per epoch, "
            f"{config.total_train_draws:,} total)..."
        )
        final_epoch_train_loss = float("nan")
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            n_seen = 0
            while n_seen < config.train_draws_per_epoch:
                batch_count = min(
                    config.batch_size,
                    config.train_draws_per_epoch - n_seen,
                )
                inputs, labels = sample_quadrant_data(
                    n_total=batch_count,
                    generator=training_stream,
                )
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_count
                n_seen += batch_count
            final_epoch_train_loss = running_loss / n_seen
            if (epoch + 1) % print_every == 0:
                print(
                    f"  Epoch {epoch + 1}/{config.epochs} - "
                    f"Loss: {final_epoch_train_loss:.4f}"
                )

    print(
        f"Evaluating on {config.eval_size:,} independent points "
        f"(seed {config.eval_data_seed})..."
    )
    eval_accuracy, eval_loss = model.evaluate(
        DataLoader(TensorDataset(*eval_data), batch_size=4096, shuffle=False)
    )
    if eval_accuracy < min_eval_accuracy:
        raise RuntimeError(
            f"Independent accuracy {eval_accuracy:.2%} is below the required "
            f"{min_eval_accuracy:.2%}; refusing to construct DSCs from this model."
        )

    metrics = {
        "final_epoch_train_loss": final_epoch_train_loss,
        "eval_accuracy": eval_accuracy,
        "eval_loss": eval_loss,
    }
    if current:
        checkpoint_hash = file_sha256(checkpoint_path)
    else:
        checkpoint_hash = _save_checkpoint(model, checkpoint_path, config, metrics)
        print(f"Saved checkpoint and provenance to {checkpoint_path}")

    model.eval()
    return TrainedQuadrantRun(
        model=model,
        eval_data=eval_data,
        final_epoch_train_loss=final_epoch_train_loss,
        eval_accuracy=eval_accuracy,
        eval_loss=eval_loss,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_reused=checkpoint_reused,
    )


def write_training_summary(
    path: Path,
    config: QuadrantTrainingConfig,
    run: TrainedQuadrantRun,
) -> None:
    """Write the complete train/evaluation provenance beside experiment output."""
    lines = [
        f"Trained {config.architecture} quadrant classifier",
        f"layer_sizes: {list(config.layer_sizes)}",
        "train_sampling: fresh_uniform_draws_per_minibatch",
        f"train_draws_per_epoch: {config.train_draws_per_epoch}",
        f"total_train_draws: {config.total_train_draws}",
        f"train_data_seed: {config.train_data_seed}",
        f"eval_size: {config.eval_size}",
        f"eval_data_seed: {config.eval_data_seed}",
        f"model_seed: {config.model_seed}",
        f"train_epochs: {config.epochs}",
        f"train_batch_size: {config.batch_size}",
        f"train_learning_rate: {config.learning_rate}",
        f"final_epoch_train_loss: {run.final_epoch_train_loss:.6f}",
        f"eval_accuracy: {run.eval_accuracy:.6f}",
        f"eval_loss: {run.eval_loss:.6f}",
        f"checkpoint_sha256: {run.checkpoint_sha256}",
        f"checkpoint_reused: {run.checkpoint_reused}",
    ]
    path.write_text("\n".join(lines) + "\n")
