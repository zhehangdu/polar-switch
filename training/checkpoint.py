"""Save and restore training state. Keep the two most recent checkpoints.

Checkpoints include the model, optimizers, training step, random number state,
and position in the training data. Each file is written to a temporary path
before replacing its destination. Load checkpoints only from trusted sources.
"""

import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def atomic_json(value, path):
    """Write JSON to a temporary file, then replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(directory, step, model, optimizers, loader, config):
    """Save training state and keep the two most recent checkpoint files.

    Returns:
        The path to the saved checkpoint. latest.json points to this file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:08d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    state = {
        "format": 3, "step": step, "config": config, "model": model.state_dict(),
        "optimizers": [o.state_dict() for o in optimizers], "loader": loader.state_dict(),
        # The scheduler is a pure function of completed updates and this configuration.
        "scheduler": {"completed_updates": step},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None},
    }
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json({"checkpoint": path.name, "step": step}, directory / "latest.json")
    for old in sorted(directory.glob("step_*.pt"))[:-2]:
        old.unlink()
    return path


def restore_checkpoint(path, model, optimizers, loader, config):
    """Restore training state from a trusted checkpoint and return its step.

    Args:
        path: A checkpoint file, checkpoint directory, or training run directory.
        model, optimizers, loader: Objects whose saved state will be restored.
        config: Expected training settings, including backend source hashes.

    Raises:
        ValueError: The saved format, settings, or data files do not match.
    """
    path = Path(path)
    if path.is_dir():
        if (path / "checkpoints").is_dir():
            path /= "checkpoints"
        name = json.loads((path / "latest.json").read_text())["checkpoint"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint pointer")
        path /= name
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != 3 or saved["config"] != config:
        raise ValueError("Incompatible checkpoint format or run configuration")
    if len(saved["optimizers"]) != len(optimizers):
        raise ValueError("Optimizer count changed")
    if saved["scheduler"]["completed_updates"] != saved["step"]:
        raise ValueError("Scheduler cursor differs from checkpoint step")
    model.load_state_dict(saved["model"], strict=True)
    for opt, state in zip(optimizers, saved["optimizers"]):
        opt.load_state_dict(state)
    loader.load_state_dict(saved["loader"])
    rng = saved["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"])
    return saved["step"]
