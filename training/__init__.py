"""Training scripts for Modded-NanoGPT, Qwen3, and Gemma 3."""

import json
from pathlib import Path


def load_config(profile):
    """Read the training configuration for a model profile."""
    return json.loads((Path(__file__).parent / "configs" / f"{profile}.json").read_text())
