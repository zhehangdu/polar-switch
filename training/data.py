"""Read token batches from binary files and save the position for resuming training.

Files are read in filename order without repeating tokens after the dataset
ends. Each batch contains token IDs and targets for the next token. The loader
supports the GPT-2 FineWeb format used by Modded-NanoGPT and the format
written by prepare_data.py.
"""

from pathlib import Path

import numpy as np
import torch


def shard_info(path):
    """Validate a token file header and return its token count and storage dtype."""
    with open(path, "rb") as stream:
        h = np.frombuffer(stream.read(1024), dtype="<i4")
    if len(h) != 256:
        raise ValueError(f"Incomplete shard header: {path}")
    if (h[0], h[1]) == (20240520, 1):
        dtype = np.dtype("<u2")
    elif (h[0], h[1]) == (20260317, 2) and h[3] in (2, 4):
        dtype = np.dtype(f"<u{h[3]}")
    else:
        raise ValueError(f"Unsupported shard format: {path}")
    count = int(h[2])
    if count <= 0 or Path(path).stat().st_size != 1024 + count * dtype.itemsize:
        raise ValueError(f"Invalid shard size: {path}")
    return count, dtype


class TokenLoader:
    """Read consecutive token batches from files in filename order.

    The saved state records the position and input files. Reading past the end
    raises EOFError instead of repeating the dataset.
    """

    def __init__(self, root, split, batch_size, context, vocab_size):
        """Open a dataset split with the requested batch size and sequence length.

        Args:
            root: Directory containing binary token files.
            split: Filename split, such as "train" or "val".
            batch_size: Sequences per batch.
            context: Tokens per sequence.
            vocab_size: Upper bound on valid token IDs.
        """
        self.files = sorted(Path(root).glob(f"*_{split}_*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No {split} shards in {root}")
        self.info = [shard_info(p) for p in self.files]
        self.offsets = np.cumsum([0] + [n for n, _ in self.info]).tolist()
        self.total = self.offsets[-1]
        self.batch_size, self.context, self.vocab_size = batch_size, context, vocab_size
        self.position = 0
        self.maps = {}

    def signature(self):
        """Return the name, size, token count, and dtype of each file."""
        return [(p.name, p.stat().st_size, n, dtype.str) for p, (n, dtype) in zip(self.files, self.info)]

    def state_dict(self):
        """Return the current token position and file descriptions."""
        return {"position": self.position, "shards": self.signature()}

    def load_state_dict(self, state):
        """Restore the token position after checking that the files match."""
        if state["shards"] != self.signature() or not 0 <= state["position"] <= self.total:
            raise ValueError("Dataset shards or cursor changed")
        self.position = state["position"]

    def next_batch(self, device):
        """Return input tensors and targets for the next token on device.

        Both tensors have shape (batch_size, context). The target at each position
        is the token immediately after its input.
        """
        count = self.batch_size * self.context
        if self.position + count + 1 > self.total:
            raise EOFError("Training data exhausted; cycling is disabled")
        pieces, cursor, remaining = [], self.position, count + 1
        while remaining:
            idx = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
            if idx not in self.maps:
                self.maps[idx] = np.memmap(self.files[idx], mode="r", dtype=self.info[idx][1], offset=1024)
            offset = cursor - self.offsets[idx]
            size = min(remaining, self.info[idx][0] - offset)
            pieces.append(self.maps[idx][offset:offset + size])
            cursor += size
            remaining -= size
        tokens = np.concatenate(pieces).astype(np.int64)
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError("Token IDs exceed the configured vocabulary")
        self.position += count
        tokens = torch.from_numpy(tokens).to(device)
        return tokens[:-1].reshape(self.batch_size, self.context), tokens[1:].reshape(self.batch_size, self.context)
