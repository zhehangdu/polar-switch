# SPDX-License-Identifier: MIT
# Modified: pinned dataset revision, output directory, and command line.
"""Download the GPT-2-tokenized FineWeb shards used by Modded-NanoGPT.

Downloads the validation shard and the requested number of 100M-token training
shards from kjj0/fineweb10B-gpt2 into $DATA_PATH/data/fineweb10B. The shards
are used as published, without retokenization.

Modded-NanoGPT
Adapted from its MIT-licensed data/cached_fineweb10B.py. See THIRD_PARTY.md
for the modifications.
Adapted source: https://github.com/KellerJordan/modded-nanogpt/blob/bc3a0c2d640d0d73dedaef87eae26148d2e32afb/data/cached_fineweb10B.py
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = "889765ea1f903759787add96995d81171b632d0c"


def main():
    """Download the requested shards and print the data directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_chunks", type=int, nargs="?", default=103,
                        help="100M-token training shards; the documented training run uses 9")
    args = parser.parse_args()
    if not 1 <= args.num_chunks <= 103:
        parser.error("num_chunks must be between 1 and 103")
    root = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[1] / ".local" / "nanogpt"))
    local_dir = root / "data" / "fineweb10B"
    filenames = ["fineweb_val_000000.bin"] + [
        f"fineweb_train_{i:06d}.bin" for i in range(1, args.num_chunks + 1)]
    for filename in filenames:
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=filename,
                        repo_type="dataset", revision=REVISION, local_dir=local_dir)
    print(f"Ready: {local_dir.resolve()} ({args.num_chunks} training shards plus validation)")


if __name__ == "__main__":
    main()
