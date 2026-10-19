"""Prepare training and validation token files for Qwen3 and Gemma 3.

Streams FineWeb-Edu sample-100BT at a pinned revision and tokenizes each
document with the pinned Qwen3 or Gemma 3 tokenizer, preceded by the
end-of-sequence token. A document is used for validation when the first 8 bytes
of the SHA-256 hash of its ID, read as an integer, are divisible by 100. Each
split stops at the token count required by the training configuration, so
training never repeats data. Preparation saves progress so an interrupted run
can resume. Each dataset includes a manifest identifying its source, tokenizer
revision, and token counts.
Modded-NanoGPT data is downloaded by cached_fineweb10B.py instead.
See training/README.md for commands.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import load_config
from .checkpoint import atomic_json

TOKENIZERS = {}
FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def init_tokenizers(specs, cache, local_files_only=False):
    """Load the configured tokenizer revisions for a preparation worker."""
    from transformers import AutoTokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for profile, spec in specs.items():
        TOKENIZERS[profile] = AutoTokenizer.from_pretrained(
            spec["name"], revision=spec["revision"], cache_dir=cache,
            local_files_only=local_files_only)


def encode_document(item):
    """Assign a document to training or validation and tokenize it for each model."""
    identifier, text = item
    split = "val" if int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big") % 100 == 0 else "train"
    tokens = {}
    for name, tokenizer in TOKENIZERS.items():
        values = [tokenizer.eos_token_id] + tokenizer.encode(text, add_special_tokens=False)
        tokens[name] = np.asarray(values, dtype="<u4")
    return split, tokens


class ShardWriter:
    """Write token files and resume from their last saved lengths.

    On resume, discard unfinished writes beyond the saved token counts and
    remove token files that were not saved.
    """

    def __init__(self, root, targets, eos, state=None, shard_size=100_000_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.targets, self.eos, self.shard_size = targets, eos, shard_size
        self.counts = state or {"train": [], "val": []}
        self.handles = {}
        known = {}
        for split, counts in self.counts.items():
            for index, count in enumerate(counts):
                path = self.path(split, index)
                known[path.name] = count
                with path.open("r+b") as stream:
                    stream.truncate(1024 + 4 * count)
                    stream.seek(0)
                    stream.write(self.header(count))
        for path in self.root.glob("dataset_*_*.bin"):
            if path.name not in known:
                # Remove token files from an unfinished write that the saved state does not list.
                path.unlink()

    def path(self, split, index):
        """Return the path of a token file."""
        return self.root / f"dataset_{split}_{index:06d}.bin"

    def header(self, count):
        """Return the 1024-byte header of a token file with count tokens."""
        h = np.zeros(256, dtype="<i4")
        h[:7] = (20260317, 2, count, 4, self.eos, 1, 0)
        return h.tobytes()

    def append(self, split, tokens):
        """Append tokens up to the requested split size, opening new files as needed."""
        counts = self.counts[split]
        remaining = self.targets[split] - sum(counts)
        tokens = tokens[:remaining]
        while len(tokens):
            if not counts or counts[-1] == self.shard_size:
                counts.append(0)
            index = len(counts) - 1
            key = (split, index)
            if key not in self.handles:
                path = self.path(split, index)
                stream = path.open("r+b" if path.exists() else "w+b")
                if counts[-1] == 0:
                    stream.write(self.header(0))
                stream.seek(1024 + 4 * counts[-1])
                self.handles[key] = stream
            size = min(len(tokens), self.shard_size - counts[-1])
            self.handles[key].write(tokens[:size].tobytes())
            counts[-1] += size
            tokens = tokens[size:]

    def flush(self):
        """Update file headers, finish pending writes, and close open files."""
        for (split, index), stream in self.handles.items():
            stream.seek(0)
            stream.write(self.header(self.counts[split][index]))
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self.handles.clear()

    def complete(self):
        """Return whether both splits contain their requested token counts."""
        return all(sum(self.counts[s]) == self.targets[s] for s in self.targets)


def main():
    """Prepare the selected datasets and save their manifests and resume state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["qwen600m", "gemma1b"], required=True, help="Model configuration")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / ".local" / "datasets",
                        help="Directory for the prepared datasets (default .local/datasets)")
    parser.add_argument("--workers", type=int, default=8, help="Tokenizer worker processes (default 8)")
    parser.add_argument("--revision", help="FineWeb-Edu dataset revision (default: the pinned revision)")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    state_path = root / f"fineweb_edu_{args.profile}_prepare_state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    if previous and previous.get("complete"):
        print("The requested dataset is already complete.", flush=True)
        return
    selected = (args.profile,)
    profiles = {p: load_config(p) for p in selected}
    if previous:
        revision, specs = previous["revision"], previous["tokenizers"]
        if args.revision and args.revision != revision:
            raise ValueError("Cannot change source revision during resume")
    else:
        for profile in profiles:
            dest = root / profile
            if dest.exists() and any(dest.iterdir()):
                raise ValueError(f"Refusing to overwrite existing dataset directory: {dest}")
        revision = args.revision or FINEWEB_EDU_REVISION
        specs = {p: {"name": c["tokenizer"], "revision": c["tokenizer_revision"]}
                 for p, c in profiles.items()}
    cache = str(root / ".cache" / "tokenizers")
    print(f"Loading tokenizers for {args.profile}; resume={previous is not None}", flush=True)
    init_tokenizers(specs, cache)
    targets = {p: {"train": c["steps"] * c["global_tokens"] + 1, "val": c["val_tokens"] + 1}
               for p, c in profiles.items()}
    if previous and previous["targets"] != targets:
        raise ValueError("Token budgets changed during preparation")
    writers = {p: ShardWriter(root / p, targets[p], TOKENIZERS[p].eos_token_id,
                             previous["writers"][p] if previous else None) for p in profiles}
    print("Opening pinned FineWeb-Edu stream", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train",
                      revision=revision, streaming=True, cache_dir=str(root / ".cache" / "datasets"))
    if previous and previous.get("source_state"):
        ds.load_state_dict(previous["source_state"])
    document_count = previous["documents"] if previous else 0
    iterator = iter(ds)
    state = {"revision": revision, "tokenizers": specs, "targets": targets,
             "documents": document_count, "source_state": ds.state_dict(),
             "writers": {p: w.counts for p, w in writers.items()}, "complete": False}
    atomic_json(state, state_path)
    print(f"Resuming after {document_count} committed documents", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_tokenizers, initargs=(specs, cache, True)) as pool:
        while not all(w.complete() for w in writers.values()):
            batch = []
            for _ in range(256):
                try:
                    row = next(iterator)
                except StopIteration:
                    break
                batch.append((row.get("id") or hashlib.sha256(row["text"].encode()).hexdigest(), row["text"]))
            if not batch:
                raise EOFError("FineWeb-Edu ended before the token budget was reached; data is never repeated")
            for split, encoded in pool.map(encode_document, batch, chunksize=8):
                for profile, values in encoded.items():
                    writers[profile].append(split, values)
            for writer in writers.values():
                writer.flush()
            document_count += len(batch)
            state.update(documents=document_count, source_state=ds.state_dict(),
                         writers={p: w.counts for p, w in writers.items()})
            atomic_json(state, state_path)
            if document_count % 25600 == 0:
                print(json.dumps({"documents": document_count, "tokens": {
                    p: {s: sum(ns) for s, ns in w.counts.items()} for p, w in writers.items()}}), flush=True)
    for p, writer in writers.items():
        atomic_json({"complete": True, "source": "HuggingFaceFW/fineweb-edu", "subset": "sample-100BT",
                     "revision": revision, "tokenizer": specs[p], "targets": targets[p],
                     "split": "sha256(document_id) first 8 bytes, big endian, modulo 100 == 0 is validation",
                     "special_tokens": "prepend EOS; add_special_tokens=False",
                     "shards": {s: [{"file": writer.path(s, i).name, "tokens": n} for i, n in enumerate(ns)]
                                for s, ns in writer.counts.items()}}, root / p / "manifest.json")
    state["complete"] = True
    atomic_json(state, state_path)
    print("Data preparation complete.", flush=True)


if __name__ == "__main__":
    main()
