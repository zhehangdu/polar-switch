"""Train Qwen3 or Gemma 3 on one GPU with a selected polar approximation.

The configuration sets the model, optimizer, and training length. Gradients
are accumulated over microbatches to reach the configured tokens per update.
Muon updates hidden matrices with the selected method, and AdamW updates the
other parameters. The script writes validation losses, training metrics, and
checkpoints. See training/README.md for preparation, training, and resume
commands. Modded-NanoGPT uses train_nanogpt.py.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

import polar_switch
from polar_switch import METHODS, make_polynomial_orthogonalizer

from . import load_config
from .checkpoint import atomic_json, restore_checkpoint, save_checkpoint
from .data import TokenLoader
from .llm import build_model, chunked_loss, configure_kernels
from .muon import Muon, parameter_groups

ROOT = Path(__file__).resolve().parent


def lr_multiplier(config, update):
    """Return the learning rate multiplier for an optimizer update indexed from zero.

    Linear warmup over the first 1% of updates is followed by cosine decay to 0.1.
    """
    total = config["steps"]
    warmup = max(1, int(0.01 * total))
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup - 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, loader, tokens, device, loss_fn):
    """Compute mean cross entropy on the fixed validation prefix, then resume train mode."""
    if tokens % (loader.batch_size * loader.context):
        raise ValueError("Validation token count must divide into complete microbatches")
    loader.position = 0
    model.eval()
    total = torch.zeros((), device=device)
    batches = tokens // (loader.batch_size * loader.context)
    for _ in range(batches):
        x, y = loader.next_batch(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total += loss_fn(x, y).float()
    loss = (total / batches).item()
    model.train()
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    return loss


def main():
    """Run training, validation, checkpoint saving, and optional checkpoint resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["qwen600m", "gemma1b"], required=True, help="Model configuration")
    parser.add_argument("--method", choices=METHODS, type=str.upper, default="PS2_55",
                        help="Polar approximation (default PS2_55)")
    parser.add_argument("--data-dir", type=Path, help="Prepared dataset directory of the profile")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for logs and checkpoints")
    parser.add_argument("--lr", type=float, help="Learning rate (default: from the configuration)")
    parser.add_argument("--seed", type=int, help="Random seed (default: from the configuration)")
    parser.add_argument("--micro-batch-size", type=int,
                        help="Sequences per microbatch; must divide 256 (default: from the configuration)")
    parser.add_argument("--resume", type=Path, help="Run directory or checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, help="Stop early without changing the full learning rate schedule")
    parser.add_argument("--val-tokens", type=int,
                        help="Validation tokens per evaluation (default: from the configuration)")
    parser.add_argument("--checkpoint-every", type=int, help="Updates between checkpoints")
    parser.add_argument("--log-every", type=int, help="Updates between training log records")
    parser.add_argument("--no-compile", action="store_true",
                        help="Do not compile the model and loss; the orthogonalizer stays compiled")
    args = parser.parse_args()
    config = load_config(args.profile)
    for key in ("lr", "micro_batch_size", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This trainer supports one GPU per run; use gradient accumulation")
    micro_tokens = config["micro_batch_size"] * config["context"]
    if micro_tokens <= 0 or config["global_tokens"] % micro_tokens:
        raise ValueError("Microbatch must divide the global token batch exactly")
    accumulation = config["global_tokens"] // micro_tokens
    stop = args.max_steps if args.max_steps is not None else config["steps"]
    if not 0 < stop <= config["steps"]:
        raise ValueError("max-steps must lie within the full training schedule")
    val_tokens = args.val_tokens if args.val_tokens is not None else config["val_tokens"]
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else config["checkpoint_every"]
    log_every = args.log_every if args.log_every is not None else config["log_every"]
    if val_tokens <= 0 or checkpoint_every <= 0 or log_every <= 0:
        raise ValueError("Validation, checkpoint, and log intervals must be positive")
    data_dir = args.data_dir or ROOT.parent / ".local" / "datasets" / args.profile
    loader = TokenLoader(data_dir, "train", config["micro_batch_size"], config["context"],
                         config["model"]["vocab_size"])
    val_loader = TokenLoader(data_dir, "val", config["micro_batch_size"], config["context"],
                             config["model"]["vocab_size"])
    if loader.total < stop * config["global_tokens"] + 1 or val_loader.total < val_tokens + 1:
        raise ValueError("Insufficient non-repeating tokens for the requested run")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and args.resume is None:
        raise ValueError("Output already exists; use --resume or a new output directory")
    backend_dir = Path(polar_switch.__file__).parent
    backend_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(backend_dir.glob("*.py"))}
    run_config = {**config, "method": args.method,
                  "backend_hashes": backend_hashes, "train_shards": loader.signature(),
                  "attention_backend": "polar_cudnn", "mlp_backend": "quack",
                  "loss_backend": "quack-chunked", "validation_loss_backend": "torch-chunked"}
    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA-capable GPU")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = build_model(config).to(device)
    configure_kernels(model, args.profile)
    matrix_groups, auxiliary = parameter_groups(model)
    orthogonalize = make_polynomial_orthogonalizer(args.method)
    names = {id(p): n for n, p in model.named_parameters()}
    norm_log = None
    if os.environ.get("MUON_LOG_SPECTRAL_NORMS", "0") == "1":
        from .diagnostics import NormLog
        norm_log = NormLog(output / "norms", 0, args.method)
    matrix_opt = Muon(matrix_groups, orthogonalize, config["lr"], config["momentum"],
                      config["weight_decay"], config["lr_scale"], norm_log, names)
    adam_opt = torch.optim.AdamW(auxiliary, lr=config["lr"], betas=tuple(config["adam_betas"]),
                          eps=config["adam_eps"], weight_decay=config["weight_decay"])
    optimizers = [matrix_opt, adam_opt]
    step = restore_checkpoint(args.resume, model, optimizers, loader, run_config) if args.resume else 0
    if step > stop:
        raise ValueError("Checkpoint is beyond the requested stop")
    compile_mode = "max-autotune-no-cudagraphs"
    packages = ["torch", "transformers", "quack-kernels", "triton", "nvidia-cudnn-frontend"]
    previous_fallbacks = (json.loads((output / "manifest.json").read_text()).get("fallback_events", [])
                          if args.resume and (output / "manifest.json").exists() else [])
    manifest = {
        "run_config": run_config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "matrix_groups": [{"role": g["role"], "parameters": [
            {"name": names[id(p)], "shape": list(p.shape)} for p in g["params"]]} for g in matrix_groups],
        "adamw_parameters": [names[id(p)] for p in auxiliary],
        "fallback_events": previous_fallbacks,
        "norm_logging": norm_log is not None,
        "software": {n: importlib.metadata.version(n) for n in packages},
        "gpu": torch.cuda.get_device_name(),
        "data_dir": str(data_dir.resolve()), "resume": str(args.resume) if args.resume else None,
        "stop_update": stop, "full_updates": config["steps"], "accumulation": accumulation,
        "val_tokens": val_tokens, "compiled": not args.no_compile,
        "compile_mode": compile_mode if not args.no_compile else None,
        "checkpoint_every": checkpoint_every, "log_every": log_every,
        "training_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
    }
    atomic_json(manifest, output / "manifest.json")
    atomic_json({"status": "running", "step": step}, output / "status.json")
    print(json.dumps({"profile": args.profile, "method": args.method, "parameters": manifest["parameter_count"],
                      "accumulation": accumulation, "output": str(output)}), flush=True)
    # Include cross-entropy in the graph so large vocabulary intermediates can be fused.
    loss_fn = lambda x, y: chunked_loss(model, x, y)
    if not args.no_compile:
        loss_fn = torch.compile(loss_fn, mode=compile_mode)
    started = time.monotonic()
    train_seconds = 0.0
    model.train()
    try:
        with (output / "metrics.jsonl").open("a", buffering=1) as metrics:
            def emit(record):
                """Append a metrics record and save new fallback events in the manifest."""
                events = previous_fallbacks + orthogonalize.fallback_events
                if manifest["fallback_events"] != events:
                    manifest["fallback_events"] = events
                    atomic_json(manifest, output / "manifest.json")
                record.update(step=step, tokens=step * config["global_tokens"],
                              fallback_used=bool(events),
                              elapsed_seconds=time.monotonic() - started)
                line = json.dumps(record, allow_nan=False)
                metrics.write(line + "\n")
                print(line, flush=True)
            emit({"event": "validation", "loss": validate(model, val_loader, val_tokens, device, loss_fn)})
            while step < stop:
                torch.cuda.synchronize()
                tick = time.monotonic()
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                    for group in opt.param_groups:
                        group["lr"] = config["lr"] * lr_multiplier(config, step)
                mean_loss = torch.zeros((), device=device)
                for _ in range(accumulation):
                    x, y = loader.next_batch(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = loss_fn(x, y)
                    mean_loss += loss.detach().float() / accumulation
                    (loss / accumulation).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                if norm_log is not None:
                    norm_log.step = step
                for opt in optimizers:
                    opt.step()
                step += 1
                torch.cuda.synchronize()
                seconds = time.monotonic() - tick
                train_seconds += seconds
                if step % log_every == 0 or step == stop:
                    loss_value, norm_value = mean_loss.item(), norm.item()
                    if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                        raise FloatingPointError("Non-finite training loss or gradient norm")
                    emit({"event": "training", "loss": loss_value, "grad_norm": norm_value,
                          "lr": matrix_opt.param_groups[0]["lr"], "step_seconds": seconds,
                          "tokens_per_second": config["global_tokens"] / seconds,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
                if step % config["val_every"] == 0 or step == stop:
                    emit({"event": "validation", "loss": validate(model, val_loader, val_tokens, device, loss_fn)})
                if step % checkpoint_every == 0 or step == stop:
                    save_checkpoint(output / "checkpoints", step, model, optimizers, loader, run_config)
                    atomic_json({"status": "running", "step": step}, output / "status.json")
        atomic_json({"status": "completed" if step == config["steps"] else "stopped",
                     "fallback_used": bool(manifest["fallback_events"]),
                     "step": step, "tokens": step * config["global_tokens"],
                     "training_seconds": train_seconds, "elapsed_seconds": time.monotonic() - started},
                    output / "status.json")
    except Exception as error:
        manifest["fallback_events"] = previous_fallbacks + orthogonalize.fallback_events
        atomic_json(manifest, output / "manifest.json")
        atomic_json({"status": "failed", "step": step, "error": str(error),
                     "fallback_used": bool(manifest["fallback_events"]),
                     "numerical_failure": isinstance(error, FloatingPointError)}, output / "status.json")
        raise
    finally:
        if norm_log is not None:
            norm_log.close()


if __name__ == "__main__":
    main()
