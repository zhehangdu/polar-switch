# Training

Run these commands from the repository root. All models start from random initialization, and each run uses one GPU.

> [!IMPORTANT]
> **We tested training only on one NVIDIA B200 GPU**, with PyTorch 2.14.0, CUDA 13.0, FlashAttention-4, cuDNN attention, and QuACK kernels. Other GPUs need changes to the code and the dependencies.

## Install

Install PyTorch 2.14.0 with CUDA 13.0 and the package with its custom kernels, as the [installation instructions](../README.md#custom-gpu-kernels) show. Then install the training dependencies for Qwen3 and Gemma 3:

```bash
python -m pip install -r training/requirements.txt
```

### Modded-NanoGPT attention

Modded-NanoGPT uses FlashAttention-4. You build it from the source revision that [requirements-nanogpt.txt](requirements-nanogpt.txt) pins. The build needs Linux, Git, a C++ compiler, and the CUDA 13.0 toolkit in `CUDA_HOME`. For more information, see the upstream [FlashAttention-4](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/README.md) requirements.

```bash
python -m pip install "setuptools>=77" wheel packaging ninja setuptools-scm
python -m pip install --no-build-isolation -r training/requirements-nanogpt.txt
```

## Prepare data

For Modded-NanoGPT, download nine training shards and the validation shard of the GPT-2-tokenized FineWeb from `kjj0/fineweb10B-gpt2` (about 2 GB):

```bash
DATA_PATH=.local/nanogpt python training/cached_fineweb10B.py 9
```

For Qwen3 and Gemma 3, tokenize FineWeb-Edu `sample-100BT`. The token shards of both models use about 136 GB:

```bash
hf auth login
python -m training.prepare_data --profile qwen600m --workers 8 --output-root .local/datasets
python -m training.prepare_data --profile gemma1b --workers 8 --output-root .local/datasets
```

Gemma needs approved Hugging Face access to [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt). For each output directory, run only one preparation command at a time. If the command stops, run it again to continue. The revisions of the dataset, the tokenizer, and the downloader are pinned. The manifest of each dataset records its tokenizer revision and token counts. For training, pass the profile directory, such as `.local/datasets/qwen600m`, to `--data-dir`.

## Training setups

<div align="center">

| Profile | Parameters | Tokens | Configuration | Updates | Tokens per update | Attention |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Modded-NanoGPT #91 | 638M | 343M | [train_nanogpt.py](train_nanogpt.py) | 1,290 | 131,072 to 393,216 in 8 microbatches | FlashAttention-4 |
| Qwen3-0.6B | 596M | 12B | [qwen600m.json](configs/qwen600m.json) | 22,889 | 8 × 32 sequences of length 2,048 | cuDNN |
| Gemma 3-1B | 1B | 22B | [gemma1b.json](configs/gemma1b.json) | 41,962 | 8 × 32 sequences of length 2,048 | cuDNN |

</div>

`--micro-batch-size` counts sequences and must be a divisor of 256.

## Train and resume

```bash
DATA_PATH=.local/nanogpt POLAR_METHOD=PS2_55 torchrun --standalone --nproc_per_node=1 training/train_nanogpt.py --seed 0
python -m training.train --profile qwen600m --method PS2_55 --seed 42 --data-dir .local/datasets/qwen600m --output-dir .local/runs/qwen-PS2_55
python -m training.train --profile gemma1b --method PS2_55 --seed 42 --data-dir .local/datasets/gemma1b --output-dir .local/runs/gemma-PS2_55
```

`--method` and `POLAR_METHOD` accept any of the [12 methods](../polar_switch/README.md#methods-and-circuits). The paper uses seeds 0–19 for Modded-NanoGPT and seed 42 for Qwen3 and Gemma 3. Each run uses a separate output directory.

To continue a Qwen3 or Gemma 3 run, repeat the same command and pass its output directory to `--resume`:

```bash
python -m training.train --profile qwen600m --method PS2_55 --seed 42 --data-dir .local/datasets/qwen600m --output-dir .local/runs/qwen-PS2_55 --resume .local/runs/qwen-PS2_55
```

A resumed run checks the `polar_switch` code (even its comments) and the training shards, so do not change them. Modded-NanoGPT has no resume option.

For a short check, add `--max-steps 8 --val-tokens 131072`. Qwen3 and Gemma 3 write their losses to `metrics.jsonl` in the output directory. Modded-NanoGPT writes its logs to `logs/`.

Set `MUON_LOG_SPECTRAL_NORMS=1` to log the spectral norm of each orthogonalized matrix. Turn it off for timing comparisons.
