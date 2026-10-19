# Third-party sources and licenses

Original Polar Switch code uses the MIT License ([LICENSE](LICENSE)). Adapted code keeps its license, listed below; the license texts are in [licenses/](licenses/), and the notices are kept in the adapted files. Dependencies, models, tokenizers, and datasets are not redistributed and retain their own terms. Source links are pinned to the commits and revisions used in the paper's experiments.

## Code adapted in this repository

<div align="center">

| Source | Files in this repository | License |
| :---: | :---: | :---: |
| [QuACK](https://github.com/Dao-AILab/quack/tree/10ce505a7ecf68e46753009bd08dcf9bd6c059ee) (`quack-kernels` 0.6.5) | `polar_switch/gemm_symmetric_fused.py` | Apache-2.0 ([text](licenses/quack-Apache-2.0.txt)) |
| [Gram Newton–Schulz](https://github.com/Dao-AILab/gram-newton-schulz/tree/e45d0aca7083cb275c9a303220c05c4abecd9187) | `polar_switch/gram_newton_schulz.py` | MIT ([text](licenses/gram-newton-schulz-MIT.txt)) |
| [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt/tree/bc3a0c2d640d0d73dedaef87eae26148d2e32afb) record #91 | `training/train_nanogpt.py`, `training/cached_fineweb10B.py`, `polar_switch/triton_kernels.py` | MIT ([text](licenses/modded-nanogpt-MIT.txt)) |

</div>

### QuACK

By the QuACK team. `gemm_symmetric_fused.py` adapts QuACK's symmetric GEMM and uses its epilogue interface. Modifications add a scalar diagonal shift in the FP32 epilogue, allow the output buffer to be the matrix addend, and skip the mirrored store for diagonal tiles.

[Symmetric GEMM](https://github.com/Dao-AILab/quack/blob/10ce505a7ecf68e46753009bd08dcf9bd6c059ee/quack/gemm_symmetric.py) · [Epilogue interface](https://github.com/Dao-AILab/quack/blob/10ce505a7ecf68e46753009bd08dcf9bd6c059ee/quack/epilogue/frontend.py)

### Gram Newton–Schulz

By Jack Zhang, Noah Amsel, Berlin Chen, and Tri Dao. `gram_newton_schulz.py` keeps the Gram recurrence, its restart after the second iteration, and its Polar Express coefficient preset with safety factor 1.05. Modifications scale the coefficients of the last iteration by 1.0795859163621828, which centers the final singular value interval at one; evaluate square inputs with this package's `POLAR_EXPRESS`; and record the executed path.

[Blog](https://dao-ailab.github.io/blog/2026/gram-newton-schulz/) · [Implementation](https://github.com/Dao-AILab/gram-newton-schulz/blob/e45d0aca7083cb275c9a303220c05c4abecd9187/gram_newton_schulz/gram_newton_schulz.py) · [Coefficients](https://github.com/Dao-AILab/gram-newton-schulz/blob/e45d0aca7083cb275c9a303220c05c4abecd9187/gram_newton_schulz/coefficients.py)

### Modded-NanoGPT

By Keller Jordan and contributors. `train_nanogpt.py` combines the record #91 training script with its Triton kernels. Modifications use FlashAttention-4 instead of FlashAttention-3 on the B200 and call it, and one fused attention correction, outside `torch.compile`; call this package's orthogonalizer in the NorMuon update; add a `--seed` option, `DATA_PATH`, and optional spectral norm logging; target CUDA compilation to the current GPU; and correct an autograd backward return. `triton_kernels.py` adapts the symmetric Gram kernels by Byron Xu and keeps their fixed tile configuration, tuned there for H100. Modifications use one kernel for both the Gram matrix and the inner iterations, add the diagonal shift, and allow the output to be the addend. `cached_fineweb10B.py` adapts the data downloader. Modifications pin the dataset revision, write under `DATA_PATH`, and check the number of shards.

[train_gpt.py](https://github.com/KellerJordan/modded-nanogpt/blob/bc3a0c2d640d0d73dedaef87eae26148d2e32afb/train_gpt.py) · [triton_kernels.py](https://github.com/KellerJordan/modded-nanogpt/blob/bc3a0c2d640d0d73dedaef87eae26148d2e32afb/triton_kernels.py) · [dc_triton_kernels.py](https://github.com/KellerJordan/modded-nanogpt/blob/bc3a0c2d640d0d73dedaef87eae26148d2e32afb/dc_triton_kernels.py) · [cached_fineweb10B.py](https://github.com/KellerJordan/modded-nanogpt/blob/bc3a0c2d640d0d73dedaef87eae26148d2e32afb/data/cached_fineweb10B.py)

## Coefficients and methods

These sources define baseline coefficients or algorithms. No code from them is included.

- **Muon:** `JORDAN` uses Muon's fixed quintic $p_t(x)=3.4445x-4.7750x^3+2.0315x^5$, with the last iteration multiplied by 1.1955197492983572 to center the final interval at one. The Qwen3 and Gemma 3 trainers use Muon's Nesterov momentum update. [Blog](https://kellerjordan.github.io/posts/muon/) · [muon.py](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py)
- **Polar Express:** `POLAR_EXPRESS` uses the five quintics produced by the official code with safety factor 0.02 and cushion 0.02. [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/df88b275bef31ac96c85f0c4013734fc-Paper-Conference.pdf) · [polar_express.py](https://github.com/NoahAmsel/PolarExpress/blob/71cc37943d99cae780024c1d198977f2f8795407/polar_express.py)
- **NorMuon:** the hidden-matrix optimizer of Modded-NanoGPT record #91, kept unchanged.

## Dependencies

These are installed by `pyproject.toml` and the files in `training/`; none is redistributed here.

<div align="center">

| Package | Version | Used for | License |
| :---: | :---: | :---: | :---: |
| [PyTorch](https://github.com/pytorch/pytorch/tree/v2.14.0) | 2.14.0 (CUDA 13.0) | Tensors, compilation, and cuBLAS products | BSD-3-Clause, with bundled third-party licenses |
| [Triton](https://github.com/triton-lang/triton/tree/c01b6774b1865984607d89d89d3a10833de92037) | 3.8.0 | Circuit kernels when QuACK does not apply, such as matrices with a dimension below 256 | MIT |
| [QuACK](https://github.com/Dao-AILab/quack/tree/10ce505a7ecf68e46753009bd08dcf9bd6c059ee) (`quack-kernels`) | 0.6.5 | Symmetric and rectangular products; fused MLP and loss for Qwen3 and Gemma 3 | Apache-2.0 |
| [CUTLASS DSL](https://github.com/NVIDIA/cutlass/tree/098de2a652cf8f00fd70b2df54051c7eccbb855a) (`nvidia-cutlass-dsl`) | 4.8.0 | Kernel language of QuACK and the fused epilogue | NVIDIA Software License Agreement |
| [cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend/tree/42286a6fb8792055f3db80403350e80114ea086c) (`nvidia-cudnn-frontend`) | 1.30.0 | Attention for Qwen3 and Gemma 3 | Apache-2.0 and MIT |
| cuDNN (`nvidia-cudnn-cu13`) | 9.24.0.43 | Attention kernels called by the cuDNN frontend | NVIDIA proprietary license |
| [FlashAttention-4](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute) (`flash-attn-4`) | commit `1bda8f9` | Attention for Modded-NanoGPT | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers/tree/v5.17.0) | 5.17.0 | Qwen3 and Gemma 3 model implementations | Apache-2.0 |
| Hugging Face Hub, Datasets, Tokenizers | 1.28.0, 4.8.5, 0.23.2 | Downloads and tokenization | Apache-2.0 |
| [tiktoken](https://github.com/openai/tiktoken) | 0.14.0 | GPT-2 tokenizer for Modded-NanoGPT | MIT |
| NumPy | 2.5.3 | Training data handling and coefficient search | BSD-3-Clause, with bundled third-party licenses |
| SciPy | 1.18.1 for the coefficient search | Coefficient search | BSD-3-Clause |
| einops, apache-tvm-ffi, torch-c-dlpack-ext | 0.8.2, 0.1.14.post1, 0.1.5 | Training dependencies | MIT, Apache-2.0, Apache-2.0 |

</div>

## Models, tokenizers, and data

All models start from random initialization; no pretrained weights are used. Downloads are pinned to the following revisions.

<div align="center">

| Item | Source | Revision | Terms |
| :---: | :---: | :---: | :---: |
| FineWeb-Edu `sample-100BT` | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9) | `87f0914` | ODC-By 1.0 and the Common Crawl Terms of Use |
| GPT-2-tokenized FineWeb | [kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/tree/889765ea1f903759787add96995d81171b632d0c) | `889765e` | See the dataset card and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| Qwen3-0.6B tokenizer and architecture | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de` | See the model card |
| Gemma 3-1B tokenizer and architecture | [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt/tree/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29) | `fcf18a2` | Gemma terms; access requires approval on Hugging Face |

</div>

Model papers: [Qwen3](https://arxiv.org/abs/2505.09388) · [Gemma 3](https://arxiv.org/abs/2503.19786)
