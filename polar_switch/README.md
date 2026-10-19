# Matrix orthogonalization

## 💡 How Polar Switch works

For a momentum matrix $M=U\Sigma V^{\mathsf T}$, Muon uses the polar factor $UV^{\mathsf T}$. Let $M\in\mathbb{R}^{n\times m}$ with $n\le m$. Polar Switch normalizes $M$ and runs $T$ outer iterations. Each outer iteration is a *circuit* of $k_t$ inner iterations on $g\in\{2,3\}$ reusable square buffers:

```text
X <- fp16(M / (||M||_F + 1e-7))                  # normalized in FP32
for t = 1, ..., T:
    B_1 <- X @ X.T                               # symmetric product
    for tau = 1, ..., k_t:
        B_w <- a * B_i @ B_j + b * B_r + c * I   # one fused kernel
    X <- B_w @ X                                 # B_w is the last buffer written
return X
```

Each product runs on FP16 Tensor Cores with FP32 accumulation. The kernel applies $a$, the addend $bB_r$, and the scalar diagonal shift $cI$ in FP32, and then stores the result once in FP16.

The baselines use the same kernels. `JORDAN` and `POLAR_EXPRESS` run five quintic iterations $X\leftarrow a_tX+(b_tA+c_tA^2)X$ with $A=XX^{\mathsf T}$. `GNS` runs the official [Gram Newton–Schulz](https://dao-ailab.github.io/blog/2026/gram-newton-schulz/) recurrence with a restart after the second iteration. For square matrices, `GNS` uses `POLAR_EXPRESS`. We center the final intervals of `JORDAN` and `GNS` at one.

<a id="methods-and-circuits"></a>

## Methods

In a method name, `PS2` and `PS3` mean two and three square buffers, and the digits give the inner depths of the outer iterations. For example, `PS2_512` has three outer iterations with 5, 1, and 2 inner iterations. **We recommend `PS2_55` and `PS3_55`.** Use inner depths of at most 5. In Modded-NanoGPT training, a first outer iteration with inner depth 6 (`PS2_64`, `PS3_64`) was numerically unstable in rare cases.

The table gives the memory in matrix elements, the FLOPs, and the final interval for singular values in $[0.001,1]$, for $X\in\mathbb{R}^{n\times m}$ with $n\le m$ and $\alpha=m/n$.

<div align="center">

| Method | Square buffers | Inner depths | Memory | FLOPs | Final interval |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `JORDAN` | 2 | 1, 1, 1, 1, 1 | $n^2(2\alpha+2)$ | $(15\alpha+5)n^3$ | $[0.562544, 1.437456]$ |
| `POLAR_EXPRESS` | 2 | 1, 1, 1, 1, 1 | $n^2(2\alpha+2)$ | $(15\alpha+5)n^3$ | $[0.858872, 1.141128]$ |
| `GNS` | – | five quintics with a restart | $n^2(2\alpha+4)$ | $(6\alpha+14)n^3$ | $[0.787021, 1.212979]$ |
| `PS2_55` | 2 | 5, 5 | $n^2(2\alpha+2)$ | $(6\alpha+10)n^3$ | $[0.872263, 1.106236]$ |
| `PS2_64` | 2 | 6, 4 | $n^2(2\alpha+2)$ | $(6\alpha+10)n^3$ | $[0.897110, 1.081213]$ |
| `PS2_54` | 2 | 5, 4 | $n^2(2\alpha+2)$ | $(6\alpha+9)n^3$ | $[0.771715, 1.201363]$ |
| `PS2_512` | 2 | 5, 1, 2 | $n^2(2\alpha+2)$ | $(9\alpha+8)n^3$ | $[0.894082, 1.091006]$ |
| `PS2_4111` | 2 | 4, 1, 1, 1 | $n^2(2\alpha+2)$ | $(12\alpha+7)n^3$ | $[0.930600, 1.056149]$ |
| `PS2_11111` | 2 | 1, 1, 1, 1, 1 | $n^2(2\alpha+2)$ | $(15\alpha+5)n^3$ | $[0.867287, 1.118678]$ |
| `PS3_55` | 3 | 5, 5 | $n^2(2\alpha+3)$ | $(6\alpha+10)n^3$ | $[0.898309, 1.080954]$ |
| `PS3_64` | 3 | 6, 4 | $n^2(2\alpha+3)$ | $(6\alpha+10)n^3$ | $[0.913241, 1.070202]$ |
| `PS3_54` | 3 | 5, 4 | $n^2(2\alpha+3)$ | $(6\alpha+9)n^3$ | $[0.800196, 1.178895]$ |

</div>

## Coefficients

Each constant in [coefficients.py](coefficients.py) has the name of its method. It is a sequence of outer iterations `(residual, steps)`. Each inner iteration in `steps` is

```text
(w, i, r, a, b, c)    meaning    B[w] <- a * B[i] @ B[i] + b * B[r] + c * I
```

Buffer numbers start at 0. Thus, `B[0]` holds the Gram matrix ($B_1$ above). Every circuit uses squares, so $i=j$. The outer iteration ends with $X\leftarrow B_wX+\delta X$, where $\delta$ is `residual`.

The coefficients include the buffer scaling and the final centering factor. To compute new two-buffer circuits, see the [coefficient search guide](../coefficient_search/README.md).

## Usage notes

`make_polynomial_orthogonalizer(method, compile=True, epsilon=1e-7)` accepts a method name (not case-sensitive) or a coefficient table. It returns a callable for real floating-point CUDA tensors with shape `(..., rows, columns)`. Turn off autograd and autocast when you use the callable. It does not change the input, and the output keeps the shape and dtype of the input. `compile=False` skips `torch.compile`. The implementation is in [\_\_init\_\_.py](__init__.py). The QuACK kernels are in [gemm_symmetric_fused.py](gemm_symmetric_fused.py), the Triton kernels are in [triton_kernels.py](triton_kernels.py), and `GNS` is in [gram_newton_schulz.py](gram_newton_schulz.py).

**Backends:** The [main README](../README.md#pytorch-and-triton) describes the fallback order, the rounding, and the hardware. Each fallback gives a `RuntimeWarning` and adds an entry to `orthogonalize.fallback_events`. `orthogonalize.last_execution` records the requested method, the method that ran, and the backend. Out-of-memory errors and CUDA errors are raised, not caught. Results from different backends are not always bitwise identical.

**`GNS`:** For rectangular matrices, `GNS` uses QuACK on supported GPUs and shapes. Otherwise, it uses the official PyTorch recurrence. If this path fails, for example when QuACK is not installed on a supported GPU, `GNS` runs `POLAR_EXPRESS` for that shape and records the event. `last_execution` also records the path: `rectangular_gram`, `square_polar_express`, or `fallback_polar_express`.
