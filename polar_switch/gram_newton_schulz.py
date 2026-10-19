# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jack Zhang, Noah Amsel, Berlin Chen, and Tri Dao
# Modified: centered last iteration, square-input dispatch, and Polar Express fallback.
"""Approximate matrix polar factors with Gram Newton-Schulz (GNS).

For X with n <= m, the Gram recurrence forms R = X @ X.T once and applies each
quintic multiplier h_t(z) = a_t + b_t*z + c_t*z**2 to R instead of X. Q is the
product of the multipliers h_t(R) since the last restart. At a restart, X is
replaced by Q @ X and the Gram matrix is formed again. Tall matrices use
X.T @ X and X @ Q.

make_orthogonalizer creates the package's GNS method. Rectangular inputs use
the upstream recurrence with its Polar Express preset and a restart after the
second iteration; the coefficients of the last iteration are scaled so that the
final interval is centered at one. Square inputs use the package's POLAR_EXPRESS.

Gram Newton-Schulz
Adapted from its MIT-licensed implementation. See THIRD_PARTY.md for the
modifications.
Adapted source: https://github.com/Dao-AILab/gram-newton-schulz/blob/e45d0aca7083cb275c9a303220c05c4abecd9187/gram_newton_schulz/gram_newton_schulz.py
Coefficients: https://github.com/Dao-AILab/gram-newton-schulz/blob/e45d0aca7083cb275c9a303220c05c4abecd9187/gram_newton_schulz/coefficients.py
Blog: https://dao-ailab.github.io/blog/2026/gram-newton-schulz/
"""

import math
from types import SimpleNamespace
from typing import Dict, List, Optional
import warnings

import torch
from torch import Tensor

_unmodified_polar_express_coefficients = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
safety_factor = 1.05
POLAR_EXPRESS_COEFFICIENTS = [
    (a / safety_factor, b / safety_factor**3, c / safety_factor**5)
    for (a, b, c) in _unmodified_polar_express_coefficients
]
# The package's GNS: the preset above with its last iteration multiplied by
# 2 / (L + U) = 1.0795859163621828, where [L, U] is the preset's final interval
# for singular values in [0.001, 1]. This centers the final interval at one.
GNS_COEFFICIENTS = POLAR_EXPRESS_COEFFICIENTS[:-1] + [
    (2.365477637327074, -1.5563980023821433, 0.3542623265416755),
]


SYMMETRIC_KERNEL_TILE_SIZE = 256


_TORCH_BACKEND = SimpleNamespace(
    sym_mm=lambda A, B: A @ B,
    sym_baddbmm=lambda A, B, C, alpha=1., beta=1.: torch.baddbmm(C, A, B, alpha=alpha, beta=beta),
    mm=lambda A, B: A @ B,
    mm_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)


def _make_kernel_backend():
    """Return the QuACK operations used by the recurrences."""
    from quack.gemm_interface import gemm_symmetric, gemm, gemm_add
    return SimpleNamespace(
        sym_mm=gemm_symmetric,
        sym_baddbmm=lambda A, B, C, alpha=1., beta=1.: gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta),
        mm=lambda A, B: gemm(A, B, tuned=False),
        mm_add=lambda A, B, C, beta: gemm_add(A, B, C=C, beta=beta, tuned=False),
    )


def _make_compiled_gram(ops, ns_coefficients, gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs):
    """Build a compiled closure for gram Newton-Schulz with a fixed backend."""
    ns_coefficients = list(ns_coefficients)
    gram_newton_schulz_reset_iterations = set(gram_newton_schulz_reset_iterations)

    def _gram_newton_schulz(X: Tensor) -> Tensor:
        tall_skinny = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
        X = X.to(torch.float16)

        if tall_skinny:
            R = ops.sym_mm(X.mT, X)
        else:
            R = ops.sym_mm(X, X.mT)

        batch_size = R.size(0)
        I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        Q = None

        for i, (a, b, c) in enumerate(ns_coefficients):
            if i in gram_newton_schulz_reset_iterations and i != 0:
                if tall_skinny:
                    X = ops.mm(X, Q)
                    R = ops.sym_mm(X.mT, X)
                else:
                    X = ops.mm(Q, X)
                    R = ops.sym_mm(X, X.mT)
                Q = None

            Z = ops.sym_baddbmm(R, R, C=R, alpha=c, beta=b)
            if i == 0 or i in gram_newton_schulz_reset_iterations:
                Q = Z + a * I
            else:
                Q = ops.sym_baddbmm(Q, Z, C=Q, beta=a)
            if i < len(ns_coefficients) - 1 and i + 1 not in gram_newton_schulz_reset_iterations:
                RZ = ops.sym_baddbmm(R, Z, C=R, beta=a)
                R = ops.sym_baddbmm(Z, RZ, C=RZ, beta=a)

        if tall_skinny:
            X = ops.mm(X, Q)
        else:
            X = ops.mm(Q, X)
        return X

    if compile_kwargs is not None:
        _gram_newton_schulz = torch.compile(_gram_newton_schulz, **compile_kwargs)
    return _gram_newton_schulz


def _make_compiled_standard(ops, ns_coefficients, ns_epsilon, compile_kwargs):
    """Build a compiled closure for standard Newton-Schulz with a fixed backend."""
    ns_coefficients = list(ns_coefficients)

    def _standard_newton_schulz(X: Tensor) -> Tensor:
        tall_skinny = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
        X = X.to(torch.float16)

        for a, b, c in ns_coefficients:
            if tall_skinny:
                A = ops.sym_mm(X.mT, X)
            else:
                A = ops.sym_mm(X, X.mT)
            B = ops.sym_baddbmm(A, A, C=A, alpha=c, beta=b)
            if tall_skinny:
                X = ops.mm_add(X, B, C=X, beta=a)
            else:
                X = ops.mm_add(B, X, C=X, beta=a)
        return X

    if compile_kwargs is not None:
        _standard_newton_schulz = torch.compile(_standard_newton_schulz, **compile_kwargs)
    return _standard_newton_schulz


class GramNewtonSchulz:
    """
    Gram Newton-Schulz orthogonalization.

    Example:
        from polar_switch.gram_newton_schulz import GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS
        gram_NS = GramNewtonSchulz(
            ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
            gram_newton_schulz_reset_iterations=[2]
        )
        result = gram_NS(X)
    """

    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Optional[List[List[float]]] = None,
        use_gram_newton_schulz: bool = True,
        gram_newton_schulz_reset_iterations: List[int] = None,
        compile_kwargs: Optional[Dict] = {"fullgraph": True, "mode": "reduce-overhead"},
    ):
        """
        Initialize GramNewtonSchulz orthogonalizer.

        Args:
            ns_epsilon: Epsilon for normalization
            ns_use_kernels: Whether to use custom CuTeDSL kernels
            ns_coefficients: Coefficients for each iteration. Defaults to POLAR_EXPRESS_COEFFICIENTS.
            gram_newton_schulz_reset_iterations: Iterations at which to reset. Defaults to [2].
            compile_kwargs: Keyword arguments forwarded to torch.compile for the inner Newton-Schulz functions.
                Defaults to {"fullgraph": True, "mode": "reduce-overhead"}. Pass None to disable compilation.
        """
        self.ns_epsilon = ns_epsilon
        self.ns_use_kernels = ns_use_kernels
        self.ns_coefficients = ns_coefficients if ns_coefficients is not None else POLAR_EXPRESS_COEFFICIENTS
        self.use_gram_newton_schulz = use_gram_newton_schulz
        if use_gram_newton_schulz:
            self.gram_newton_schulz_reset_iterations = gram_newton_schulz_reset_iterations if gram_newton_schulz_reset_iterations is not None else [2]

        kernel_backend = _make_kernel_backend() if self.ns_use_kernels else None

        # Build compiled closures for each (backend, algorithm) combination.
        # This avoids compiling bound methods, which torch.compile handles poorly.
        if use_gram_newton_schulz:
            self._gram_torch = _make_compiled_gram(
                _TORCH_BACKEND, self.ns_coefficients, self.gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs)
            if kernel_backend is not None:
                self._gram_kernel = _make_compiled_gram(
                    kernel_backend, self.ns_coefficients, self.gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs)

        self._standard_torch = _make_compiled_standard(
            _TORCH_BACKEND, self.ns_coefficients, ns_epsilon, compile_kwargs)
        if kernel_backend is not None:
            self._standard_kernel = _make_compiled_standard(
                kernel_backend, self.ns_coefficients, ns_epsilon, compile_kwargs)

        self._kernel_backend = kernel_backend

    def _select_gram(self, X: Tensor) -> Tensor:
        if self._kernel_backend is not None and min(X.size(-2), X.size(-1)) >= SYMMETRIC_KERNEL_TILE_SIZE:
            return self._gram_kernel(X)
        return self._gram_torch(X)

    def _select_standard(self, X: Tensor) -> Tensor:
        if self._kernel_backend is not None and min(X.size(-2), X.size(-1)) >= SYMMETRIC_KERNEL_TILE_SIZE:
            return self._standard_kernel(X)
        return self._standard_torch(X)

    def __call__(self, X: Tensor) -> Tensor:
        """
        Orthogonalize a batch of matrices using Gram Newton-Schulz iteration.

        Args:
            X: Input tensor of shape (batch, M, N) or (M, N)
               Will be treated as a batch of 2D matrices

        Returns:
            Orthogonalized tensor with same shape as input
        """
        original_shape = X.shape
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.view(-1, *X.shape[-2:])

        original_dtype = X.dtype

        if self.use_gram_newton_schulz and max(X.shape[-2:]) > min(X.shape[-2:]):
            X = self._select_gram(X)
        else:
            X = self._select_standard(X)

        return X.to(original_dtype).view(original_shape)


class StandardNewtonSchulz(GramNewtonSchulz):
    """
    Standard Newton-Schulz orthogonalization.

    Equivalent to GramNewtonSchulz with use_gram_newton_schulz=False.

    Example:
        from polar_switch.gram_newton_schulz import StandardNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS
        standard_NS = StandardNewtonSchulz(ns_coefficients=POLAR_EXPRESS_COEFFICIENTS)
        result = standard_NS(X)
    """

    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Optional[List[List[float]]] = None,
        compile_kwargs: Optional[Dict] = {"fullgraph": True, "mode": "reduce-overhead"},
    ):
        super().__init__(
            ns_epsilon=ns_epsilon,
            ns_use_kernels=ns_use_kernels,
            ns_coefficients=ns_coefficients,
            use_gram_newton_schulz=False,
            compile_kwargs=compile_kwargs,
        )


def make_orthogonalizer(pe_factory, compile=True, epsilon=1e-7):
    """Create the GNS orthogonalizer.

    Args:
        pe_factory: make_polynomial_orthogonalizer, used to create the
            package's POLAR_EXPRESS.
        compile: Compile the evaluation with torch.compile.
        epsilon: Added to the Frobenius norm before normalization.

    Returns:
        A callable with the interface of make_polynomial_orthogonalizer.
        Rectangular inputs use the centered Gram recurrence with the QuACK
        kernels when the GPU and shape support them, otherwise with PyTorch.
        Square inputs use POLAR_EXPRESS.

    If the Gram recurrence fails, the callable falls back to POLAR_EXPRESS for
    that shape (see polar_switch/README.md).
    """
    from . import _fatal

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Use a positive finite epsilon")
    pe = pe_factory("POLAR_EXPRESS", compile=compile, epsilon=epsilon)
    instances, failed, capability = {}, set(), {}
    compile_kwargs = (dict(fullgraph=True, dynamic=False,
                           options={"triton.cudagraphs": False}) if compile else None)

    def kernels_supported(X):
        """Return True if the QuACK kernels support the device and shape of X."""
        if (not X.is_cuda or min(X.shape[-2:]) < SYMMETRIC_KERNEL_TILE_SIZE
                or any(n % 8 for n in X.shape[-2:])):
            return False
        if X.device not in capability:
            capability[X.device] = torch.cuda.get_device_capability(X.device)[0]
        return capability[X.device] in (9, 10, 11, 12)

    def polar_express(X, dispatch):
        """Evaluate POLAR_EXPRESS and record the GNS dispatch."""
        result = pe(X)
        polar.last_execution = dict(pe.last_execution, requested="GNS",
                                    dispatch=dispatch)
        return result

    def polar(X):
        """Approximate the polar factor of each matrix in X."""
        if X.ndim < 2 or X.numel() == 0 or not X.is_floating_point():
            raise ValueError("Expected nonempty real matrices")
        if X.shape[-2] == X.shape[-1]:
            return polar_express(X, "square_polar_express")
        key = (X.device, X.dtype, tuple(X.shape), X.stride())
        if key not in failed:
            use_kernels = kernels_supported(X)
            try:
                if (X.device, use_kernels) not in instances:
                    instances[X.device, use_kernels] = GramNewtonSchulz(
                        ns_epsilon=epsilon, ns_use_kernels=use_kernels,
                        ns_coefficients=GNS_COEFFICIENTS,
                        gram_newton_schulz_reset_iterations=[2],
                        compile_kwargs=compile_kwargs)
                work = X.reshape(-1, *X.shape[-2:])
                result = instances[X.device, use_kernels]._select_gram(work)
                polar.last_execution = dict(requested="GNS", effective="GNS",
                                            backend="quack" if use_kernels else "torch",
                                            dispatch="rectangular_gram")
                return result.to(X.dtype).reshape(X.shape)
            except Exception as error:
                if _fatal(error):
                    raise
                failed.add(key)
                event = dict(requested="GNS", effective="POLAR_EXPRESS",
                             failed_backend="quack" if use_kernels else "torch",
                             backend=None, shape=list(X.shape), dtype=str(X.dtype),
                             device=str(X.device), reason=f"{type(error).__name__}: {error}")
                pe.fallback_events.append(event)
                warnings.warn(f"Polar fallback: {event}", RuntimeWarning, stacklevel=2)
        return polar_express(X, "fallback_polar_express")

    # Square dispatch is part of the GNS definition. Only real failures, of the
    # Gram path or of the delegated Polar Express backends, add events here.
    polar.fallback_events = pe.fallback_events
    polar.last_execution = None
    return polar
