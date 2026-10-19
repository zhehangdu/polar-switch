# SPDX-License-Identifier: MIT
"""Approximate matrix polar factors with Polar Switch polynomial circuits.

Each method is a sequence of outer iterations. An outer iteration forms the
Gram matrix of X, evaluates a circuit of inner iterations on two or three
square buffers, and multiplies the last buffer into X. The coefficient tables
are in coefficients.py; Gram Newton-Schulz is in gram_newton_schulz.py.
See polar_switch/README.md for the circuits, methods, and backends.
"""

from functools import partial
import importlib
import math
import warnings

import torch

from . import coefficients

# Method names accepted by make_polynomial_orthogonalizer and the trainers.
METHODS = ("JORDAN", "POLAR_EXPRESS", "GNS", "PS2_55", "PS2_64", "PS2_54", "PS2_512",
           "PS2_4111", "PS2_11111", "PS3_55", "PS3_64", "PS3_54")


def _run(X, table, ops, epsilon, buffers):
    """Evaluate a coefficient table on a batch of matrices.

    X is normalized in float32 and stored in float16. Each outer iteration
    writes the Gram matrix into buffer 0, runs its inner iterations
    B[w] = a*B[i]@B[i] + b*B[r] + c*I, and updates X = B[w] @ X + residual*X,
    or X @ B[w] + residual*X for tall matrices. ops provides gram, square, and
    apply for one backend.
    """
    X = X.float()
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)).half().contiguous()
    tall = X.shape[-2] > X.shape[-1]
    n = min(X.shape[-2:])
    work = [torch.empty((*X.shape[:-2], n, n), device=X.device, dtype=X.dtype)
            for _ in range(buffers)]
    for residual, steps in table:
        ops.gram(X, tall, work[0])
        for dst, src, add, alpha, beta, diag in steps:
            ops.square(work[src], work[add], work[dst], alpha, beta, diag)
        X = ops.apply(X, work[steps[-1][0]], tall, residual)
    return X


class _TorchPE:
    """PyTorch operations for the final Polar Express fallback."""

    @staticmethod
    def gram(X, tall, out):
        torch.bmm(X.mT, X, out=out) if tall else torch.bmm(X, X.mT, out=out)

    @staticmethod
    def square(A, C, out, alpha, beta, diag):
        # This adapter runs PE only, whose inner iterations have no diagonal shift.
        torch.baddbmm(C, A, A, alpha=alpha, beta=beta, out=out)

    @staticmethod
    def apply(X, B, tall, residual):
        return (torch.baddbmm(X, X, B, beta=residual) if tall
                else torch.baddbmm(X, B, X, beta=residual))


def _fatal(error):
    """Return True for out-of-memory and CUDA errors, which must not fall back."""
    while error is not None:
        if isinstance(error, torch.OutOfMemoryError):
            return True
        message = str(error).lower()
        if any(s in message for s in ("cuda error:", "illegal memory access", "device-side assert")):
            return True
        error = error.__cause__ or error.__context__
    return False


def make_polynomial_orthogonalizer(method, compile=True, epsilon=1e-7):
    """Create an orthogonalizer for a named method or a coefficient table.

    Args:
        method: A name in METHODS, such as "PS2_55" (case-insensitive), or a
            coefficient table in the format of coefficients.py.
        compile: Compile the evaluation with torch.compile.
        epsilon: Added to the Frobenius norm before normalization.

    Returns:
        A callable that accepts a real floating point CUDA tensor with shape
        (..., rows, columns) and returns its approximate polar factor with the
        same shape and dtype. The input is not modified. Reuse the callable
        across updates, without autograd tracking and with autocast disabled.

    The circuit runs with QuACK, then Triton, then POLAR_EXPRESS in PyTorch
    (see the README). Each fallback emits a RuntimeWarning and is appended to
    fallback_events, and last_execution records the method and backend that
    ran. "GNS" returns the orthogonalizer of gram_newton_schulz.py.
    """
    if isinstance(method, str) and method.upper() == "GNS":
        from .gram_newton_schulz import make_orthogonalizer
        return make_orthogonalizer(make_polynomial_orthogonalizer, compile, epsilon)
    table = method
    if isinstance(method, str):
        if method.upper() not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        table = getattr(coefficients, method.upper())
    name = next((n for n in METHODS if n != "GNS" and table == getattr(coefficients, n)), "CUSTOM")
    if not math.isfinite(epsilon) or epsilon <= 0 or not table:
        raise ValueError("Use a nonempty circuit and positive finite epsilon")
    buffers = 1
    for residual, steps in table:
        initialized = {0}
        if not steps or not math.isfinite(residual):
            raise ValueError("Each outer iteration needs a finite residual and at least one inner iteration")
        for dst, src, add, alpha, beta, diag in steps:
            if (any(not isinstance(i, int) or i < 0 for i in (dst, src, add))
                    or dst == src or src not in initialized or add not in initialized
                    or not all(math.isfinite(v) for v in (alpha, beta, diag))):
                raise ValueError("Invalid inner iteration: buffer indices must be nonnegative integers, the "
                                 "destination must differ from the squared buffer, operands must be B[0] "
                                 "or buffers written earlier, and coefficients must be finite")
            initialized.add(dst)
            buffers = max(buffers, dst + 1, src + 1, add + 1)
    modules, unavailable, failed, runners, hardware = {}, {}, {}, {}, {}
    events = []
    torch_pe = partial(_run, table=coefficients.POLAR_EXPRESS,
                       ops=_TorchPE, epsilon=epsilon, buffers=2)

    def polar(X):
        """Approximate the polar factor of each matrix in X."""
        if X.ndim < 2 or X.numel() == 0 or not X.is_floating_point():
            raise ValueError("Expected nonempty real matrices")
        work = X.reshape(-1, *X.shape[-2:])
        key = (X.device, X.dtype, tuple(X.shape), X.stride())
        for backend in ("quack", "triton"):
            if (backend, key) in failed:
                continue
            try:
                if not X.is_cuda:
                    raise NotImplementedError("CUDA backend unavailable on CPU")
                if X.device not in hardware:
                    hardware[X.device] = torch.cuda.get_device_capability(X.device)[0]
                capability = hardware[X.device]
                if backend == "quack" and capability not in (9, 10, 11, 12):
                    raise NotImplementedError(f"QuACK does not support SM{capability}x")
                if backend == "quack" and (min(X.shape[-2:]) < 256 or any(n % 8 for n in X.shape[-2:])):
                    raise NotImplementedError("QuACK requires dimensions >=256 and divisible by 8")
                if backend in unavailable:
                    raise ImportError(unavailable[backend])
                if backend not in modules:
                    try:
                        module_name = "gemm_symmetric_fused" if backend == "quack" else "triton_kernels"
                        modules[backend] = importlib.import_module(f".{module_name}", __name__)
                    except ImportError as error:
                        unavailable[backend] = str(error)
                        raise
                runner_key = (backend, X.device)
                if runner_key not in runners:
                    runner = partial(_run, table=table, ops=modules[backend],
                                     epsilon=epsilon, buffers=buffers)
                    # Without isolate_recompiles, all circuits share one recompile limit, and orthogonalizers
                    # created after the first eight would fall back to Polar Express.
                    runners[runner_key] = (torch.compile(runner, fullgraph=True, dynamic=False,
                                                        isolate_recompiles=True,
                                                        options={"triton.cudagraphs": False})
                                           if compile else runner)
                result = runners[runner_key](work)
                polar.last_execution = dict(requested=name, effective=name, backend=backend)
                return result.to(X.dtype).reshape(X.shape)
            except Exception as error:
                if _fatal(error):
                    raise
                failed[backend, key] = str(error)
                next_backend = "triton" if backend == "quack" else "torch"
                event = dict(requested=name, effective=name if next_backend == "triton" else "POLAR_EXPRESS",
                             failed_backend=backend, backend=next_backend, shape=list(X.shape),
                             dtype=str(X.dtype), device=str(X.device),
                             reason=f"{type(error).__name__}: {error}")
                events.append(event)
                warnings.warn(f"Polar fallback: {event}", RuntimeWarning, stacklevel=2)
        result = torch_pe(work)
        polar.last_execution = dict(requested=name, effective="POLAR_EXPRESS", backend="torch")
        return result.to(X.dtype).reshape(X.shape)

    polar.fallback_events = events
    polar.last_execution = None
    return polar
