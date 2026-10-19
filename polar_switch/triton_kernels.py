# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Keller Jordan
# Modified: one kernel for the Gram matrix and the fused inner iterations.
"""Compute symmetric matrix products with fused additions in Triton.

This is the Triton backend of make_polynomial_orthogonalizer, used when QuACK
does not apply, for example for matrices with a dimension below 256. It
evaluates the same circuit as gemm_symmetric_fused.py: each program computes
one lower-triangle tile in FP32, applies a, b*B[r], and c*I, stores the result
once in FP16, and writes the transposed tile into the upper triangle. Products
with X use PyTorch matrix multiplication.

Modded-NanoGPT
Adapted from the MIT-licensed symmetric matrix multiplication kernels by Byron
Xu in Modded-NanoGPT. See THIRD_PARTY.md for the modifications.
Adapted source: https://github.com/KellerJordan/modded-nanogpt/blob/bc3a0c2d640d0d73dedaef87eae26148d2e32afb/triton_kernels.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _symmetric(A, C, D, N: tl.constexpr, K: tl.constexpr,
               ab, ar, ac, cb, cr, cc, db, dr, dc,
               alpha, beta, diag, ADD: tl.constexpr,
               BLOCK: tl.constexpr, BK: tl.constexpr):
    """Compute one lower tile of alpha*A @ A.T + beta*C + diag*I and mirror it."""
    tiles = tl.cdiv(N, BLOCK)
    pid = tl.program_id(0)
    batch = pid // (tiles * tiles)
    row, col = (pid % (tiles * tiles)) // tiles, pid % tiles
    row, col = tl.swizzle2d(row, col, tiles, tiles, 8)
    if col > row:
        return
    rows = row * BLOCK + tl.arange(0, BLOCK)
    cols = col * BLOCK + tl.arange(0, BLOCK)
    kk = tl.arange(0, BK)
    ap = A + batch * ab + rows[:, None] * ar + kk[None, :] * ac
    bp = A + batch * ab + cols[:, None] * ar + kk[None, :] * ac
    acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        a = tl.load(ap, (rows[:, None] < N) & (kk[None, :] < K - k * BK), 0.)
        b = tl.load(bp, (cols[:, None] < N) & (kk[None, :] < K - k * BK), 0.)
        acc = tl.dot(a, tl.trans(b), acc)
        ap += BK * ac
        bp += BK * ac
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    if ADD:
        addend = tl.load(C + batch * cb + rows[:, None] * cr + cols[None, :] * cc,
                        mask, 0.).to(tl.float32)
        acc = alpha * acc + beta * addend
        acc += tl.where(rows[:, None] == cols[None, :], diag, 0.)
    value = acc.to(D.dtype.element_ty)
    tl.store(D + batch * db + rows[:, None] * dr + cols[None, :] * dc, value, mask)
    # One owner per symmetric tile permits C == D without cross-program races.
    if row != col:
        tl.store(D + batch * db + cols[:, None] * dr + rows[None, :] * dc,
                 tl.trans(value), tl.trans(mask))


def _launch(A, C, out, alpha=1.0, beta=0.0, diag=0.0, add=False):
    """Launch one program per 128 x 128 tile of out."""
    batch, n, k = A.shape
    _symmetric[(batch * triton.cdiv(n, 128) ** 2,)](
        A, C, out, n, k, *A.stride(), *C.stride(), *out.stride(),
        alpha, beta, diag, ADD=add, BLOCK=128, BK=64, num_warps=8, num_stages=4)
    return out


def gram(X, tall, out):
    """Write X @ X.T, or X.T @ X for tall X, into out."""
    return _launch(X.mT if tall else X, out, out)


def square(A, C, out, alpha, beta, diag):
    """Write alpha*A @ A + beta*C + diag*I into out; out may be C but not A."""
    return _launch(A, C, out, alpha, beta, diag, add=True)


def apply(X, B, tall, residual):
    """Return B @ X + residual*X, or X @ B + residual*X for tall X."""
    left, right = (X, B) if tall else (B, X)
    if residual == 0.0:
        return torch.bmm(left, right)
    return torch.baddbmm(X, left, right, beta=residual)
