# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, QuACK team
# Modified: diagonal epilogue, symmetric in-place output, and Polar Switch operations.
"""Compute symmetric matrix products with fused additions and diagonal shifts.

This is the QuACK backend of make_polynomial_orthogonalizer. Each inner
iteration B[w] = a*B[i] @ B[i] + b*B[r] + c*I is one symmetric GEMM. The kernel
computes the tiles of one triangle, applies a, b*B[r], and c*I in the FP32
epilogue, stores the result once in FP16, and writes the transposed tile into
the other triangle. The diagonal shift is a scalar; no identity matrix is
formed. The output can be the addend buffer, w = r. gram, square, and apply
are the operations called by the outer iteration in __init__.py.

QuACK
Adapted from QuACK's Apache-2.0 gemm_symmetric.py and its epilogue interface.
See THIRD_PARTY.md for the modifications.
Adapted source: https://github.com/Dao-AILab/quack/blob/10ce505a7ecf68e46753009bd08dcf9bd6c059ee/quack/gemm_symmetric.py
Epilogue interface: https://github.com/Dao-AILab/quack/blob/10ce505a7ecf68e46753009bd08dcf9bd6c059ee/quack/epilogue/frontend.py
"""

import functools
from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import get_device_capacity, mlir_namedtuple
from quack.epilogue.frontend import epilogue_from_class
from quack.epilogue.ops import Scalar, TileStore
from quack.gemm_config import GemmConfig
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_interface import gemm, gemm_add, gemm_symmetric
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_sm120 import GemmSm120
from quack.rounding import RoundingMode
from quack.tile_scheduler import TriangularTileScheduler
import quack.utils as utils


def _symmetric_offdiag_pred(gemm, tile_coord_mnkl):
    """Write the transposed tile only when the tile is off the diagonal."""
    square_tile_m = tile_coord_mnkl[0] // gemm.cluster_shape_mnk[0]
    square_tile_n = tile_coord_mnkl[1] // gemm.cluster_shape_mnk[1]
    return square_tile_m != square_tile_n


class Diagonal(Scalar):
    """Add a scalar to diagonal entries of the output tile."""

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        value = utils.load_scalar_or_pointer(param)
        coords = ctx.partition_for_epilogue_fn(
            cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
        )
        row_offset = ctx.tile_coord_mnkl[0] * ctx.tile_M
        col_offset = ctx.tile_coord_mnkl[1] * ctx.tile_N
        square_tile_m = ctx.tile_coord_mnkl[0] // gemm.cluster_shape_mnk[0]
        square_tile_n = ctx.tile_coord_mnkl[1] // gemm.cluster_shape_mnk[1]
        rows = ctx.tile_M // cute.size(coords.shape[3])
        cols = ctx.tile_N // cute.size(coords.shape[4])
        return value, coords, row_offset, col_offset, square_tile_m == square_tile_n, rows, cols

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        coords = state[1][None, None, None, epi_coord[0], epi_coord[1]]
        # A diagonal cluster also contains subtiles that cannot touch the diagonal.
        row_start = state[2] + epi_coord[0] * state[5]
        col_start = state[3] + epi_coord[1] * state[6]
        overlaps = state[4] & (row_start < col_start + state[6]) & (col_start < row_start + state[5])
        return state[0], coords, state[2], state[3], overlaps


class GemmSymmetricFusedMixin(GemmDefaultEpiMixin):
    """Add the diagonal shift and the transposed store to QuACK's epilogue."""

    _epi_ops = GemmDefaultEpiMixin._epi_ops + (
        Diagonal("diag"),
        TileStore("mAuxOut", store_pred_fn=_symmetric_offdiag_pred),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: cute.Tensor
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        diag: Optional[Float32 | cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def get_scheduler_class(self, varlen_m: bool = False):
        return TriangularTileScheduler

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        values = self._epi_ops_to_params_dict(args)
        values["split_k_semaphore"] = None
        values["split_k_workspace"] = None
        return self.EpilogueParams(**values)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )
        diagonal = epi_loop_tensors.get("diag")
        if const_expr(diagonal is not None):
            value, coords, row_offset, col_offset, diagonal_tile = diagonal
            if diagonal_tile:
                for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                    coord = coords[i]
                    row = row_offset + coord[0]
                    col = col_offset + coord[1]
                    if row == col:
                        tRS_rD[i] += value
        return (tRS_rD,)


# Distinct class names keep QuACK's kernel cache specific to this epilogue.
class GemmSymmetricSubtileSm90(GemmSymmetricFusedMixin, GemmSm90):
    pass


class GemmSymmetricSubtileSm100(GemmSymmetricFusedMixin, GemmSm100):
    pass


class GemmSymmetricSubtileSm120(GemmSymmetricFusedMixin, GemmSm120):
    pass


@functools.lru_cache(maxsize=None)
def _resolve_config(capacity):
    """Return the GEMM class and tile configuration for a compute capability."""
    if capacity not in (9, 10, 11, 12):
        raise NotImplementedError(
            "fused symmetric GEMM requires SM90, SM100, SM110, or SM120"
        )
    gemm_class, kwargs = {
        9: (GemmSymmetricSubtileSm90, dict(tile_m=128, tile_n=256, pingpong=False,
                                         is_dynamic_persistent=False, cluster_m=2)),
        10: (GemmSymmetricSubtileSm100, dict(tile_m=256, tile_n=256, pingpong=False, cluster_m=2)),
        11: (GemmSymmetricSubtileSm100, dict(tile_m=256, tile_n=256, pingpong=False, cluster_m=2)),
        12: (GemmSymmetricSubtileSm120, dict(tile_m=128, tile_n=128, pingpong=True, cluster_m=1)),
    }[capacity]
    return gemm_class, GemmConfig(device_capacity=capacity, **kwargs)


_PLAN_CACHE = {}


def _tensor_key(tensor):
    """Return the dtype, shape, and strides that identify a cached plan."""
    if tensor is None:
        return None
    return tensor.dtype, tuple(tensor.shape), tensor.stride()


def _gemm_symmetric_fused_execute(A, B, out, C, alpha, beta, diag):
    """Run the fused kernel with a plan cached by device, layouts, and scalars."""
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("fused symmetric GEMM expects batched 3-D operands")
    if A.shape != B.shape or out.shape != (*A.shape[:-1], A.shape[-2]):
        raise ValueError("fused symmetric GEMM expects A/B (L, M, K) and out (L, M, M)")
    if A.dtype != torch.float16 or B.dtype != torch.float16 or out.dtype != torch.float16:
        raise TypeError("fused symmetric GEMM expects FP16 operands and output")
    if C is not None and C.shape != out.shape:
        raise ValueError("C must have the same shape as out")

    capacity = get_device_capacity(A.device)[0]
    gemm_class, config = _resolve_config(capacity)

    alpha = None if alpha == 1.0 else alpha
    beta = None if C is None or beta == 1.0 else beta
    diag = None if diag == 0.0 else diag

    epi_args = {"mAuxOut": out.mT}
    for name, value in (
        ("alpha", alpha),
        ("beta", beta),
        ("diag", diag),
    ):
        if value is not None:
            epi_args[name] = value

    key = (
        A.device,
        capacity,
        _tensor_key(A),
        _tensor_key(B),
        _tensor_key(out),
        _tensor_key(C),
        alpha is not None,
        beta is not None,
        diag is not None,
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = epilogue_from_class(gemm_class).plan(
            A, B, out, C, epi_args=epi_args, config=config
        )
        _PLAN_CACHE[key] = plan
    plan.run(A, B, C, out={"D": out}, **epi_args)


@torch.library.custom_op(
    "polar_switch::gemm_symmetric_fused_out",
    mutates_args=("out",),
    device_types="cuda",
)
def gemm_symmetric_fused_out(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag: float = 0.0,
    accumulate: bool = False,
) -> None:
    """Run the fused kernel as a custom operator inside torch.compile."""
    _gemm_symmetric_fused_execute(
        A, B, out, out if accumulate else C, alpha, beta, diag
    )


@gemm_symmetric_fused_out.register_fake
def gemm_symmetric_fused_out_fake(
    A,
    B,
    out,
    C=None,
    alpha=1.0,
    beta=1.0,
    diag=0.0,
    accumulate=False,
):
    """Describe the in-place operator to torch.compile."""
    return None


def gemm_symmetric_fused(
    A,
    B,
    C=None,
    alpha=1.0,
    beta=1.0,
    diag=0.0,
    out=None,
):
    """Compute alpha*(A @ B) + beta*C + diag*I with one FP16 store.

    A and B use QuACK's row/column layouts, and the product must be symmetric.
    If out is supplied without C, its current contents are the addend. The FP32
    epilogue applies alpha, the addend, and the diagonal shift before storing
    the FP16 result.
    """
    provided_out = out is not None
    if not provided_out:
        out = torch.empty(
            (*A.shape[:-2], A.shape[-2], B.shape[-2]),
            dtype=A.dtype,
            device=A.device,
        )
    accumulate = provided_out and C is None
    if out.numel() == 0:
        return out
    if torch.compiler.is_compiling():
        gemm_symmetric_fused_out(
            A,
            B,
            out,
            None if accumulate else C,
            alpha=alpha,
            beta=beta,
            diag=diag,
            accumulate=accumulate,
        )
    else:
        _gemm_symmetric_fused_execute(
            A, B, out, out if accumulate else C, alpha, beta, diag
        )
    return out


def gram(X, tall, out):
    """Write X @ X.T, or X.T @ X for tall X, into out."""
    return gemm_symmetric(X.mT, X, out=out) if tall else gemm_symmetric(X, X.mT, out=out)


def square(A, C, out, alpha, beta, diag):
    """Write alpha*A @ A + beta*C + diag*I into out; out may be C but not A."""
    if diag == 0.0 and C is not out:
        return gemm_symmetric(A, A, C=C, out=out, alpha=alpha, beta=beta)
    return gemm_symmetric_fused(A, A, C=None if C is out else C,
                              out=out, alpha=alpha, beta=beta, diag=diag)


def apply(X, B, tall, residual):
    """Return B @ X + residual*X, or X @ B + residual*X for tall X."""
    left, right = (X, B) if tall else (B, X)
    if residual == 0.0:
        return gemm(left, right, tuned=False)
    return gemm_add(left, right, C=X, beta=residual, tuned=False)
