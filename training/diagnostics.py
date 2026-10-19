# SPDX-License-Identifier: MIT
"""Estimate and log spectral norms of orthogonalized updates.

With MUON_LOG_SPECTRAL_NORMS=1, the trainers record for every orthogonalized
matrix a spectral norm estimate, the largest absolute entry, and the numbers of
NaN and infinite entries, before optimizer scaling. Records are compressed
JSONL files under the run directory.
"""

import gzip
import json
from pathlib import Path
import time
import warnings

import torch


def _step(a, q, previous, beta):
    """Advance independent Lanczos iterations on a.T @ a by one step without forming it."""
    z = (q @ a.mT) @ a
    alpha = (q * z).sum(-1)
    z = z - alpha.unsqueeze(-1) * q - beta.unsqueeze(-1) * previous
    # Local reorthogonalization suppresses roundoff at Lanczos breakdown.
    z = z - (z * q).sum(-1, keepdim=True) * q
    z = z - (z * previous).sum(-1, keepdim=True) * previous
    next_beta = z.norm(dim=-1)
    active = next_beta > 8 * torch.finfo(torch.float32).eps * alpha.abs()
    next_beta = torch.where(active, next_beta, 0.)
    next_q = z / next_beta.clamp_min(1e-30).unsqueeze(-1)
    return torch.where(active.unsqueeze(-1), next_q, 0.), alpha, next_beta


_compiled_step = torch.compile(_step, fullgraph=True, dynamic=True,
                               options={"triton.cudagraphs": False})


@torch.no_grad()
def spectral_norm(x, rtol=1e-3, probes=4, max_iter=96, compile=True):
    """Estimate spectral norms of a batch of finite matrices in FP32.

    Lanczos iterations from fixed random probes estimate the largest singular
    value without forming a Gram matrix. Each estimate is ||A v|| for a unit
    Ritz vector, so it is not a certified upper bound.

    Args:
        x: Real matrices with shape (batch, rows, columns).
        rtol: Relative tolerance for the Ritz residual and for the change
            between convergence checks.
        probes: Starting vectors per matrix; the largest estimate is returned.
        max_iter: Maximum number of iterations, at least 16. Convergence is
            checked every 16 iterations.
        compile: Compile the Lanczos step for CUDA tensors.

    Returns:
        The estimates and a dict with convergence flags, the number of
        iterations, Ritz residuals, relative changes, and FP64 retries.
    """
    if x.ndim != 3 or not x.is_floating_point() or min(x.shape) < 1:
        raise ValueError("Expected a nonempty batch of real floating matrices")
    if rtol <= 0 or probes < 1 or max_iter < 16:
        raise ValueError("Use positive tolerance/probes and at least 16 iterations")
    a = x.float()
    if a.shape[-2] < a.shape[-1]:
        a = a.mT
    scale = a.abs().amax(dim=(-2, -1))
    a = a / torch.where(scale > 0, scale, 1.)[:, None, None]
    generator = torch.Generator(device=x.device).manual_seed(1729)
    q = torch.randn(min(probes, a.shape[-1]), a.shape[-1], device=x.device,
                    dtype=torch.float32, generator=generator)
    q = (q / q.norm(dim=-1, keepdim=True)).expand(len(x), -1, -1).contiguous()
    previous = torch.zeros_like(q)
    beta = torch.zeros(q.shape[:-1], device=x.device)
    vectors, diagonal, off_diagonal = [], [], []
    last = torch.zeros(len(x), device=x.device)
    projection_fp64_retries = 0
    advance = _compiled_step if compile and x.is_cuda else _step
    for iteration in range(1, max_iter + 1):
        vectors.append(q)
        next_q, alpha, next_beta = advance(a, q, previous, beta)
        previous, q, beta = q, next_q, next_beta
        diagonal.append(alpha)
        off_diagonal.append(beta)
        if iteration % 16 and iteration != max_iter:
            continue
        tri = torch.diag_embed(torch.stack(diagonal, dim=-1))
        if iteration > 1:
            upper = torch.diag_embed(torch.stack(off_diagonal[:-1], dim=-1), offset=1)
            tri = tri + upper + upper.mT
        try:
            _, eigenvectors = torch.linalg.eigh(tri)
        except torch.linalg.LinAlgError:
            if not bool(torch.isfinite(tri).all()):
                raise
            warnings.warn("FP32 norm projection failed; retrying in FP64", RuntimeWarning)
            _, eigenvectors = torch.linalg.eigh(tri.double())
            eigenvectors = eigenvectors.float()
            projection_fp64_retries += 1
        basis = torch.stack(vectors, dim=-2)
        ritz = torch.einsum("bpkn,bpk->bpn", basis, eigenvectors[..., -1])
        ritz = ritz / ritz.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        image = ritz @ a.mT
        candidates = image.norm(dim=-1)
        estimate, best = candidates.max(dim=-1)
        residuals = ((image @ a) - candidates.square().unsqueeze(-1) * ritz).norm(dim=-1)
        residuals = residuals / candidates.square().clamp_min(1e-30)
        residual = residuals.gather(-1, best[:, None]).squeeze(-1)
        change = (estimate - last).abs() / estimate.clamp_min(1e-30)
        converged = (residual <= rtol) & (change <= rtol * .25)
        converged = converged | (scale == 0)
        last = estimate
        if bool(converged.all()):
            break
    assessed = torch.full_like(converged, iteration >= 32) | (scale == 0)
    reported_change = torch.where(assessed, change, torch.full_like(change, float("nan")))
    return estimate * scale, dict(converged=converged, convergence_assessed=assessed, iterations=iteration,
                                ritz_residual=residual, relative_change=reported_change,
                                projection_fp64_retries=projection_fp64_retries)


class NormLog:
    """Write per-matrix estimates and nonfinite counts to compressed JSONL.

    Each instance creates a new file in the directory path. The first record
    describes the estimator, and each later record describes one batch of
    orthogonalized matrices. Set step before each optimizer step.
    """

    def __init__(self, path, rank, method):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.method = method
        self.path = path / f"rank-{rank:04d}-{time.time_ns()}.jsonl.gz"
        self.stream = gzip.open(self.path, "xt", compresslevel=1)
        self.stream.write(json.dumps(dict(
            event="metadata", format="polar-output-norm-v2", method=method, rank=rank,
            estimator="Lanczos", probes=1, max_iter=64, rtol=0.001,
            estimate_only=True, stage="post_orthogonalization_before_optimizer_scaling",
            step_index="zero-based, excludes warmup")) + "\n")

    @torch.no_grad()
    def record(self, label, output, matrix_start=0, execution=None, matrix_names=None):
        """Record the norms of one batch of orthogonalized matrices.

        Matrices with nonfinite entries are counted but not estimated. An
        estimator failure is recorded instead of raised.
        """
        matrices = output.detach().reshape(-1, *output.shape[-2:])
        nan = torch.isnan(matrices).sum(dim=(-2, -1))
        inf = torch.isinf(matrices).sum(dim=(-2, -1))
        finite = (nan + inf) == 0
        count = len(matrices)
        norms = torch.full((count,), float("nan"), device=output.device)
        residual = torch.full_like(norms, float("nan"))
        change = torch.full_like(norms, float("nan"))
        converged = torch.zeros(count, dtype=torch.bool, device=output.device)
        assessed = torch.zeros_like(converged)
        info = dict(iterations=0, projection_fp64_retries=0)
        error = None
        if bool(finite.any()):
            try:
                estimates, info = spectral_norm(matrices[finite], probes=1, max_iter=64, rtol=0.001)
                norms[finite] = estimates
                residual[finite] = info["ritz_residual"]
                change[finite] = info["relative_change"]
                converged[finite] = info["converged"]
                assessed[finite] = info["convergence_assessed"]
            except torch.linalg.LinAlgError as exc:
                # Diagnostic nonconvergence does not replace the selected polar method.
                error = str(exc)
        values = torch.stack((norms, matrices.abs().amax(dim=(-2, -1)).float(),
                              residual, change)).cpu().tolist()
        values = [[v if -float("inf") < v < float("inf") else None for v in row]
                  for row in values]
        record = dict(event="orthogonalization", step=self.step, label=label,
                      matrix_start=matrix_start, matrix_names=matrix_names,
                      shape=list(matrices.shape), dtype=str(output.dtype),
                      requested=self.method, execution=execution,
                      spectral_norm=values[0], max_abs=values[1],
                      nan_count=nan.cpu().tolist(), inf_count=inf.cpu().tolist(),
                      converged=converged.cpu().tolist(),
                      convergence_assessed=assessed.cpu().tolist(),
                      ritz_residual=values[2], relative_change=values[3],
                      iterations=info["iterations"],
                      projection_fp64_retries=info["projection_fp64_retries"],
                      estimator_error=error)
        self.stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        if self.step % 32 == 31:
            self.stream.flush()

    def close(self):
        """Close the log file."""
        self.stream.close()
