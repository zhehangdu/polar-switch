"""Update hidden matrix weights with Muon.

Muon applies Nesterov momentum, approximates the polar factor of the result,
and scales the update according to the matrix dimensions. Embeddings, output
weights, and parameters that are not matrices are returned separately for
AdamW. The selected orthogonalizer determines the polar approximation.
"""

import math

import torch


def parameter_groups(model):
    """Separate hidden matrix weights from parameters updated by AdamW.

    Return Muon groups organized by layer role, followed by the AdamW parameter
    list. Embeddings, output weights, and parameters that are not matrices use AdamW.
    """
    matrices, scalars = {}, []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or any(s in name for s in ("embed_tokens", "lm_head")):
            scalars.append(parameter)
        else:
            role = name.split(".")[-2]
            matrices.setdefault(role, []).append(parameter)
    return [{"params": params, "role": role} for role, params in sorted(matrices.items())], scalars


class Muon(torch.optim.Optimizer):
    """Apply Nesterov Muon using a supplied polar approximation function.

    Hidden matrices of matching shape, dtype, and device are orthogonalized
    together as one batch. The "rms" scale passes the momentum directions to the
    orthogonalizer in BF16 and uses 0.2*sqrt(max(rows, columns)). Other scale
    values use sqrt(rows/columns). Dimensions refer to the original weight matrix.

    Args:
        groups: Muon groups from parameter_groups.
        orthogonalize: A callable from make_polynomial_orthogonalizer.
        lr, momentum, weight_decay: Learning rate, momentum, and decoupled
            weight decay.
        scale: "rms" or another value, as described above.
        norm_log: Optional diagnostics.NormLog that records every update.
        parameter_names: Optional names of the parameters, keyed by id.
    """

    def __init__(self, groups, orthogonalize, lr, momentum=0.95, weight_decay=0.0, scale="rms",
                 norm_log=None, parameter_names=None):
        super().__init__(groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, scale=scale))
        self.orthogonalize = orthogonalize
        self.norm_log = norm_log
        self.parameter_names = parameter_names or {}

    @torch.no_grad()
    def step(self):
        """Update parameters from their gradients using momentum and polar approximation."""
        for group in self.param_groups:
            buckets = {}
            for p in group["params"]:
                if p.grad is not None:
                    buckets.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
            for params in buckets.values():
                directions = []
                for p in params:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    directions.append(p.grad.add(buf, alpha=group["momentum"]))
                update = torch.stack(directions)
                # Pass BF16 directions to the orthogonalizer, as in the Gram Newton-Schulz reference Muon;
                # the polar iterations themselves run in FP16.
                if group["scale"] == "rms":
                    update = update.bfloat16()
                update = self.orthogonalize(update).clone()
                if self.norm_log is not None:
                    self.norm_log.record(group["role"], update,
                                         execution=self.orthogonalize.last_execution,
                                         matrix_names=[self.parameter_names.get(id(p)) for p in params])
                rows, cols = params[0].shape
                factor = 0.2 * math.sqrt(max(rows, cols)) if group["scale"] == "rms" else math.sqrt(rows / cols)
                for p, u in zip(params, update.unbind()):
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(u.to(p.dtype), alpha=-group["lr"] * factor)
