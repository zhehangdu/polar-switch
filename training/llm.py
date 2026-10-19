"""Build Qwen3 and Gemma 3 models with random initialization and fused kernels.

Uses the Qwen3 and Gemma 3 text architectures provided by Transformers. Model
dimensions come from the training configuration. Released model weights are
not loaded. configure_kernels uses the cuDNN fused attention kernel and
QuACK's fused MLP, and chunked_loss uses QuACK's chunked linear cross entropy.
"""

from types import MethodType

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, Qwen3Config, Qwen3ForCausalLM


def build_model(config):
    """Build a randomly initialized Qwen3 or Gemma 3 model from its configuration."""
    if config["profile"] == "qwen600m":
        model_class, config_class = Qwen3ForCausalLM, Qwen3Config
    else:
        model_class, config_class = Gemma3ForCausalLM, Gemma3TextConfig
    model_config = config_class(**config["model"])
    model_config._attn_implementation = "sdpa"
    return model_class(model_config)


def configure_kernels(model, profile):
    """Use cuDNN attention and QuACK's fused gated MLP in every layer.

    The gate, up, and down projections remain separate parameters, so Muon
    updates each of them as a matrix. Attention supports causal sequences of
    equal length without padding or a key-value cache, including Gemma's
    sliding window.
    """
    from quack.mlp import mlp_func
    from cudnn.sdpa.fwd.torch_op import sdpa
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    activation = "swiglu" if profile == "qwen600m" else "geglu"

    def mlp_forward(self, x):
        weight = torch.cat((self.gate_proj.weight, self.up_proj.weight))
        return mlp_func(x, weight, self.down_proj.weight, activation, concat_layout=True)

    for layer in model.model.layers:
        layer.mlp.forward = MethodType(mlp_forward, layer.mlp)

    def attention(module, q, k, v, mask, scaling=None, dropout=0.0, sliding_window=None, **kwargs):
        # Equal-length, unpadded causal sequences without KV caching.
        assert mask is None and dropout == 0.0 and module.is_causal
        # cuDNN counts the current token in window_left; retain Gemma's 512-token window.
        out = sdpa(q.to(v.dtype), k.to(v.dtype), v, scale=scaling, is_causal=True,
                   window_left=sliding_window or -1, return_lse=True)[0]
        return out.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register("polar_cudnn", attention)
    # Reuse HF's no-dense-mask rule; this does not invoke the FA2 package.
    ALL_MASK_ATTENTION_FUNCTIONS.register("polar_cudnn", ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"])
    model.set_attn_implementation("polar_cudnn")


def chunked_loss(model, x, y):
    """Compute mean next-token cross entropy without storing all logits.

    With gradients enabled, QuACK's chunked linear cross entropy is used.
    Otherwise, PyTorch computes the logits in chunks of 4096 tokens.
    """
    from quack.linear_cross_entropy import chunked_linear_cross_entropy

    hidden = model.model(x).last_hidden_state
    if not torch.is_grad_enabled():
        # QuACK 0.6.5's loss-only path is nonfinite for the Qwen vocabulary.
        hidden, target = hidden.reshape(-1, hidden.shape[-1]), y.reshape(-1)
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, target.numel(), 4096):
            logits = torch.nn.functional.linear(hidden[start:start + 4096], model.lm_head.weight)
            total += torch.nn.functional.cross_entropy(logits, target[start:start + 4096], reduction="sum")
        return total / target.numel()
    return chunked_linear_cross_entropy(hidden, model.lm_head.weight, y, chunk_size=4096, use_scaled_exp=False)
