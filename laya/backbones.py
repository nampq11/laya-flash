"""Backbone abstraction: swap the encoder under the decision head without touching it.

`LayaBackbone` is the contract an encoder must satisfy to sit inside a DecisionModel:

    forward(input_ids, attention_mask) -> hidden_states            # [batch, seq, hidden_size]
    hidden_size                         -> int                     # width of those states
    prepare_tokenizer(tok)              -> None                    # optional special-token fixups

Two implementations ship:

* `as_backbone(encoder)` rebinds any HF `AutoModel` encoder onto the contract in place.
  The returned object *is* the encoder - same parameters, same state_dict keys - so
  every published checkpoint keeps loading with `strict=True`. A projection to a fixed
  `head_dim` is attached only when one is requested; without it the parameter set is
  exactly what it was before this module existed.
* `lfm2_backbone(...)` builds an LFM2 encoder (e.g. LiquidAI/LFM2.5-Encoder-230M,
  hidden_size 1024) patched for bidirectional attention. LFM2 ships as a causal
  decoder; the patches below are adapted from LiquidAI's Apache-2.0
  `modeling_lfm2_bidirectional.py` so no `trust_remote_code` execution is needed.

`cfg["head_dim"]` in a checkpoint's rl_agent_config.json selects the projection width:
the decision head then keeps a fixed input dimension across backbones. When it is
unset (all published checkpoints), the head is sized from the encoder's native width
and no projection module exists.

Caveat: installing the LFM2 patches flips transformers' *shared* lfm2 module to
bidirectional, so any causal LFM2 model in the same process (e.g. an LFM2 chat model)
would silently attend bidirectionally too. Upstream's remote-code file has the same
effect; if you serve causal LFM2 models, do it in another process.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "LayaBackbone",
    "as_backbone",
    "backbone_hidden_size",
    "hidden_states_of",
    "lfm2_backbone",
]


class LayaBackbone(nn.Module, ABC):
    """Contract for encoders usable as the DecisionModel backbone.

    Subclasses hold the encoder's own parameters directly (no wrapper level), so a
    backbone's state_dict keys are exactly the encoder's. `head_proj`, when attached
    by as_backbone/lfm2_backbone, is the only key the bare encoder does not have.
    """

    @property
    def hidden_size(self) -> int:
        """Width of the hidden states forward() returns (after any projection)."""
        proj = self._modules.get("head_proj")
        return proj.out_features if proj is not None else self.config.hidden_size

    def _apply_head_proj(self, h: torch.Tensor) -> torch.Tensor:
        proj = self._modules.get("head_proj")
        return proj(h) if proj is not None else h

    @abstractmethod
    def forward(self, input_ids=None, attention_mask=None) -> torch.Tensor:
        """Map token ids + mask to hidden states [batch, seq, hidden_size]."""

    def prepare_tokenizer(self, tok) -> None:
        """Fix up special tokens before build_sequence runs. Default: nothing to do."""


def backbone_hidden_size(encoder) -> int:
    """Output width of a backbone, tolerating raw HF encoders (tests, notebooks).

    Prefers a LayaBackbone's `hidden_size` because with a head projection the output
    width deliberately differs from encoder.config.hidden_size.
    """
    size = getattr(encoder, "hidden_size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        size = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("encoder exposes neither hidden_size nor config.hidden_size")
    return size


def hidden_states_of(output) -> torch.Tensor:
    """Hidden states from a backbone call: Tensor for LayaBackbone, ModelOutput for raw HF."""
    return output if torch.is_tensor(output) else output.last_hidden_state


def _attach_projection(encoder: nn.Module, head_dim: Optional[int]) -> None:
    """Attach head_proj when head_dim asks for a width the encoder does not natively produce."""
    if head_dim is not None and head_dim != encoder.config.hidden_size:
        encoder.head_proj = nn.Linear(encoder.config.hidden_size, head_dim)


# --------------------------------------------------------------------------------------
# Generic HF encoders (ModernBERT, mmBERT, BERT, ...)
# --------------------------------------------------------------------------------------

_BACKBONE_CLASSES: Dict[type, type] = {}


def _hf_backbone_class(base: type) -> type:
    """One mixed class per concrete encoder class, cached so type identity is stable."""
    cls = _BACKBONE_CLASSES.get(base)
    if cls is None:

        class _HFBackbone(base, LayaBackbone):
            # The concrete base comes first in the MRO so its __init__/forward win;
            # LayaBackbone only contributes the contract and prepare_tokenizer.
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                out = base.forward(self, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                return self._apply_head_proj(hidden_states_of(out))

        cls = _HFBackbone
        # Publish under a module-level name: pickle/torch.save of a whole model resolves
        # classes by module + qualname, and a function-local qualname cannot be found.
        cls.__name__ = cls.__qualname__ = "_Backbone_" + base.__name__
        globals()[cls.__name__] = cls
        _BACKBONE_CLASSES[base] = cls
    return cls


def as_backbone(encoder: nn.Module, head_dim: Optional[int] = None) -> LayaBackbone:
    """Rebind a HF encoder instance onto the LayaBackbone contract, in place.

    Wrapping by composition (self.inner = encoder) would rename every checkpoint key
    from `encoder.X` to `encoder.inner.X` and orphan all published weights, so instead
    the instance's class is swapped for a subclass of its own type mixed with
    LayaBackbone. Parameters, buffers and state_dict keys are untouched; only the
    forward return type (hidden-state tensor instead of ModelOutput) and the
    `hidden_size`/`prepare_tokenizer` surface change.

    `head_dim` different from the encoder's native width attaches a `head_proj`
    Linear that projects hidden states for a fixed-width decision head.
    """
    encoder.__class__ = _hf_backbone_class(type(encoder))
    _attach_projection(encoder, head_dim)
    return encoder


# --------------------------------------------------------------------------------------
# LFM2 (LiquidAI), e.g. LiquidAI/LFM2.5-Encoder-230M: hidden_size 1024, 14 layers
# --------------------------------------------------------------------------------------

_LFM2_MIN_TRANSFORMERS = "4.55"


def _import_lfm2():
    """transformers' native lfm2 module, with an actionable error when too old."""
    try:
        from transformers.models.lfm2 import modeling_lfm2 as module

        module.Lfm2Model, module.Lfm2Attention, module.Lfm2ShortConv  # attribute smoke test
        return module
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "The LFM2 backbone needs transformers >= %s (native lfm2 support), found an older "
            "install. Upgrade with: pip install 'laya[lfm2]'" % _LFM2_MIN_TRANSFORMERS
        ) from e


def _install_lfm2_patches(module) -> None:
    """Make transformers' causal LFM2 bidirectional. Adapted from LiquidAI's
    Apache-2.0 modeling_lfm2_bidirectional.py (LFM2.5-Encoder-230M repo).

    Patches the module globals rather than subclass overrides because Lfm2Model.forward
    resolves `create_causal_mask` and the short-conv forward at call time from the
    module/class namespaces. Side effect (the same one upstream's remote-code file
    has): every Lfm2Model in this process becomes bidirectional once patched. Laya
    never loads a causal LFM2 alongside, and ModernBERT/mmBERT are unaffected.
    """
    import torch.nn.functional as F

    if getattr(module, "_laya_bidirectional", False):
        return

    def _bidirectional_mask(
        config,
        input_embeds=None,
        attention_mask=None,
        cache_position=None,
        past_key_values=None,
        position_ids=None,
        **kwargs,
    ):
        # transformers has renamed the embeds kwarg across versions
        # (input_embeds <-> inputs_embeds); accept either to stay forward-compatible.
        if input_embeds is None:
            input_embeds = kwargs.get("inputs_embeds")
        if config._attn_implementation == "flash_attention_2":
            # FA2 only uses the 2D padding mask to unpad sequences; causality is
            # controlled by Lfm2Attention.is_causal (set to False at init).
            if attention_mask is not None and not attention_mask.all():
                return attention_mask
            return None

        device, dtype = input_embeds.device, input_embeds.dtype
        bsz, q_len = input_embeds.shape[:2]
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = past + q_len
        mask = torch.zeros((bsz, 1, q_len, kv_len), device=device, dtype=dtype)
        if attention_mask is not None:
            cur_len = attention_mask.size(-1)
            key_pad_flags = (attention_mask == 0).to(device=device, dtype=torch.float32)
            pad_vec = torch.zeros((bsz, kv_len), device=device, dtype=torch.float32)
            if cur_len > 0:
                pad_vec[:, past : past + cur_len] = key_pad_flags * -1e9
            mask = mask + pad_vec.to(dtype)[:, None, None, :]
        return mask

    def _noncausal_shortconv_forward(
        self, hidden_states, past_key_values=None, cache_position=None, attention_mask=None, **kwargs
    ):
        # The stock conv pads only on the left (causal); symmetric k//2 padding makes
        # every position see its neighbours on both sides, as an encoder must.
        x = module.apply_mask_to_padding_states(hidden_states, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x
        k = self.conv.weight.shape[-1]
        conv_out = F.conv1d(
            Bx,
            weight=self.conv.weight,
            bias=self.conv.bias,
            stride=1,
            padding=k // 2,
            dilation=1,
            groups=Bx.shape[1],
        )
        if conv_out.shape[-1] > Bx.shape[-1]:
            conv_out = conv_out[..., : Bx.shape[-1]]
        elif conv_out.shape[-1] < Bx.shape[-1]:
            conv_out = F.pad(conv_out, (0, Bx.shape[-1] - conv_out.shape[-1]))
        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        return self.out_proj(y)

    module.create_causal_mask = _bidirectional_mask
    module.Lfm2ShortConv.slow_forward = _noncausal_shortconv_forward
    module.Lfm2ShortConv.forward = lambda self, *args, **kwargs: self.slow_forward(*args, **kwargs)
    module._laya_bidirectional = True


_LFM2_BACKBONE_CLASS = None


def _lfm2_backbone_class() -> type:
    global _LFM2_BACKBONE_CLASS
    if _LFM2_BACKBONE_CLASS is None:
        module = _import_lfm2()

        class _LayaLfm2Backbone(module.Lfm2Model, LayaBackbone):
            """LFM2 patched for encoder-style use: bidirectional attention + non-causal conv."""

            # Published LFM2-encoder checkpoints (e.g. LFM2.5-Encoder-230M) nest the base
            # model under `lfm2.*` inside a MaskedLM wrapper; declaring the prefix lets
            # from_pretrained strip it, so the pretrained weights load without remote code.
            base_model_prefix = "lfm2"

            def __init__(self, config):
                _install_lfm2_patches(module)
                config.use_cache = False  # encoder passes never consume a cache
                super().__init__(config)
                for m in self.modules():
                    if isinstance(m, module.Lfm2Attention):
                        m.is_causal = False

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                out = super().forward(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
                return self._apply_head_proj(hidden_states_of(out))

            def prepare_tokenizer(self, tok) -> None:
                # LFM2 ships mask/pad tokens but no CLS/SEP. Reuse its document delimiters
                # so build_sequence's [CLS] ... [SEP] template keeps working; assigning
                # already-registered token strings reuses their ids (no vocab growth, so
                # the encoder's embedding matrix still matches).
                if getattr(tok, "cls_token", None) is None:
                    tok.cls_token = "<|startoftext|>"
                if getattr(tok, "sep_token", None) is None:
                    tok.sep_token = "<|endoftext|>"

        _LFM2_BACKBONE_CLASS = _LayaLfm2Backbone
        # Module-level name so whole-model pickle/torch.save can resolve the class.
        _LayaLfm2Backbone.__qualname__ = "LayaLfm2Backbone"
        globals()["LayaLfm2Backbone"] = _LayaLfm2Backbone
    return _LFM2_BACKBONE_CLASS


def lfm2_backbone(
    model_id_or_config: Union[str, object],
    head_dim: Optional[int] = None,
    **from_pretrained_kwargs,
) -> LayaBackbone:
    """Build an LFM2 backbone from a hub id, local path, or Lfm2Config.

    Keyword arguments (attn_implementation, torch_dtype, token, ...) pass through to
    from_pretrained when loading by id; when building from a config only
    attn_implementation applies (torch_dtype/token are download-time concerns).

    head_dim is applied after construction rather than passed through from_pretrained,
    so a future Lfm2Config field of the same name can never silently repurpose it.
    """
    cls = _lfm2_backbone_class()
    if isinstance(model_id_or_config, str):
        enc = cls.from_pretrained(model_id_or_config, **from_pretrained_kwargs)
    else:
        attn = from_pretrained_kwargs.pop("attn_implementation", None)
        if attn is not None:
            # The same hook from_pretrained uses; without it the from-config path
            # (every Agent checkpoint load) would silently run eager attention.
            model_id_or_config._attn_implementation = attn
        enc = cls(model_id_or_config)
    _attach_projection(enc, head_dim)
    return enc
