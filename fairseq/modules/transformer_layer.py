# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

import re
from dataclasses import dataclass, field, fields
from typing import List, Optional

from omegaconf import II

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.utils import safe_getattr, safe_hasattr

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

_NAME_PARSER = r"(decoder|encoder|quant_noise)_(.*)"


class EncDecBaseConfig(FairseqDataclass):
    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained embedding"}
    )
    embed_dim: Optional[int] = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for FFN"}
    )
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each block"}
    )
    learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings"}
    )
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    layerdrop: float = field(default=0, metadata={"help": "LayerDrop probability"})
    layers_to_keep: Optional[List[int]] = field(
        default=None, metadata={"help": "which layers to *keep* when pruning"}
    )



class DecoderConfig(EncDecBaseConfig):
    input_dim: int = II("model.decoder.embed_dim")
    output_dim: int = field(
        default=II("model.decoder.embed_dim"),
        metadata={
            "help": "decoder output dimension (extra linear layer if different from decoder embed dim)"
        },
    )

    def __post_init__(self):
        #  II doesn't work if we are just creating the object outside of hydra so fix that
        if self.input_dim == II("model.decoder.embed_dim"):
            self.input_dim = self.embed_dim
        if self.output_dim == II("model.decoder.embed_dim"):
            self.output_dim = self.embed_dim



class QuantNoiseConfig(FairseqDataclass):
    pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )


@dataclass
class TransformerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu",
        metadata={"help": "activation function to use"},
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )
    adaptive_input: bool = False
    encoder: EncDecBaseConfig = EncDecBaseConfig()
    # TODO should really be in the encoder config
    max_source_positions: int = field(
        default=DEFAULT_MAX_SOURCE_POSITIONS,
        metadata={"help": "Maximum input length supported by the encoder"},
    )
    decoder: DecoderConfig = DecoderConfig()
    # TODO should really be in the decoder config
    max_target_positions: int = field(
        default=DEFAULT_MAX_TARGET_POSITIONS,
        metadata={"help": "Maximum output length supported by the decoder"},
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={
            "help": "share encoder, decoder and output embeddings (requires shared dictionary and embed dim)"
        },
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if True, disables positional embeddings (outside self attention)"
        },
    )
    adaptive_softmax_cutoff: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0.0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."
        },
    )
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    no_cross_attention: bool = field(
        default=False, metadata={"help": "do not perform cross-attention"}
    )
    cross_self_attention: bool = field(
        default=False, metadata={"help": "perform cross+self-attention"}
    )
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )
    # DEPRECATED field, but some old checkpoints might have it
    char_inputs: bool = field(
        default=False, metadata={"help": "if set, model takes character ids as input"}
    )
    relu_dropout: float = 0.0
    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[int] = field(
        default=1,
        metadata={"help": "shuffle tokens between workers before computing assignment"},
    )

    export: bool = field(
        default=False,
        metadata={"help": "make the layernorm exportable with torchscript."},
    )

    # copied from transformer_lm but expected in transformer_decoder:
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )

    # We need to make this hierarchical dataclass like the flat namespace
    # __getattr__ and __setattr__ here allow backward compatibility
    # for subclasses of Transformer(Legacy) that depend on read/write on
    # the flat namespace.

    def __getattr__(self, name):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f"invalid argument {name}.")

    def __setattr__(self, name, value):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def _copy_keys(args, cls, prefix, seen):
        """
        copy the prefixed keys (decoder_embed_dim) to the DC fields: decoder.embed_dim
        """
        cfg = cls()
        for fld in fields(cls):
            # for all the fields in the DC, find the fields (e.g. embed_dim)
            # in the namespace with the prefix (e.g. decoder)
            # and set it on the dc.
            args_key = f"{prefix}_{fld.name}"
            if safe_hasattr(args, args_key):
                seen.add(args_key)
                setattr(cfg, fld.name, safe_getattr(args, args_key))
            if safe_hasattr(args, fld.name):
                seen.add(fld.name)
                setattr(cfg, fld.name, safe_getattr(args, fld.name))
        return cfg

    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args


class TransformerEncoderLayerBase(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool),
                -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


# backward compatible with the legacy argparse format
class TransformerEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, args):
        super().__init__(TransformerConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )


class TransformerDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.attn_ln = LayerNorm(self.embed_dim) if utils.safe_getattr(cfg, 'scale_attn', False) else None
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, 'scale_heads', False)
        self.c_attn = nn.Parameter(torch.ones((self.nh,)), requires_grad=True) if scale_heads else None

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.ffn_layernorm = LayerNorm(cfg.decoder.ffn_embed_dim) if utils.safe_getattr(cfg, 'scale_fc', False) else None
        self.w_resid = nn.Parameter(torch.ones(self.embed_dim, ), requires_grad=True) if utils.safe_getattr(cfg, 'scale_resids', False) else None

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class TransformerDecoderLayer(TransformerDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
        )
