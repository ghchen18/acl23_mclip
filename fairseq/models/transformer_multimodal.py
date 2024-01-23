# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import register_model, register_model_architecture
from fairseq.models import  FairseqEncoder,FairseqEncoderModel
from fairseq.models.transformer.transformer_config import TransformerConfig
# from fairseq.models.roberta.model import roberta_base_architecture, roberta_large_architecture
# from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules import transformer_layer
from fairseq import utils
from fairseq.data import encoders
from fairseq.utils import safe_getattr, safe_hasattr

import torch,os
import math 
from torch import Tensor

from torch import nn, einsum
from typing import Dict, List
from fairseq.distributed import fsdp_wrap
from fairseq.models.roberta import XLMRModel
import clip
import torch.nn.functional as F 

import logging 
logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class MultimodalTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens=None):
        self.args = args
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.src_dict = dictionary
        self.padding_idx = dictionary.pad()
        self.max_source_positions = args.max_source_positions
        self.xlmr_task = getattr(args, 'xlmr_task', 'fixall_clinear_xlayer')
        self.vit_type = getattr(args, 'ViT_type', 'clip')

        model_root_path = os.path.dirname(args.save_dir)  
        mpm_type = getattr(args, 'MPLM_type', 'xlmr')
        self.bpe = encoders.build_bpe(args)
        try:
            self.xlmr = XLMRModel.from_pretrained(f'{model_root_path}/{mpm_type}_base', checkpoint_file='mte_mclip.pt')  ## for mclip_plus model, change into 'mte_mclip_plus.pt'
        except:
            self.xlmr = XLMRModel.from_pretrained(f'{model_root_path}/{mpm_type}_base', checkpoint_file='model.pt')  # load vanilla xlmr instead
            logger.warning("Enhanced MTE checkpoint not found. User vanilla XLM-R checkpoint instead. The model weights will be updated with the pretrained mCLIP model if this happens during the inference.")
        
        for _, param in self.xlmr.named_parameters():
            param.requires_grad = False 

        if self.vit_type == 'swinT':
            import timm
            mtype = 'swin_base_patch4_window7_224_in22k'  
            self.clip = timm.create_model(f'{mtype}', pretrained=False, checkpoint_path=f'{model_root_path}/clip_base/{mtype}.pth')              
            image_hid_size = 1024
        else:
            self.clip, _ = clip.load(f'{model_root_path}/clip_base/ViT-B-32.pt', device="cpu")  
            image_hid_size = 512
        
        self.clip.eval()
        for _, param in self.clip.named_parameters():
            param.requires_grad = False 
        
        self.use_old_name = True 
        
        if self.use_old_name:
            self.dim_proj = Linear(image_hid_size, args.encoder_embed_dim, bias=False) 
            cfg = TransformerConfig.from_namespace(args) 
            enclayer = [self.build_encoder_layer(cfg) for i in range(args.encoder_layers)]
            self.encoder_adapter = nn.Sequential(*enclayer)
        else:
            self.c_projector = Linear(image_hid_size, args.encoder_embed_dim, bias=False) 
            cfg = TransformerConfig.from_namespace(args) 
            enclayer = [self.build_encoder_layer(cfg) for i in range(args.encoder_layers)]
            self.x_projector = nn.Sequential(*enclayer)


    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_image_clip(self, features, pool_type='cls'):
        if features.dim() == 4 :  ## for raw image input
            if features.dtype != torch.float16:
                features = features.half()

            with torch.no_grad():
                if 'clip' in self.vit_type:
                    features, _ = self.clip.encode_image(features) 
                else:
                    features = self.clip.forward_features(features) 
                    features = features.unsqueeze(dim = 1)

        bsz, srclen, _ = features.size()
        if self.use_old_name:
            features = self.dim_proj(features)
        else:
            features = self.c_projector(features)
        features = features.transpose(0, 1)  ## B x T x C -> T x B x C 
        cls_image_feature = features[0].contiguous() 

        return  features, cls_image_feature
    
    def forward_text_clip(self, src_tokens):
        bsz, srclen = src_tokens.size()
        raw_text = [self.bpe.decode(x.replace('<pad>','').strip()) for x in self.src_dict.string(src_tokens).split('\n')] 
        clip_text_tokens = clip.tokenize(raw_text, truncate = True).to(src_tokens.device)
        
        with torch.no_grad():
            clip_text_feature = self.clip.encode_text(clip_text_tokens)

        if self.use_old_name:
            clip_text_feature = self.dim_proj(clip_text_feature)
        else:
            clip_text_feature = self.c_projector(clip_text_feature)
        return clip_text_feature, clip_text_tokens 

    def forward_xlmr_text(self, hyp_src_tokens):
        with torch.no_grad():
            x = self.xlmr.extract_features(hyp_src_tokens, return_all_hiddens = False)

        x = x.transpose(0,1) ## B x T x C -> T x B x C
        encoder_padding_mask = hyp_src_tokens.eq(self.padding_idx) 

        if self.use_old_name:
            for projector in self.encoder_adapter:
                x = projector(x, encoder_padding_mask = encoder_padding_mask)
        else:
            for projector in self.x_projector:
                x = projector(x, encoder_padding_mask = encoder_padding_mask)
    
        cls_x = x[-1] 

        return x, cls_x


    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]

        if len(encoder_out["encoder_padding_mask"]) == 0 or encoder_out["encoder_padding_mask"][0] is None:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


@register_model("multimodal_encoder")
class MultiModelEncoderModel(FairseqEncoderModel):

    def __init__(self, encoder, args):
        super().__init__(encoder)
        self.args = args 

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--checkpoint-activations",
            action="store_true",
            help="checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute",
        )
        parser.add_argument(
            "--offload-activations",
            action="store_true",
            help="checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.",
        )
        parser.add_argument(
            "--min-params-to-wrap", type=int, default=DEFAULT_MIN_PARAMS_TO_WRAP, help="number of positional embeddings to learn"
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens=None):
        return MultimodalTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_model(cls, args, task):       
        src_dict = task.source_dictionary
        encoder = cls.build_encoder(args, src_dict)
        return cls(encoder, args)

    def forward(self, src_tokens, **extra_args):
        encoder_out = self.encoder.forward(src_tokens,  **extra_args)
        return  encoder_out 


def roberta_base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", False)
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )


def roberta_large_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    roberta_base_architecture(args)


@register_model_architecture("multimodal_encoder", "multimodal_encoder_base")
def multimodal_encoder_base(args):
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    roberta_base_architecture(args)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", args.encoder_normalize_before)


@register_model_architecture("multimodal_encoder", "multimodal_encoder_large")
def multimodal_encoder_large(args):
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    roberta_large_architecture(args)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", args.encoder_normalize_before)
