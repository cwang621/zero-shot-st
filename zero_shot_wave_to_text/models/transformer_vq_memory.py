#!/usr/bin/env python3

import math
from typing import Tuple

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.data.data_utils import compute_mask_indices, lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
)
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerModelBase,
    TransformerEncoderBase,
    TransformerEncoder,
    TransformerDecoder,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer import base_architecture as transformer_architecture
from fairseq.modules import GumbelVectorQuantizer, LayerNorm, MultiheadAttention, FairseqDropout
from fairseq.distributed import fsdp_wrap
from zero_shot_wave_to_text.modules.vq_memory import VQMemory


class SemanticEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, dictionary, embed_tokens)
        self.use_mlm = args.use_mlm

        self.vq_memory = VQMemory(
            args.memory_num,
            args.encoder_attention_heads,
            args.encoder_embed_dim,
            args.vq_vars,
            args.vq_temp,
            args.vq_groups,
            args.encoder_embed_dim,
            weight_proj_depth=args.weight_proj_depth,
            weight_proj_factor=args.weight_proj_factor
        )

        if self.use_mlm:
            self.output_dropout_module = FairseqDropout(
                p=args.dropout, module_name=self.__class__.__name__
            )
            self.output_projection = nn.Linear(
                args.encoder_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
            )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args)
        )

    def forward(self, src_tokens, src_lengths, **kwargs):
        is_text = not src_tokens.dtype.is_floating_point
        if is_text:
            feature = self.embed_tokens(src_tokens).transpose(0,1)
            input_lengths = src_lengths
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
        else:
            feature, input_lengths = src_tokens, src_lengths
            encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        w2v_out = {
            "w2v_out": feature,
            "input_lengths": input_lengths
        }

        x = self.embed_scale * feature
        if self.embed_positions is not None:
            positions = self.embed_positions(encoder_padding_mask).transpose(0,1)
            x = x + positions
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x) # T x B x C
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        x = x.transpose(0,1)
        has_pads = encoder_padding_mask.any()
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0,1)

        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask
            )

        if self.layers is not None:
            x = self.layer_norm(x)

        memory_info = {
            "memory_input": x,
            "memory_padding_mask": encoder_padding_mask
        }

        x, encoder_padding_mask, vq_info = self.vq_memory(x, encoder_padding_mask)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "w2v_out": w2v_out, # dict
            "vq_info": vq_info, # dict
            "memory_info": memory_info, #dict
        }

    def compute_output_logits(self, encoder_out):
        assert self.use_mlm, "MLM is not available"
        if isinstance(encoder_out, dict) and "memory_info" in encoder_out:
            encoder_state = encoder_out["memory_info"]["memory_input"]
        else:
            encoder_state = encoder_out
        logits = self.output_projection(self.output_dropout_module(encoder_state))

        return logits


@register_model("transformer_vq_memory")
class VQMemoryTransformerModel(TransformerModelBase):
    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )
        parser.add_argument(
            "--memory-num",
            type=int,
            help="number of shared memory module"
        )
        parser.add_argument(
            "--vq-vars",
            type=int,
            help="number of vq vars"
        )
        parser.add_argument(
            "--vq-groups",
            type=int,
            help="number of vq groups"
        )
        parser.add_argument(
            "--vq-temp",
            type=Tuple[float, float, float],
            help="can be tuple of 3 values (start, end, decay)"
        )
        parser.add_argument(
            "--weight-proj-depth",
            type=int,
            help="weight proj depth in codebook"
        )
        parser.add_argument(
            "--weight-proj-factor",
            type=int,
            help="weight proj factor in codebook"
        )
        parser.add_argument(
            "--use-mlm",
            action="store_true",
            help="use masked language model projection"
        )

    @classmethod
    def build_model(cls, args, task):
        base_archtecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if args.offload_activations:
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=args.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=args.min_params_to_wrap)

        return cls(args, encoder, decoder)


    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return SemanticEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=args.no_cross_attention)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(args, dictionary, embed_dim, path)


@register_model_architecture("transformer_vq_memory", "transformer_vq_memory")
def base_archtecture(args):
    args.memory_num = getattr(args, "memory_num", 64)
    args.vq_vars = getattr(args, "vq_vars", 50)
    args.vq_groups = getattr(args, "vq_groups", 32)
    args.vq_temp = getattr(args, "vq_temp", (2, 0.5, 0.999995))
    args.weight_proj_depth = getattr(args, "weight_proj_depth", 1)
    args.weight_proj_factor = getattr(args, "weight_proj_factor", 2)
    transformer_architecture(args)


@register_model_architecture("transformer_vq_memory", "transformer_vq_memory_s")
def vq_memory_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_archtecture(args)
