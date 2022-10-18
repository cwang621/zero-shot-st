#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple
import os
import os.path as op

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
import fairseq.tasks as tasks
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.wav2vec import (
    Wav2Vec2Model,
    Wav2VecCtc,
    Wav2Vec2Config,
    Wav2Vec2CtcConfig,
)
from fairseq.models.transformer import (
    TransformerDecoder,
    Embedding,
    TransformerConfig,
    base_architecture as transformer_architecture,
)
from fairseq.models.speech_to_text import Conv1dSubsampler, TransformerDecoderScriptable
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from torch import Tensor

from .transformer_vq_memory import SemanticEncoder


logger = logging.getLogger(__name__)


@register_model("wave_transformer_vq_memory")
class WaveTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Transformer
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )
        # input
        parser.add_argument("--w2v2-model-path", type=str, metavar="N",
                            help="path to wav2vec model")
        parser.add_argument("--reset-w2v", action="store_true",
                            help="whether to train w2v from scratch")
        parser.add_argument("--use-asr-finetune-w2v", action="store_true",
                            help="if we want to load wav2vec2.0 asr finetuned data")
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # additional module
        parser.add_argument(
            "--use-ctc",
            action="store_true",
            help="add ctc projection after w2v encoder"
        )
        parser.add_argument(
            "--use-mlm",
            action="store_true",
            help="add mlm projection after transformer encoder"
        )
        # vq memory
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
        ###
        parser.add_argument(
            "--frozen-w2v-encoder",
            action="store_true",
            help="frozen acoustic encoder during training"
        )
        parser.add_argument(
            "--frozen-transformer-encoder",
            action="store_true",
            help="frozen semantic encoder during training"
        )
        parser.add_argument(
            "--frozen-encoder",
            action="store_true",
            help="frozen encoder during training"
        )
        parser.add_argument(
            "--frozen-decoder",
            action="store_true",
            help="frozen decoder during training"
        )
        parser.add_argument(
            "--frozen-w2v-updates",
            type=int,
            help="frozen acoustic encoder during first n updates"
        )
        pass

    @classmethod
    def build_encoder(cls, args, w2v_args, task, wav2vec_model, transformer_encoder, embed_tokens):
        return WaveTransformerEncoder(args, w2v_args, task, wav2vec_model, transformer_encoder, embed_tokens)

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoder(args, task.target_dictionary, embed_tokens, no_encoder_attn=args.no_cross_attention)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        if getattr(args, "share_all_embeddings", False):
            args.share_decoder_input_output_embed = True

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder_embed_tokens = decoder_embed_tokens
        if (not getattr(args, "share_all_embeddings", False)) and (task.src_dict is not None):
            encoder_embed_tokens = build_embedding(task.src_dict, args.encoder_embed_dim)

        wav2vec_model, w2v_args = WaveTransformerEncoder.build_wav2vec_model(args)
        if getattr(args, "frozen_w2v_encoder", False):
            logging.info("frozen w2v encoder")
            for p in wav2vec_model.parameters():
                p.requires_grad = False

        transformer_encoder = WaveTransformerEncoder.build_transformer_encoder(args, task.src_dict, encoder_embed_tokens)
        if getattr(args, "load_pretrained_mt_model_from", None):
            logger.info(
                f"loaded pretrained transformer encoder from: "
                f"{args.load_pretrained_mt_model_from}"
            )
            transformer_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=transformer_encoder, checkpoint=args.load_pretrained_mt_model_from
            )
        if getattr(args, "frozen_transformer_encoder", False):
            logging.info("frozen transformer encoder")
            for p in transformer_encoder.parameters():
                p.requires_grad = False
        encoder = cls.build_encoder(args, w2v_args, task, wav2vec_model, transformer_encoder, encoder_embed_tokens)
        if getattr(args, "frozen_encoder", False):
            logging.info("frozen encoder")
            for p in encoder.parameters():
                p.requires_grad = False

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "load_pretrained_mt_model_from", None):
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_mt_model_from}"
            )
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_mt_model_from
            )
        if getattr(args, "frozen_decoder", False):
            logging.info("frozen decoder")
            for p in decoder.parameters():
                p.requires_grad = False
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class WaveTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, w2v_args, task=None, wav2vec_model=None,
                 transformer_encoder=None, embed_tokens=None):
        super().__init__(None)

        self.args = args
        self.w2v_args = w2v_args
        self.num_updates = 0

        self.wav2vec_model = wav2vec_model

        self.use_ctc = args.use_ctc
        if self.use_ctc:
            self.ctc_projection = nn.Linear(args.encoder_embed_dim, len(task.src_dict), bias=False)
            nn.init.normal_(
                self.ctc_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
            )
            self.ctc_dropout_module = FairseqDropout(
                p=args.dropout, module_name=self.__class__.__name__
            )
            self.softmax = nn.Softmax(dim=-1)

        w2v_output_dim = self.w2v_args.encoder_embed_dim
        self.subsample = Conv1dSubsampler(
            w2v_output_dim,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.transformer_encoder = transformer_encoder

    @classmethod
    def build_wav2vec_model(cls, args):
        assert args.w2v2_model_path is not None
        assert op.isfile(args.w2v2_model_path)
        w2v2_model_path = args.w2v2_model_path
        use_asr_finetune_w2v = args.use_asr_finetune_w2v
        reset_w2v = args.reset_w2v

        ckpt = torch.load(w2v2_model_path)

        if not use_asr_finetune_w2v:
            w2v_args = ckpt["args"]
            wav2vec_model = Wav2Vec2Model.build_model(
                Wav2Vec2Config.from_namespace(ckpt["args"]), task=None
            )
            if not reset_w2v:
                wav2vec_model.load_state_dict(ckpt["model"])
            else:
                logger.info("not loading wave2vec pretrained weights")
        else:
            ckpt["args"].data = args.data
            if not op.exists(op.join(ckpt["args"].data, f"dict.{ckpt['args'].labels}.txt")):
                os.system(f"wget -P {ckpt['args'].data} https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt")

            w2v_task = tasks.setup_task(ckpt["args"])
            model_finetuned = Wav2VecCtc.build_model(
                Wav2Vec2CtcConfig.from_namespace(ckpt["args"]), task=w2v_task
            )
            model_finetuned.load_state_dict(ckpt["model"])
            wav2vec_model = model_finetuned.w2v_encoder.w2v_model
            w2v_args = ckpt["args"].w2v_args["model"]
        return wav2vec_model, w2v_args

    @classmethod
    def build_transformer_encoder(cls, args, dictionary, embed_tokens):
        return SemanticEncoder(args, dictionary, embed_tokens) if args.encoder_layers > 0 else None

    def _get_w2v_feature(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        res = self.wav2vec_model.extract_features(src_tokens, padding_mask)
        w2v_feature = res["x"]
        output_length = self.wav2vec_model._get_feat_extract_output_lengths(src_lengths)
        padding_mask = lengths_to_padding_mask(output_length)
        return w2v_feature, padding_mask, output_length

    def forward(self, src_tokens, src_lengths, **extra_args):
        is_text = not src_tokens.dtype.is_floating_point
        if is_text:
            feature = src_tokens
            input_lengths = src_lengths
        else:
            if self.num_updates <= getattr(self.args, "frozen_w2v_updates", -1):
                with torch.no_grad():
                    feature, _, input_lengths = self._get_w2v_feature(src_tokens, src_lengths)
            else:
                feature, _, input_lengths = self._get_w2v_feature(src_tokens, src_lengths)
            feature, input_lengths = self.subsample(feature, input_lengths)

        if self.transformer_encoder is None:
            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            w2v_out = {
                "w2v_out": feature,
                "input_lengths": input_lengths
            }
            memory_info = {
                "memory_input": feature,
                "memory_padding_mask": encoder_padding_mask
            }
            return {
                "encoder_out": [feature],  # T x B x C
                "encoder_padding_mask": [encoder_padding_mask],  # B x T
                "encoder_embedding": [],  # B x T x C
                "encoder_states": [],  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
                "w2v_out": w2v_out,  # dict
                "vq_info": None,  # dict
                "memory_info": memory_info,  # dict
            }

        encoder_out = self.transformer_encoder(feature, input_lengths)

        return encoder_out

    def compute_ctc_logits(self, encoder_out):
        if not self.use_ctc:
            return self.compute_output_logits(encoder_out)

        if isinstance(encoder_out, dict) and "memory_info" in encoder_out:
            encoder_state = encoder_out["memory_info"]["memory_input"]
        else:
            encoder_state = encoder_out
        ctc_logits = self.ctc_projection(self.ctc_dropout_module(encoder_state))

        return ctc_logits

    def compute_output_logits(self, encoder_out):
        return self.transformer_encoder.compute_output_logits(encoder_out)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def max_positions(self):
        return None

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)


@register_model_architecture(model_name="wave_transformer_vq_memory", arch_name="wave_transformer_vq_memory")
def base_architecture(args):
    # wav2vec2.0 feature-extractor
    args.w2v2_model_path = getattr(args, "w2v2_model_path", "./wav2vec_small.pt")
    args.reset_w2v = getattr(args, "reset_w2v", False)
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)

    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)

    # Additional modules
    args.use_ctc = getattr(args, "use_ctc", False)
    args.use_mlm = getattr(args, "use_mlm", False)

    # VQ Memory
    args.memory_num = getattr(args, "memory_num", 64)
    args.vq_vars = getattr(args, "vq_vars", 50)
    args.vq_groups = getattr(args, "vq_groups", 16)
    args.vq_temp = getattr(args, "vq_temp", (2, 0.5, 0.999995))
    args.weight_proj_depth = getattr(args, "weight_proj_depth", 2)
    args.weight_proj_factor = getattr(args, "weight_proj_factor", 2)

    ###
    args.frozen_acoustic_encoder_updates = getattr(args, "frozen_w2v_updates", -1)

    # Transformer
    transformer_architecture(args)


@register_model_architecture("wave_transformer_vq_memory", "wave_transformer_vq_memory_s")
def wave_transformer_vq_memory_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("wave_transformer_vq_memory", "wave_transformer_vq_memory_xs")
def wave_transformer_vq_memory_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    wave_transformer_vq_memory_s(args)


@register_model_architecture("wave_transformer_vq_memory", "wave_transformer_vq_memory_sp")
def wave_transformer_vq_memory_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    wave_transformer_vq_memory_s(args)


@register_model_architecture("wave_transformer_vq_memory", "wave_transformer_vq_memory_m")
def wave_transformer_vq_memory_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("wave_transformer_vq_memory", "wave_transformer_vq_memory_mp")
def wave_transformer_vq_memory_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    wave_transformer_vq_memory_m(args)


@register_model_architecture("wave_transformer_vq_memory", "wave_transformer_vq_memory_l")
def wave_transformer_vq_memory_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("wave_transformer_vq_memory", "wave_transformer_vq_memory_lp")
def wave_transformer_vq_memory_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    wave_transformer_vq_memory_l(args)
