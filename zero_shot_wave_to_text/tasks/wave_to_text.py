# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace

from fairseq.data import Dictionary, encoders
from fairseq.tasks import LegacyFairseqTask, register_task
from zero_shot_wave_to_text.data.wave_to_text_dataset import (
    W2TDataConfig,
    WaveToTextDataset,
    WaveToTextDatasetCreator,
    get_features_or_waveform
)

logger = logging.getLogger(__name__)

@register_task("wave_to_text")
class WaveToTextTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions-audio",
            type=int,
            default=1000000,
            help="max source positions for audio input"
        )
        parser.add_argument(
            "--generate",
            action="store_true",
            help="when generate, the source dictionary will be None"
        )


    def __init__(self, args, tgt_dict, src_dict=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.generate = args.generate
        self.data_cfg = W2TDataConfig(op.join(args.data, args.config_yaml))

        self.mask_symbol = "<mask>"
        self.blank_symbol = "<blank>"

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = W2TDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.tgt_vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        special_symbols = ["<mask>", "<blank>"]
        tgt_dict = Dictionary.load(dict_path)
        for symbol in special_symbols:
            tgt_dict.add_symbol(symbol)
        logger.info(
            f"target dictionary size ({data_cfg.tgt_vocab_filename}): " f"{len(tgt_dict):,}"
        )

        src_dict = None
        if getattr(data_cfg, "share_src_and_tgt", False):
            src_vocab_filename = data_cfg.tgt_vocab_filename
        else:
            src_vocab_filename = getattr(data_cfg, "src_vocab_filename", None)
        if src_vocab_filename is not None:
            dict_path = op.join(args.data, src_vocab_filename)
            if not op.isfile(dict_path):
                raise FileNotFoundError(f"Dict not found: {dict_path}")
            src_dict = Dictionary.load(dict_path)
            for symbol in special_symbols:
                src_dict.add_symbol(symbol)
            logger.info(
                f"source dictionary size ({src_vocab_filename}): " f"{len(src_dict):,}"
            )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict, src_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        tgt_bpe_tokenizer = self.build_bpe(self.args)
        if self.data_cfg.src_bpe_tokenizer is not None:
            src_bpe_tokenizer = self.build_src_bpe(self.args)
        else:
            src_bpe_tokenizer = tgt_bpe_tokenizer
            # if self.data_cfg.share_src_and_tgt:
            #     src_bpe_tokenizer = bpe_tokenizer
            # else:
            #     src_bpe_tokenizer = None
        self.datasets[split] = WaveToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            tgt_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            src_dict=self.src_dict,
            src_bpe_tokenizer=src_bpe_tokenizer
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None if self.generate else self.src_dict

    def max_positions(self):
        return self.args.max_source_positions_audio, self.args.max_target_positions

    def build_model(self, args):
        return super(WaveToTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        special_tokens = {
            self.tgt_dict.index("<mask>"),
            self.tgt_dict.index("<blank>")
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": special_tokens}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tgt tokenizer: {self.data_cfg.tgt_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.tgt_bpe_tokenizer))

    def build_src_bpe(self, args):
        logger.info(f"src tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return WaveToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
