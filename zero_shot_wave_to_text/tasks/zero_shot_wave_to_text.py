# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace
import torch

from fairseq.data import Dictionary, encoders
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.data.audio.multi_modality_dataset import(
    MultiModalityDataset,
    LangPairMaskDataset,
    ModalityDatasetItem,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from zero_shot_wave_to_text.data.wave_to_text_dataset import (
    W2TDataConfig,
    WaveToTextDataset,
    WaveToTextDatasetCreator,
    get_features_or_waveform
)
from .translation_mlm import load_langpair_dataset
from .wave_to_text import WaveToTextTask

logger = logging.getLogger(__name__)

@register_task("zero_shot_wave_to_text")
class ZeroShotWaveToTextTask(WaveToTextTask):

    @classmethod
    def add_args(cls, parser):
        super(ZeroShotWaveToTextTask, cls).add_args(parser)
        ###
        parser.add_argument(
            "--parallel-text-data",
            default="",
            help="path to parallel text data directory"
        )
        parser.add_argument(
            "--source-lang",
            type=str,
            help="source language"
        )
        parser.add_argument(
            "--target-lang",
            type=str,
            help="target language"
        )
        parser.add_argument(
            "--max-tokens-text",
            type=int,
            metavar="N",
            help="maximum tokens for encoder text input ",
        )
        parser.add_argument(
            "--batch-size-text",
            type=int,
            metavar="N",
            help="batch size for encoder text input ",
        )
        parser.add_argument(
            "--speech-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for speech dataset with transcripts ",
        )
        parser.add_argument(
            "--text-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for text set ",
        )
        parser.add_argument(
            "--update-mix-data",
            action="store_true",
            help="use mixed data in one update when update-freq > 1",
        )
        ###
        parser.add_argument(
            "--load-pretrained-mt-model-from",
            type=str,
            metavar="STR",
            help="model to take semantic encoder and decoder weights from (for initialization)",
        )

    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        assert self.tgt_dict.pad() == self.src_dict.pad()
        assert self.tgt_dict.eos() == self.src_dict.eos()


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        tgt_bpe_tokenizer = self.build_bpe(self.args)
        if self.data_cfg.src_bpe_tokenizer is not None:
            src_bpe_tokenizer = self.build_src_bpe(self.args)
        else:
            src_bpe_tokenizer = tgt_bpe_tokenizer

        speech_dataset = WaveToTextDatasetCreator.from_tsv(
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
        text_dataset = None
        if self.args.parallel_text_data != "" and is_train_split:
            text_dataset = load_langpair_dataset(
                self.args.parallel_text_data,
                "train",
                self.args.source_lang,
                self.src_dict,
                self.args.target_lang,
                self.tgt_dict,
                self.src_dict.index("<mask>"),
                0.15,
                combine=True,
                dataset_impl=None,
                upsample_primary=1,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                truncate_source=False
            )

        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "speech",
                    speech_dataset,
                    (self.args.max_source_positions_audio, self.args.max_target_positions),
                    self.args.max_tokens,
                    self.args.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.args.max_source_positions, self.args.max_target_positions),
                    self.args.max_tokens_text
                    if self.args.max_tokens_text is not None
                    else self.args.max_tokens,
                    self.args.batch_size_text
                    if self.args.batch_size_text is not None
                    else self.args.batch_size,
                )
            ]
            speech_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = speech_dataset

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        if not isinstance(dataset, MultiModalityDataset):
            return super(ZeroShotWaveToTextTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache,
            )

        mult_ratio = [self.args.speech_sample_ratio, self.args.text_sample_ratio]
        assert len(dataset.datasets) == 2

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1 if self.args.update_mix_data else max(self.args.update_freq),
            buffer_size=data_buffer_size,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter

