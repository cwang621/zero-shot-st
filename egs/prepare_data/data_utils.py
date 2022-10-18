#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import os.path as op
import zipfile
from functools import reduce
from glob import glob
from multiprocessing import cpu_count
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sentencepiece as sp
from fairseq.data.audio.audio_utils import _get_kaldi_fbank, _get_torchaudio_fbank
from fairseq.data.audio.feature_transforms.utterance_cmvn import UtteranceCMVN
from tqdm import tqdm


UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


def gen_config_yaml(
    data_root,
    spm_filename,
    yaml_filename="config.yaml",
    specaugment_policy="lb",
    prepend_tgt_lang_tag=False,
    sampling_alpha=1.0,
    use_audio_input=False,
    src_spm_filename=None,
):
    data_root = op.abspath(data_root)
    writer = S2TDataConfigWriter(op.join(data_root, yaml_filename))
    writer.set_vocab_filename(spm_filename.replace(".model", ".txt"))
    writer.set_bpe_tokenizer(
        {
            "bpe": "sentencepiece",
            "sentencepiece_model": op.join(data_root, spm_filename),
        }
    )
    if src_spm_filename is not None:
        writer.set_src_vocab_filename(
            src_spm_filename.replace(".model", ".txt"))
        writer.set_src_bpe_tokenizer(
            {
                "bpe": "sentencepiece",
                "sentencepiece_model": op.join(data_root, src_spm_filename),
            }
        )
    if prepend_tgt_lang_tag:
        writer.set_prepend_tgt_lang_tag(True)
    writer.set_sampling_alpha(sampling_alpha)
    if not use_audio_input:
        specaugment_setters = {
            "lb": writer.set_specaugment_lb_policy,
            "ld": writer.set_specaugment_ld_policy,
            "sm": writer.set_specaugment_sm_policy,
            "ss": writer.set_specaugment_ss_policy,
        }
        assert specaugment_policy in specaugment_setters
        specaugment_setters[specaugment_policy]()
        writer.set_feature_transforms("_train", ["specaugment"])
    writer.set_use_audio_input(use_audio_input)
    writer.set_sample_rate()
    writer.flush()


def load_df_from_tsv(path: str):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )


def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def filter_manifest_df(
    df, is_train_split=False, extra_filters=None, min_n_frames=5, max_n_frames=3000
):
    filters = {
        "no speech": df["audio"] == "",
        f"short speech (<{min_n_frames} frames)": df["n_frames"] < min_n_frames,
        "empty sentence": df["tgt_text"] == "",
    }
    if is_train_split:
        filters[f"long speech (>{max_n_frames} frames)"] = df["n_frames"] > max_n_frames
    if extra_filters is not None:
        filters.update(extra_filters)
    invalid = reduce(lambda x, y: x | y, filters.values())
    valid = ~invalid
    print(
        "| "
        + ", ".join(f"{n}: {f.sum()}" for n, f in filters.items())
        + f", total {invalid.sum()} filtered, {valid.sum()} remained."
    )
    return df[valid]


class S2TDataConfigWriter(object):
    DEFAULT_VOCAB_FILENAME = "dict.txt"
    DEFAULT_INPUT_FEAT_PER_CHANNEL = 80
    DEFAULT_INPUT_CHANNELS = 1
    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for S2T data config")
        self.yaml = yaml
        self.yaml_path = yaml_path
        self.config = {}

    def flush(self):
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.config, f)

    def set_audio_root(self, audio_root=""):
        self.config["audio_root"] = audio_root

    def set_vocab_filename(self, vocab_filename="dict.txt"):
        self.config["tgt_vocab_filename"] = vocab_filename

    def set_src_vocab_filename(self, vocab_filename="dict.txt"):
        self.config["src_vocab_filename"] = vocab_filename

    def set_specaugment(
        self,
        time_wrap_w: int,
        freq_mask_n: int,
        freq_mask_f: int,
        time_mask_n: int,
        time_mask_t: int,
        time_mask_p: float,
    ):
        self.config["specaugment"] = {
            "time_wrap_W": time_wrap_w,
            "freq_mask_N": freq_mask_n,
            "freq_mask_F": freq_mask_f,
            "time_mask_N": time_mask_n,
            "time_mask_T": time_mask_t,
            "time_mask_p": time_mask_p,
        }

    def set_specaugment_lb_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=1,
            freq_mask_f=27,
            time_mask_n=1,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_ld_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_sm_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=15,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_specaugment_ss_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )


    def set_bpe_tokenizer(self, bpe_tokenizer: Dict[str, Any]):
        self.config["tgt_bpe_tokenizer"] = bpe_tokenizer

    def set_src_bpe_tokenizer(self, bpe_tokenizer: Dict[str, Any]):
        self.config["src_bpe_tokenizer"] = bpe_tokenizer

    def set_feature_transforms(self, split, transforms: List[str]):
        if "transforms" not in self.config:
            self.config["transforms"] = {}
        self.config["transforms"][split] = transforms

    def set_prepend_tgt_lang_tag(self, flag=True):
        self.config["prepend_tgt_lang_tag"] = flag

    def set_sampling_alpha(self, sampling_alpha=1.0):
        self.config["sampling_alpha"] = sampling_alpha

    def set_use_audio_input(self, use_audio_input=False):
        self.config["use_audio_input"] = use_audio_input

    def set_sample_rate(self, sample_rate=DEFAULT_SAMPLE_RATE):
        self.config["sample_rate"] = sample_rate
