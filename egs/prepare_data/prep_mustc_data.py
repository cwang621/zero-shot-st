#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
import os
import os.path as op
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import torchaudio
from data_utils import (
    filter_manifest_df,
    gen_config_yaml,
    # load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def os_system(command):
    logger.info(command)
    return os.system(command)



class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str, processed: bool = False) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = op.join(root, f"en-{lang}", "data", split)
        suffix = ".processed" if processed else ""
        wav_root, txt_root = op.join(_root, "wav"), op.join(_root, f"txt{suffix}")
        assert op.isdir(_root) and op.isdir(wav_root) and op.isdir(txt_root)
        # Load audio segments
        try:
            import yaml
        except ImportError:
            pass
        logger.info(f"loading task yaml config: {split}.yaml")
        with open(op.join(txt_root, f"{split}.yaml")) as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            logger.info(f"loading text sources {split}.{_lang}")
            with open(op.join(txt_root, f"{split}.{_lang}")) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = op.join(wav_root, wav_filename)
            sample_rate = torchaudio.info(wav_path).sample_rate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{op.splitext(wav_filename)[0]}_{i}"
                self.data.append(
                    (
                        wav_path,
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[
            Tensor, int, str, str, str, str, str, int]:
        wav_path, offset, n_frames, sr,\
            src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = torchaudio.load(
            wav_path, frame_offset=offset, num_frames=n_frames)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id,\
            wav_path, offset

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    data_root = op.abspath(args.data_root)
    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]
    logger.info(f"MANIFEST_COLUMNS: {MANIFEST_COLUMNS}")
    if args.languages == "":
        languages = MUSTC.LANGUAGES
    else:
        languages = args.languages.split(",")

    for lang in languages:
        cur_root = op.join(data_root, f"en-{lang}")
        if not op.isdir(cur_root):
            logger.info(f"{cur_root} does not exist. Skipped.")
            continue

        # Generate TSV manifest
        logger.info("Generating manifest...")
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")

            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(data_root, lang, split, args.processed)
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id,\
                    wav_file, offset in tqdm(dataset):
                manifest["id"].append(utt_id)
                length = int(wav.size(1))
                manifest["audio"].append(f"{wav_file}:{offset}:{length}")
                manifest["n_frames"].append(length)
                manifest["src_text"].append(src_utt)
                manifest["tgt_text"].append(tgt_utt)
                manifest["speaker"].append(speaker_id)

            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split,
                                    min_n_frames=10000,
                                    max_n_frames=1000000)
            save_df_to_tsv(df, op.join(cur_root, f"{split}_wave_triple.tsv"))

        # Generate vocab
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = \
            f"spm_{args.vocab_type}{v_size_str}_wave_joint"

        # Generate config YAML
        config_yaml = f'config_wave.yaml'
        logger.info(f"generating config: {config_yaml}")
        gen_config_yaml(
            cur_root,
            spm_filename_prefix + ".model",
            yaml_filename=config_yaml,
            specaugment_policy="lb",
            use_audio_input=True,
            src_spm_filename=spm_filename_prefix+".model"
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--languages", type=str, default="")
    parser.add_argument("--processed", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
