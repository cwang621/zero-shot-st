# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils

logger = logging.getLogger(__name__)

def collate(
    samples,
    pad_idx,
    eos_idx,
    mask_index,
    vocab_list,
    mask_prob=0.15,
    left_pad_source=True,
    left_pad_target=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx if key!="mlm_target" else -100,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def mask_source(sample):
        source = sample["source"].clone().detach()
        mlm_target = source.clone().detach()
        if mask_prob > 0:
            rand = torch.rand(source.shape)
            masked_padding = (rand < mask_prob) * (source != eos_idx)
            rand = torch.rand(source.shape)
            do_mask_padding = masked_padding * (rand < 0.8)
            do_replace_padding = masked_padding * (rand > 0.9)

            if do_mask_padding.any():
                source[do_mask_padding] = mask_index
            if do_replace_padding.any():
                n = do_replace_padding.sum().item()
                value = torch.LongTensor(np.random.choice(vocab_list, n))
                source[do_replace_padding] = value
            mlm_target.masked_fill_(~masked_padding, -100)
        sample["masked_source"] = source
        sample["mlm_target"] = mlm_target
        return sample

    samples = [mask_source(sample) for sample in samples]
    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    masked_source = merge(
        "masked_source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    mlm_target = merge(
        "mlm_target",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )

    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    masked_source = masked_source.index_select(0, sort_order)
    mlm_target = mlm_target.index_select(0, sort_order)

    target = merge(
        "target",
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    target = target.index_select(0, sort_order)
    tgt_lengths = torch.LongTensor(
        [s["target"].ne(pad_idx).long().sum() for s in samples]
    ).index_select(0, sort_order)
    ntokens = tgt_lengths.sum().item()

    prev_output_tokens = merge(
        "target",
        left_pad=left_pad_target,
        move_eos_to_beginning=True,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths,},
        "target": target,
        "masked_source": masked_source,
        "mlm_target": mlm_target,
    }
    batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
        0, sort_order
    )


    return batch

class MLMLangPairDataset(FairseqDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that transform bos on
    collated samples of language pair dataset.

    Note that the transformation is applied in :func:`collater`.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset that collates sample into
            LanguagePairDataset schema
        src_eos (int): original source end-of-sentence symbol index to be replaced
        new_src_eos (int, optional): new end-of-sentence symbol index to replace source eos symbol
        tgt_bos (int, optional): original target beginning-of-sentence symbol index to be replaced
        new_tgt_bos (int, optional): new beginning-of-sentence symbol index to replace at the
            beginning of 'prev_output_tokens'
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        mask_index: int,
        mask_prob: float = 0.15,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=False,
        left_pad_target=False,
        shuffle=True,
        pad_to_multiple=1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.eos = src_dict.eos()
        self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        self.mask_index = mask_index
        self.mask_prob = mask_prob
        special_symbol = [mask_index, src_dict.bos(), src_dict.eos(), src_dict.pad(), src_dict.unk()]
        vocab_list = [item for item in np.arange(len(src_dict)) if item not in special_symbol]
        self.vocab_list = np.array(vocab_list)

    def get_batch_shapes(self):
        return self.buckets

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        if len(samples) == 0:
            return {}
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            mask_index=self.mask_index,
            vocab_list=self.vocab_list,
            mask_prob=self.mask_prob,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
                enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
                filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
                on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
                getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes, self.tgt_sizes, indices, max_sizes,
        )
