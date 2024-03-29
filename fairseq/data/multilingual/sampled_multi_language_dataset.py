# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import hashlib
import logging
import time
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import List

import numpy as np
import torch
from fairseq import distributed_utils
from fairseq.data import FairseqDataset, data_utils
from fairseq.data import SampledMultiDataset


def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


logger = logging.getLogger(__name__)


class SampledMultiLangDataset(SampledMultiDataset):
    """Samples from multiple sub-datasets according to given sampling ratios.
    Args:
        datasets (
            List[~torch.utils.data.Dataset]
            or OrderedDict[str, ~torch.utils.data.Dataset]
        ): datasets
        sampling_ratios (List[float]): list of probability of each dataset to be sampled
            (default: None, which corresponds to concatenating all dataset together).
        seed (int): RNG seed to use (default: 2).
        epoch (int): starting epoch number (default: 1).
        eval_key (str, optional): a key used at evaluation time that causes
            this instance to pass-through batches from *datasets[eval_key]*.
        collate_format (CollateFormat):  collater output format, either CollateFormat.ordered_dict or
            CollateFormat.single (default: CollateFormat.single) where CollateFormat.single configures
            the collater to output batches of data mixed from all sub-datasets,
            and CollateFormat.ordered_dict configures the collater to output a dictionary of batches indexed by keys
            of sub-datasets.
            Note that not all sub-datasets will present in a single batch in both formats.
        virtual_size (int, or callable): the expected virtual size of the dataset (default: default_virtual_size_func).
        split (str): the split of the data, e.g. 'train', 'valid' or 'test'.
        shared_collater (bool): whether or not to all sub-datasets have the same collater.
        shuffle (bool): whether or not to shuffle data (default: True).
    """

    def ordered_indices(self):
        multi_lang_sizes = self.sizes
        multi_lang_sizes = [
            multi_lang_sizes[
                0 if i == 0 else self.cumulated_sizes[i - 1] : self.cumulated_sizes[i]
            ]
            for i in range(len(self.datasets))
        ]
        multi_lang_sort_indices = []
        
        for i, sizes in enumerate(multi_lang_sizes):
            if self.shuffle:
                indices = np.random.permutation(len(sizes))
            else:
                indices = np.arange(len(sizes))
            tgt_sizes = sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
            src_sizes = (
                sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes
            )

            if tgt_sizes is not None:
                sort_indices = indices[np.argsort(np.maximum(src_sizes[indices], tgt_sizes[indices]), kind="mergesort")]
            else:
                sort_indices = indices[np.argsort(src_sizes[indices], kind="mergesort")]
            
            multi_lang_sort_indices.append(sort_indices + (0 if i==0 else self.cumulated_sizes[i - 1]))
        multi_lang_sort_indices = np.concatenate(multi_lang_sort_indices)

        return multi_lang_sort_indices

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
        batch_for_clip=False, 
    ):
        if not batch_for_clip:
            multi_lang_sort_indices = [
                indices[
                    0 if i == 0 else self.cumulated_sizes[i - 1] : self.cumulated_sizes[i]
                ]
                for i in range(len(self.datasets))
            ]
            batches = []
            for single_lang_sort_indices in multi_lang_sort_indices:
                batches += super().batch_by_size(
                    single_lang_sort_indices,
                    max_tokens,
                    max_sentences,
                    required_batch_size_multiple
                )
        else:
            smp_size = self.cfg[0].distributed_training.distributed_world_size * self.cfg[0].optimization.update_freq[0]
            chunk_size = smp_size * self.cfg[0].dataset.batch_size         

            multi_lang_sort_indices = []
            for i in range(len(self.datasets)):
                single_lang_index = indices[
                    0 if i == 0 else self.cumulated_sizes[i - 1] : self.cumulated_sizes[i]
                ]
                cshare = len(single_lang_index) // chunk_size
                crest = len(single_lang_index) % chunk_size

                chunked_indices = np.split(single_lang_index[crest:], cshare) 

                multi_lang_sort_indices = multi_lang_sort_indices + chunked_indices
            
            multi_lang_sort_indices = np.array(multi_lang_sort_indices)
            np.random.default_rng().shuffle(multi_lang_sort_indices)
            batches = []
            for item in multi_lang_sort_indices:
                for batch in np.split(item, smp_size):
                    batches.append(batch)
        
        return batches 


    def collater(self, samples, **extra_args):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return None
        
        if self.collate_format == "ordered_dict":
            collect_samples = [[] for _ in range(len(self.datasets))]
            for (i, sample) in samples:
                collect_samples[i].append(sample)
            batch = OrderedDict(
                [
                    (self.keys[i], dataset.collater(collect_samples[i]))
                    for i, (key, dataset) in enumerate(zip(self.keys, self.datasets))
                    if len(collect_samples[i]) > 0
                ]
            )
        elif self.shared_collater:
            batch = self.datasets[0].collater([s for _, s in samples])
        else:
            samples_dict = defaultdict(list)
            pad_to_length = (
                defaultdict(int)
                if "pad_to_length" not in extra_args
                else extra_args["pad_to_length"]
            )

            for ds_idx, s in samples:
                pad_to_length["source"] = max(
                    pad_to_length["source"], s["source"].size(0)
                )
                if s["target"] is not None:
                    pad_to_length["target"] = max(
                        pad_to_length["target"], s["target"].size(0)
                    )
                samples_dict[ds_idx].append(s)
            batches = [
                self.datasets[i].collater(samples_dict[i], pad_to_length=pad_to_length)
                for i in range(len(self.datasets))
                if len(samples_dict[i]) > 0
            ]

            def straight_data(tensors):
                batch = torch.cat(tensors, dim=0)
                return batch

            src_lengths = straight_data([b["net_input"]["src_lengths"] for b in batches])
            src_lengths, sort_order = src_lengths.sort(descending=True)

            def straight_order(tensors):
                batch = straight_data(tensors)
                return batch.index_select(0, sort_order)
            
            try:
                batch = {
                    "id": straight_order([b["id"] for b in batches]),
                    "nsentences": sum(b["nsentences"] for b in batches),
                    "ntokens": sum(b["ntokens"] for b in batches),
                    "net_input": {
                        "src_tokens": straight_order(
                            [b["net_input"]["src_tokens"] for b in batches]
                        ) if "src_tokens" in batches[0]['net_input'] and batches[0]["net_input"]["src_tokens"] is not None  else None,
                        "src_lengths": src_lengths,
                        "features": straight_order([b["net_input"]["features"] for b in batches]) if "features" in batches[0]['net_input'] and batches[0]["net_input"]["features"] is not None else None,
                    },
                    "target": straight_order([b["target"] for b in batches]) if batches[0]["target"] is not None else None,
                    "image_ids": straight_order([b["image_ids"] for b in batches]) if  "image_ids" in batches[0] and batches[0]["image_ids"] is not None else None,
                }
            except:
                print(f"error for sampled_multi_lang.py: L190, batch length is {len(samples)} and {len(batches)}. return null batch")
                batch = {'id': []}
                return batch

            def check_alignment(alignment, src_len, tgt_len):
                if alignment is None or len(alignment) == 0:
                    return False
                if (
                        alignment[:, 0].max().item() >= src_len - 1
                        or alignment[:, 1].max().item() >= tgt_len - 1
                ):
                    logger.warning("alignment size mismatch found, skipping alignment!")
                    return False
                return True

            if "prev_output_tokens" in batches[0]["net_input"]:
                batch["net_input"]["prev_output_tokens"] = straight_order(
                    [b["net_input"]["prev_output_tokens"] for b in batches]
                )
            
            if "src_lang_id" in batches[0]["net_input"]:
                batch["net_input"]["src_lang_id"] = straight_order(
                    [b["net_input"]["src_lang_id"] for b in batches]
                )

            if "tgt_lang_id" in batches[0]["net_input"]:
                batch["net_input"]["tgt_lang_id"] = straight_order(
                    [b["net_input"]["tgt_lang_id"] for b in batches]
                )

        return batch