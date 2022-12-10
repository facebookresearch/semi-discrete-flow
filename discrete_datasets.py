"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

from charlm.data.dataset_text8 import Text8Dataset
from charlm.data.dataset_enwik8 import EnWik8Dataset


class Text8(Text8Dataset):
    def __init__(self, root, data_dir="data/text8", split="train", raw_dicts=None):
        del raw_dicts  # not used for now.
        self.raw_dicts = None

        self.root = root
        self.split = split
        self.K = [27] * 256
        splits = {"train": "train", "val": "valid", "test": "test"}
        super().__init__(
            os.path.join(root, data_dir),
            seq_len=256,
            split=splits[split],
            download=(split == "train"),
        )

    def __getitem__(self, index):
        return (self.data[index][0],)


class EnWik8(EnWik8Dataset):
    def __init__(self, root, data_dir="data/text8", split="train", raw_dicts=None):
        del raw_dicts  # not used for now.
        self.raw_dicts = None

        self.root = root
        self.split = split
        self.K = [256] * 320
        splits = {"train": "train", "val": "valid", "test": "test"}
        super().__init__(
            os.path.join(root, data_dir),
            seq_len=320,
            split=splits[split],
            download=(split == "train"),
        )

    def __getitem__(self, index):
        return (self.data[index][0],)


class UCIDataset(TensorDataset):

    _VALID_SPLITS = set(["train", "val", "test"])
    nattrs = 0
    rm_first_col = False
    rm_last_col = False

    def __init__(self, root, data_dir, split, raw_dicts):
        assert split in self._VALID_SPLITS
        X, self.raw_dicts = _extract_categorical(
            os.path.join(root, data_dir, f"{split}.data"),
            nattrs=self.nattrs,
            raw_dicts=raw_dicts,
        )
        X, self.dicts = _remove_redundant(X, self.raw_dicts)
        if self.rm_first_col:
            X, self.dicts = X[:, 1:], self.dicts[1:]
        if self.rm_last_col:
            X, self.dicts = X[:, :-1], self.dicts[:-1]
        self.K = [len(self.dicts[j]) for j in range(len(self.dicts))]
        super().__init__(torch.as_tensor(X))

    def __str__(self):
        return f"{self.__class__.__name__}(num_examples={len(self)}, num_variables={len(self.K)}), K={self.K})"


class Mushroom(UCIDataset):
    """Various mushroom specifies, tested for edible or poisonous.

    Download url: https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data

    Notes:
     - First column is the label {e='edible', p='poisonous'}
    """

    nattrs = 23
    rm_first_col = True

    def __init__(
        self, root="./", data_dir="data/mushroom", split="train", raw_dicts=None,
    ):
        super().__init__(root, data_dir, split, raw_dicts)


class Nursery(UCIDataset):
    """Applications for nursery schools in Ljubljana, Slovenia in the 1980's.

    Download url: https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data

    Notes:
     - Last column is the label {'recommend', 'priority', 'not_recom', 'very_recom', 'spec_prior'}
    """

    nattrs = 9
    rm_last_col = True

    def __init__(
        self, root="./", data_dir="data/nursery", split="train", raw_dicts=None
    ):
        super().__init__(root, data_dir, split, raw_dicts)


class Connect4(UCIDataset):
    """Board states of Connect-4 games that resulted in either a win, loss, or draw.

    Download url: http://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z

    (Use `uncompress` command line.)

    Notes:
     - Last column is the label {'win', 'loss', 'draw'}
    """

    nattrs = 43
    rm_last_col = True

    def __init__(
        self, root="./", data_dir="data/connect4", split="train", raw_dicts=None
    ):
        super().__init__(root, data_dir, split, raw_dicts)


class USCensus90(UCIDataset):
    """

    Download url: https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt

    The first attribute is caseid and should be ignored

    """

    nattrs = 68

    def __init__(
        self, root="./", data_dir="data/uscensus90", split="train", raw_dicts=None
    ):
        super().__init__(root, data_dir, split, raw_dicts)


class PokerHand(UCIDataset):
    """Dataset of poker hands.

    Dataset url: http://archive.ics.uci.edu/ml/datasets/Poker+Hand

    Notes:
     - Last column is the label for type of hand.
    """

    nattrs = 11
    rm_last_col = True

    def __init__(self, root="./", data_dir="data/poker", split="train", raw_dicts=None):
        super().__init__(root, data_dir, split, raw_dicts)


class Forests(UCIDataset):
    """Dataset of forest cartographic data.

    Dataset url: https://archive.ics.uci.edu/ml/datasets/Covertype

    Notes:
     - Last column is the label for cover type.
    """

    nattrs = 55
    rm_last_col = True

    def __init__(
        self, root="./", data_dir="data/covtype", split="train", raw_dicts=None
    ):
        super().__init__(root, data_dir, split, raw_dicts)


def _extract_categorical(filename, nattrs, raw_dicts=None):
    dicts = [Dictionary() for _ in range(nattrs)] if raw_dicts is None else raw_dicts

    with open(filename, "r") as f:
        lines = f.readlines()

    processed = []
    for line in lines:
        line = line.strip().split(",")
        if len(line) != nattrs:
            continue
        asint = []
        for j in range(len(line)):
            asint.append(dicts[j].add_str(line[j], raw_dicts is not None))
        processed.append(asint)
    X = np.array(processed)

    return X, dicts


def _remove_redundant(X, dicts):
    # Remove features that only have one possible value.
    for i in range(len(dicts) - 1, -1, -1):
        if len(dicts[i]) < 2:
            X = np.concatenate([X[:, :i], X[:, i + 1 :]], axis=1)
            dicts = dicts[:i] + dicts[i + 1 :]
    return X, dicts


class Dictionary:
    def __init__(self):
        self.str2int = dict()
        self.int2str = []

    def __len__(self):
        return len(self.int2str)

    def add_str(self, s, assert_in_dict):
        if s not in self.str2int.keys():
            assert not assert_in_dict, f"key {s} not in dict"
            self.str2int[s] = len(self.int2str)
            self.int2str.append(s)
        return self.str2int[s]


def dicts_to_segment_ids(dicts):
    segment_ids = []
    for i, d in enumerate(dicts):
        segment_ids += [i] * len(d)
    return torch.tensor(segment_ids)


def _extract_itemset(filename, raw_dict=None):
    data_dict = Dictionary() if raw_dict is None else raw_dict
    data = np.loadtxt(filename, dtype=int, delimiter=",")
    for orig_id in np.unique(data):
        i = data_dict.add_str(orig_id, raw_dict is not None)
        data[data == orig_id] = i
    X = torch.tensor(data)
    return X, data_dict


class ItemsetDataset(TensorDataset):

    _VALID_SPLITS = set(["train", "val", "test"])

    def __init__(self, root, data_dir, split, raw_dict=None):
        assert split in self._VALID_SPLITS
        X, raw_dict = _extract_itemset(
            os.path.join(root, data_dir, f"{split}.data"), raw_dict=raw_dict,
        )
        self.raw_dict = raw_dict
        super().__init__(torch.as_tensor(X))

    def __str__(self):
        return f"{self.__class__.__name__}(num_examples={len(self)}, num_classes={len(self.dict)})"


class Retail(ItemsetDataset):

    num_items = 4

    def __init__(self, root="./", data_dir="data/retail", split="train", raw_dict=None):
        super().__init__(root, data_dir, split, raw_dict)


class Accidents(ItemsetDataset):

    num_items = 4

    def __init__(
        self, root="./", data_dir="data/accidents", split="train", raw_dict=None
    ):
        super().__init__(root, data_dir, split, raw_dict)
