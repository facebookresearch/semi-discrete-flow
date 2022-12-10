"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import wget
import subprocess
import numpy as np
from collections import OrderedDict


def prune_itemset(fname, *, vocab_size, freq_threshold, subsample_more=False):
    with open(fname) as f:
        lines = f.readlines()

    rows = OrderedDict()
    for row_id, line in enumerate(lines):
        if row_id not in rows.keys():
            rows[row_id] = []
        for item_id in map(int, (line.rstrip().split(" "))):
            rows[row_id].append(item_id)

    # remove rows with insufficient number of items.
    for row_id in list(rows.keys()):
        if len(rows[row_id]) < vocab_size:
            del rows[row_id]

    # subsample each row to a fixed number of items.
    next_row_id = np.max(list(rows.keys())) + 1
    for row_id in list(rows.keys()):

        if subsample_more:

            if len(rows[row_id]) > vocab_size * 4:
                for _ in range(5):
                    subsample = list(
                        np.random.choice(rows[row_id], size=vocab_size, replace=False)
                    )
                    rows[next_row_id] = subsample
                    next_row_id += 1

            if len(rows[row_id]) > vocab_size * 2:
                for _ in range(4):
                    subsample = list(
                        np.random.choice(rows[row_id], size=vocab_size, replace=False)
                    )
                    rows[next_row_id] = subsample
                    next_row_id += 1

        if len(rows[row_id]) > vocab_size:
            subsample = list(
                np.random.choice(rows[row_id], size=vocab_size, replace=False)
            )
            rows[next_row_id] = subsample
            next_row_id += 1
            del rows[row_id]

    # create items list
    items = OrderedDict()
    for row_id in list(rows.keys()):
        for item_id in rows[row_id]:
            if item_id not in items.keys():
                items[item_id] = []
            items[item_id].append(row_id)

    # remove items that don't occur frequently.
    for item_id in list(items.keys()):
        if len(items[item_id]) < freq_threshold:
            for row_id in items[item_id]:
                if row_id in rows.keys():
                    del rows[row_id]
            del items[item_id]

    return rows, items


def split_and_save(dirname, rows, items):
    D = np.array(list(rows.values()), dtype=int)
    total_size = D.shape[0]
    train_size = int(np.floor(total_size * 0.8))
    val_size = int(np.floor(total_size * 0.1))
    test_size = int(total_size - train_size - val_size)
    print(f"Splitting into {train_size} train, {val_size} val, and {test_size} test")
    perm = np.random.permutation(total_size)
    train_set = D[perm[:train_size]]
    val_set = D[perm[train_size : train_size + val_size]]
    test_set = D[perm[train_size + val_size :]]
    assert len(np.unique(train_set)) == len(
        items
    ), "train set does not contain all items"
    np.savetxt(
        os.path.join(dirname, "train.data"),
        train_set.astype(int),
        fmt="%i",
        delimiter=",",
    )
    np.savetxt(
        os.path.join(dirname, "val.data"), val_set.astype(int), fmt="%i", delimiter=","
    )
    np.savetxt(
        os.path.join(dirname, "test.data"),
        test_set.astype(int),
        fmt="%i",
        delimiter=",",
    )


if __name__ == "__main__":
    os.makedirs("retail", exist_ok=True)
    os.makedirs("accidents", exist_ok=True)

    print("Downloading datasets.")
    wget.download(
        "http://fimi.uantwerpen.be/data/retail.dat", out="retail",
    )
    wget.download(
        "http://fimi.uantwerpen.be/data/accidents.dat", out="accidents",
    )

    np.random.seed(123)
    rows, items = prune_itemset(
        "retail/retail.dat", vocab_size=4, freq_threshold=300, subsample_more=True
    )
    print(len(rows), len(items))
    split_and_save("retail", rows, items)

    np.random.seed(234)
    rows, items = prune_itemset(
        "accidents/accidents.dat",
        vocab_size=4,
        freq_threshold=100,
        subsample_more=False,
    )
    print(len(rows), len(items))
    split_and_save("accidents", rows, items)
