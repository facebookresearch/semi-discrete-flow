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

os.makedirs("connect4", exist_ok=True)
os.makedirs("mushroom", exist_ok=True)
os.makedirs("nursery", exist_ok=True)
os.makedirs("uscensus90", exist_ok=True)
os.makedirs("poker", exist_ok=True)
os.makedirs("covtype", exist_ok=True)

print("Downloading datasets.")
wget.download(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z",
    out="connect4",
)
subprocess.call("uncompress -f connect4/connect-4.data.Z", shell=True)
wget.download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
    out="mushroom",
)
wget.download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data",
    out="nursery",
)
wget.download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt",
    out="uscensus90",
)
wget.download(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data",
    out="poker",
)
wget.download(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data",
    out="poker",
)
wget.download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
    out="covtype",
)
print("")

print("Preprocessing Connect4 dataset.")
np.random.seed(123)
with open("connect4/connect-4.data", "r") as f:
    lines = np.array(f.readlines())
total_size = len(lines)
train_size = int(np.floor(total_size * 0.8))
val_size = int(np.floor(total_size * 0.1))
test_size = int(total_size - train_size - val_size)
print(f"Splitting into {train_size} train, {val_size} val, and {test_size} test")
perm = np.random.permutation(total_size)
train_set = lines[perm[:train_size]]
val_set = lines[perm[train_size : train_size + val_size]]
test_set = lines[perm[train_size + val_size :]]
with open("connect4/train.data", "w") as f:
    f.writelines(train_set)
with open("connect4/val.data", "w") as f:
    f.writelines(val_set)
with open("connect4/test.data", "w") as f:
    f.writelines(test_set)

print("Preprocessing Mushroom dataset.")
np.random.seed(234)
with open("mushroom/agaricus-lepiota.data", "r") as f:
    lines = np.array(f.readlines())
total_size = len(lines)
train_size = int(np.floor(total_size * 0.8))
val_size = int(np.floor(total_size * 0.1))
test_size = int(total_size - train_size - val_size)
print(f"Splitting into {train_size} train, {val_size} val, and {test_size} test")
perm = np.random.permutation(total_size)
train_set = lines[perm[:train_size]]
val_set = lines[perm[train_size : train_size + val_size]]
test_set = lines[perm[train_size + val_size :]]
with open("mushroom/train.data", "w") as f:
    f.writelines(train_set)
with open("mushroom/val.data", "w") as f:
    f.writelines(val_set)
with open("mushroom/test.data", "w") as f:
    f.writelines(test_set)

print("Preprocessing Nursery dataset.")
np.random.seed(345)
with open("nursery/nursery.data", "r") as f:
    lines = np.array(f.readlines())
total_size = len(lines)
train_size = int(np.floor(total_size * 0.8))
val_size = int(np.floor(total_size * 0.1))
test_size = int(total_size - train_size - val_size)
print(f"Splitting into {train_size} train, {val_size} val, and {test_size} test")
perm = np.random.permutation(total_size)
train_set = lines[perm[:train_size]]
val_set = lines[perm[train_size : train_size + val_size]]
test_set = lines[perm[train_size + val_size :]]
with open("nursery/train.data", "w") as f:
    f.writelines(train_set)
with open("nursery/val.data", "w") as f:
    f.writelines(val_set)
with open("nursery/test.data", "w") as f:
    f.writelines(test_set)

print("Preprocessing USCensus90 dataset.")
np.random.seed(456)
with open("uscensus90/USCensus1990.data.txt", "r") as f:
    lines = np.array(f.readlines())
lines = lines[1:]  # skip the header
lines = np.array([",".join(line.split(",")[1:]) for line in lines])  # ignore the first column
total_size = len(lines)
train_size = int(np.floor(total_size * 0.9))
val_size = int(np.floor(total_size * 0.05))
test_size = int(total_size - train_size - val_size)
print(f"Splitting into {train_size} train, {val_size} val, and {test_size} test")
perm = np.random.permutation(total_size)
train_set = lines[perm[:train_size]]
val_set = lines[perm[train_size : train_size + val_size]]
test_set = lines[perm[train_size + val_size :]]
with open("uscensus90/train.data", "w") as f:
    f.writelines(train_set)
with open("uscensus90/val.data", "w") as f:
    f.writelines(val_set)
with open("uscensus90/test.data", "w") as f:
    f.writelines(test_set)

print("Preprocessing PokerHand dataset.")
np.random.seed(567)
with open("poker/poker-hand-testing.data", "r") as f:
    lines0 = np.array(f.readlines())
with open("poker/poker-hand-training-true.data", "r") as f:
    lines1 = np.array(f.readlines())
lines = np.concatenate([lines0, lines1])
total_size = len(lines)
train_size = int(np.floor(total_size * 0.8))
val_size = int(np.floor(total_size * 0.1))
test_size = int(total_size - train_size - val_size)
print(f"Splitting into {train_size} train, {val_size} val, and {test_size} test")
perm = np.random.permutation(total_size)
train_set = lines[perm[:train_size]]
val_set = lines[perm[train_size : train_size + val_size]]
test_set = lines[perm[train_size + val_size :]]
with open("poker/train.data", "w") as f:
    f.writelines(train_set)
with open("poker/val.data", "w") as f:
    f.writelines(val_set)
with open("poker/test.data", "w") as f:
    f.writelines(test_set)

print("Preprocessing CoverType dataset.")
np.random.seed(678)
data = np.genfromtxt("covtype/covtype.data", delimiter=",", dtype=int)
binned_data = [data[:, i] for i in range(data.shape[1])]
for i in range(10):
    bins = np.linspace(binned_data[i].min(), binned_data[i].max() + 1, 11)
    binned_data[i] = np.digitize(binned_data[i], bins)
binned_data = np.column_stack(binned_data)
np.savetxt("covtype/covtype_binned.data", binned_data.astype(int), fmt='%i', delimiter=",")
with open("covtype/covtype_binned.data", "r") as f:
    lines = np.array(f.readlines())
total_size = len(lines)
train_size = int(np.floor(total_size * 0.8))
val_size = int(np.floor(total_size * 0.1))
test_size = int(total_size - train_size - val_size)
print(f"Splitting into {train_size} train, {val_size} val, and {test_size} test")
perm = np.random.permutation(total_size)
train_set = lines[perm[:train_size]]
val_set = lines[perm[train_size : train_size + val_size]]
test_set = lines[perm[train_size + val_size :]]
with open("covtype/train.data", "w") as f:
    f.writelines(train_set)
with open("covtype/val.data", "w") as f:
    f.writelines(val_set)
with open("covtype/test.data", "w") as f:
    f.writelines(test_set)

print("Done.")