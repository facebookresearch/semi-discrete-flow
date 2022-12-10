import torch
from torch.utils.data import DataLoader, ConcatDataset
from .dataset_text8 import Text8
from .dataset_enwik8 import EnWik8

dataset_choices = {"text8", "enwik8"}


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    if args.dataset == "text8":
        data = Text8(seq_len=256)
        data_shape = (1, 256)
        num_classes = 27
    elif args.dataset == "enwik8":
        data = EnWik8(seq_len=320)
        data_shape = (1, 320)
        num_classes = 256

    # Data Loader
    dataset_train = ConcatDataset([data.train, data.valid])
    train_loader = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        data.valid, batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        data.test, batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )
    return train_loader, valid_loader, test_loader, data_shape, num_classes
