"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch

from .imagefolder_dataset import ImageTestDataset
from .ttf_dataset import TTFTrainDataset, TTFValDataset
from .ttf_utils import get_filtered_chars, read_font, render
from torch.utils.data import DataLoader


def get_trn_loader(cfg, primals, decomposition, transform, use_ddp=False, **kwargs):
    dset = TTFTrainDataset(
        primals=primals,
        decomposition=decomposition,
        transform=transform,
        **cfg
    )
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        kwargs["shuffle"] = False
    else:
        sampler = None
    loader = DataLoader(dset, sampler=sampler, collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_val_loader(cfg, transform, **kwargs):
    char_filter = ["\u0e01", "\u0e02", "\u0e03", "\u0e04", "\u0e05", "\u0e06", "\u0e07", "\u0e08", "\u0e09", "\u0e0a", "\u0e0b", "\u0e0c", "\u0e0d", "\u0e0e", "\u0e0f", "\u0e10", "\u0e11", "\u0e12", "\u0e13", "\u0e14", "\u0e15", "\u0e16", "\u0e17", "\u0e18", "\u0e19"]
    dset = TTFValDataset(
        char_filter=char_filter,
        transform=transform,
        **cfg
    )
    loader = DataLoader(dset, collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_test_loader(cfg, transform, **kwargs):
    dset = ImageTestDataset(
        transform=transform,
        **cfg.dset.test,
    )
    loader = DataLoader(dset, batch_size=cfg.batch_size, collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


__all__ = ["get_trn_loader", "get_val_loader", "get_test_loader", "get_filtered_chars", "read_font", "render"]
