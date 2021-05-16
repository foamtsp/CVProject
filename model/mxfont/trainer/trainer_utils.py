"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linear_sum_assignment


def cyclize(loader):
    """ Cyclize loader """
    while True:
        for x in loader:
            yield x


def has_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True

    return False


def unflatten_B(t):
    """ Unflatten [B*3, ...] tensor to [B, 3, ...] tensor
    t is flattened tensor from component batch, which is [B, 3, ...] tensor
    """
    shape = t.shape
    return t.view(shape[0]//3, 3, *shape[1:])


def overwrite_weight(model, pre_weight):
    model_dict = model.state_dict()
    pre_weight = {k: v for k, v in pre_weight.items() if k in model_dict}

    model_dict.update(pre_weight)
    model.load_state_dict(model_dict)


def load_checkpoint(path, gen, disc, aux_clf, cfg, force_overwrite=False):
    ckpt = torch.load(path)
   
    if force_overwrite:
        overwrite_weight(gen, ckpt['generator'])
    else:
        gen.cpu().load_state_dict(ckpt['generator'])
        #g_optim.load_state_dict(ckpt['optimizer'])

    if disc is not None:
        if force_overwrite:
            overwrite_weight(disc, ckpt['discriminator'])
        else:
            disc.cpu().load_state_dict(ckpt['discriminator'])
            #d_optim.load_state_dict(ckpt['d_optimizer'])

    if aux_clf is not None:
        if force_overwrite:
            overwrite_weight(aux_clf, ckpt['aux_clf'])
        else:
            aux_clf.cpu().load_state_dict(ckpt['aux_clf'])
            #ac_optim.load_state_dict(ckpt['ac_optimizer'])
    gen.cuda()
    disc.cuda()
    aux_clf.cuda()
    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg.ac_lr, betas=cfg.adam_betas)
    st_epoch = ckpt['epoch']
    if force_overwrite:
        st_epoch = 0
    loss = ckpt['loss']
    del ckpt
    
    
    return st_epoch, loss, g_optim,d_optim,ac_optim


def binarize_labels(label_ids, n_labels):
    binary_labels = []
    for _lids in label_ids:
        _blabel = torch.eye(n_labels)[_lids].sum(0).bool()
        binary_labels.append(_blabel)
    binary_labels = torch.stack(binary_labels)

    return binary_labels


def expert_assign(prob_org):
    n_comp, n_exp = prob_org.shape
    neg_prob = -prob_org.T if n_comp < n_exp else -prob_org
    n_row, n_col = neg_prob.shape

    prob_in = neg_prob
    remain_rs = np.arange(n_row)
    selected_rs = []
    selected_cs = []

    while len(remain_rs):
        r_in, c_in = linear_sum_assignment(prob_in)
        r_org = remain_rs[r_in]
        selected_rs.append(r_org)
        selected_cs.append(c_in)
        remain_rs = np.delete(remain_rs, r_in)
        prob_in = neg_prob[remain_rs]

    cat_selected_rs = np.concatenate(selected_cs) if n_comp < n_exp else np.concatenate(selected_rs)
    cat_selected_cs = np.concatenate(selected_rs) if n_comp < n_exp else np.concatenate(selected_cs)

    cat_selected_rs = torch.LongTensor(cat_selected_rs).cuda()
    cat_selected_cs = torch.LongTensor(cat_selected_cs).cuda()

    return cat_selected_rs, cat_selected_cs
