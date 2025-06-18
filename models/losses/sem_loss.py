import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CE_loss_2D(pred, target, ignore_index=255):

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    loss_valid_mean = 0
    N, h, w = target.shape
    B, N, C, H, W = pred.shape
    target = target.view(B, N, h, w)
    target1 = nn.functional.interpolate(target, (H, W), mode='nearest')
    # softmax = nn.Softmax(dim=2)
    # pred = softmax(pred)
    pred = pred.view(B*N, C, H, W)
    target1 = target1.view(B*N, H, W)
    loss = criterion(pred, target1.long())
    loss_valid = loss[target1 != ignore_index]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean