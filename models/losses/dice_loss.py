import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, weight=1.0):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, predictions, targets, mask=None, ignore_label=-1, empty_label=-1):
        dice_per_class = torch.zeros(self.num_classes, device=predictions.device)
        pred = F.softmax(predictions, dim=1)
        for c in range(self.num_classes):
            if c == ignore_label or c == empty_label:
                continue
            target = (targets == c).float()
            prediction = pred[:, c]
            if mask != None:
                intersection = torch.sum(prediction * target *mask )
                union = torch.sum(prediction * mask) + torch.sum(target * mask)
            else : 
                intersection = torch.sum(prediction * target )
                union = torch.sum(prediction) + torch.sum(target)      
            dice_per_class[c] = (2.0 * intersection) / (union + 1e-8)

        # Calculate average Dice Loss over classes
        dice_loss = (1.0 - torch.mean(dice_per_class)) * self.weight

        return dice_loss