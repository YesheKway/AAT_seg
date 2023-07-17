#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 08:43:35 2021
@author: yeshe kway
"""
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F

def get_loss(config):
    if config['loss_function'] == 'multiclass_dice':
        return DiceLoss(n_classes=config['n_output_channels'])
    elif config['loss_function'] == 'BCEDiceLoss':
        return BCEDiceLoss(n_classes=config['n_output_channels'])
    else:
        print('ERROR: no loss fuction selected')
        sys.exit()

# def multiclass_dice(input, target, epsilon=1e-6, weight=None):
#     """
#     Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
#     Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
#     Args:
#          input (torch.Tensor): NxCxSpatial input tensor
#          target (torch.Tensor): NxCxSpatial target tensor
#          epsilon (float): prevents division by zero
#          weight (torch.Tensor): Cx1 tensor of weight per channel/class
#     """
#     # input and target shapes must match
#     assert input.size() == target.size(), "'input' and 'target' must have the same shape"

#     input = flatten(input)
#     target = flatten(target)
#     target = target.float()

#     # compute per channel Dice Coefficient
#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect

#     # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
#     denominator = (input * input).sum(-1) + (target * target).sum(-1)
#     dice = 2 * (intersect / denominator.clamp(min=epsilon))
#     # average Dice score across all channels/classes
#     return 1. - torch.mean(dice)



# class BCEDiceLoss(nn.Module):
#     """Linear combination of BCE and Dice losses"""

#     def __init__(self, alpha, beta):
#         super(BCEDiceLoss, self).__init__()
#         self.alpha = alpha
#         self.bce = nn.BCEWithLogitsLoss()
#         self.beta = beta
#         self.dice = multiclass_dice()

#     def forward(self, input, target):
#         return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)



    
# def dice_loss(input, target, epsilon=1e-6, weight=None):
    
#     # input and target shapes must match
#     assert input.size() == target.size() 

#     input = flatten(input)
#     target = flatten(target)
#     target = target.float()

#     # compute per channel Dice Coefficient
#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect

#     # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
#     denominator = (input * input).sum(-1) + (target * target).sum(-1)
#     dice = 2 * (intersect / denominator.clamp(min=epsilon))
#     # average Dice score across all channels/classes
#     return 1. - torch.mean(dice)


def dice_loss(inputs, targets, smooth=1):
    intersection = (inputs * targets).sum()                            
    dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
    return dice_loss    


class DiceLoss(nn.Module):    
    def __init__(self, n_classes=6, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.dice_loss = dice_loss
        self.n_classes = n_classes
        
        
    def forward(self, inputs, targets, smooth=1):
        # compute dice loss
        dice_l = 0.0
        # BCE = 0.0
        for class_ in range(self.n_classes):
            #flatten label and prediction tensors
            input_ = inputs[:, class_, :, : , :]
            target_ = targets[: , class_, :, : , :]
            dice_l += self.dice_loss(input_.view(-1), target_.view(-1))
            # BCE += F.binary_cross_entropy(input_, target_, reduction='mean')    
        return dice_l
        # Dice_BCE = BCE + (self.classes - dice_l)
        # return Dice_BCE


class BCEDiceLoss(nn.Module):    
    def __init__(self, n_classes=6, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = dice_loss
        self.n_classes = n_classes
        
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)
        # compute dice loss
        dice_l = 0.0
        BCE = 0.0
        for class_ in range(self.n_classes):
            # extract class tensor 
            input_ = inputs[:, class_, :, : , :]
            target_ = targets[: , class_, :, : , :]
            # flatten label and prediction tensors
            input_ = input_.contiguous().view(-1)
            target_ = target_.contiguous().view(-1)
            # compute dice loss 
            dice_l += self.dice_loss(input_, target_)
            # compute bce loss 
            BCE += F.binary_cross_entropy(input_, target_, reduction='mean')    
        Dice_BCE = BCE +  dice_l
        return Dice_BCE


# def main():
    
#     print('dd')
#     inputs = torch.ones(1, 7, 40, 320, 320)
#     targets = torch.ones(2, 7, 40, 320, 320)


#     class_1 = targets[: , 0, :, : , :]
#     class_1  = class_1.unsqueeze(0)
    
#     ins_before =  class_1.contiguous()
#     ins = ins_before.view(-1)
#     # ins_1 = class_1.reshape()
    
#     inputs = flatten(class_1)
#     # print(inputs.shape)
    
#     config = {'n_output_channels' : 7, 
#               'loss_function': 'multiclass_dice'}
    
#     # dl = get_loss(config)
#     # res = dl(inputs, targets)
#     dl =  BCEDiceLoss(7)
#     res = dl(inputs, targets)
    
#     print(res)
#     test = 10
    
# main()    


