#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 12:24:49 2021
@author: Yeshe Kway
"""

import numpy as np
from denseCRF import densecrf


def Unpacklabels(labeldata):   
    label_values = np.unique(labeldata)
    channel_format = list()
    for value in label_values:
        channel_format.append(labeldata == value)
    return np.stack(channel_format, -1)


def crf_2D(raw_volume, class_probability, unpack_labels=True):
    '''
    Parameters
    ----------
    raw_volume : TYPE
        DESCRIPTION.
    class_probability : TYPE
        DESCRIPTION.
    Returns
    -------
    TYPE
        DESCRIPTION. shape (z, w, h, c)
    '''
    w1    = 1.0  # weight of bilateral term  
    alpha = 10    # spatial std (panalize the pixel proximity)
    beta  = 13    # rgb std
    w2    = 3.0   # weight of spatial term
    gamma = 2     # spatial std
    # it    = 10.0   # iteration
    it = 1
    param = (w1, alpha, beta, w2, gamma, it)
    new_image = list()
    
    # cast type for crf method requirement 
    class_probability = class_probability.astype(np.float32)
    class_probability = np.moveaxis(class_probability, 0, -1)
    
    raw_volume = raw_volume.astype(np.uint8) 
    
    for i in range(raw_volume.shape[0]):
        raw_img = raw_volume[i]
        prob_img = class_probability[i]    
        Iq = raw_img
        prob = prob_img
        lab = densecrf(Iq, prob, param)    
        # print(np.max(lab))
        # print(np.min(lab))

        new_image.append(lab)
    new_image = np.stack(new_image, 0)
    if unpack_labels:
        return Unpacklabels(new_image)
    else:
        return new_image
    
    
