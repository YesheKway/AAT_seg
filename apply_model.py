#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 06:01:31 2021

@author: Yehe Kway
"""
from Utils import load_config, get_device, load_Nifti, load_model
from ImageProcessing import (disabmle_volume,
                             preprocess_X,
                             postprocess, 
                             removePadding, 
                             labels_to_axis)
from ResNetUnet import get_model
import nibabel as nib
from tqdm import tqdm
import numpy as np
import torch 
import os 

def get_Niftii_img_shape(niftii_file):
    shapee = np.moveaxis(np.squeeze(niftii_file.get_fdata()), -1, 0).shape
    return (shapee[0], shapee[2], shapee[1])  
    
def inference(model, volume, config, segmask=False):
    '''
    This functions perfromes preprocessing and applies the model to the 
    given volume 
    '''
    model_input_shape = config['patch_dim']
    z_dim = model_input_shape[0]
    # --- store original shape 
    original_shape = volume.shape 
    # --- check volume is multiple of model input z dimension 
    residual = volume.shape[0]-z_dim
    # --- extract patches 
    patches, residual = disabmle_volume(volume, z_dim)
    segmentations = []
    for patch in patches:
        # preprocessing 
        X_data = preprocess_X(volume = patch,
                              input_dim = model_input_shape,
                              normalization= config['input_normalization']) 
        # make prediction      
        X_data = np.expand_dims(X_data, axis=0)
        X_data = torch.tensor(X_data, dtype=torch.float)
        X_data = X_data.cuda()
        # self.model.eval()
        probs = model(X_data)
        probs = postprocess(probs)
        # print(np.unique(probs))
        segmentations.append(probs)
    
    if residual != 0:
        res_vol = segmentations[-1]
        res_vol = res_vol[-residual:, ...]
        segmentations.pop()
        segmentations.insert(len(segmentations), res_vol)
    # --- stack back to one volume 
    segmentations = np.concatenate(segmentations, 0)
    # --- remove zerro padding 
    seg = removePadding(segmentations, original_shape)
    if segmask != True:
        seg = labels_to_axis(seg)
    return seg    


import time 


def applyModel(dataPath, modelPath, model_name='new', segmask=True):
    # --- get device
    device =  get_device()
    # device = 'cpu'
    # --- load config
    config = load_config(modelPath)
    # --- load model 
    model = load_model(config, modelPath)
    model.eval()
    model.to(device)
    # --- loop through data 
    for folder in tqdm(os.listdir(dataPath)):
        # load input
        ni_path = os.path.join(dataPath, folder, 'Abdo_fat.nii.gz')
        if not os.path.isfile(ni_path):
            ni_path = os.path.join(dataPath, folder, 'Abdo_fat.nii')
        
        ni_obj = nib.load(ni_path)
        X_data = load_Nifti(ni_path)

        seg = inference(model, X_data, config, segmask=segmask)    
        end_time = time.time()
        
        
        # for i in range(seg.shape[0]):
        #     s = seg[i]
        #     s = transformTensorForIMG(s)
        #     name = str(i) + '_pred.nii'
        #     new_path = os.path.join(dataPath, folder, name)
        #     saveAsNII(s, ni_obj.header, ni_obj.affine, new_path)             
        seg = transformTensorForIMG(seg)
        new_path = os.path.join(dataPath, folder, model_name + '_pred.nii.gz')
        saveAsNII(seg, ni_obj.header, ni_obj.affine, new_path)    


def saveAsNII(arr, header, affine, path):        
    # create new img file 
    img = nib.Nifti1Image(arr, affine, header)        
    nib.save(img, path)    
    
    
def transformTensorForIMG(tensor):
    """
     transform tensor accordingly to save its into .img file 
    """
    tensor = np.rot90(tensor, axes=(-1, -2))
    tensor = np.moveaxis(tensor, 0, -1)
    return tensor    
    

def main():
    print('start')
    # -- local paths 
    model_path = "//home/kwaygo/Documents/Projects/Torch/3D_AbdoFat-segmentation-kfold-training-EntireCohort/Results/Main_Model/1_model_checkpoint"
    data_path = "//media/kwaygo/BackupPlus/Data/Extracted/SPRESTO/Preconception/TMP/test"    
    # -- server paths 
    # model_path = '/app/Code/PyTorch/3D_segmentation-kfold-training/TrainedModels/fiftyPercentChanceAug_includeDSAT_excludeOthers_dynamic_unet_3D_5x5_2/CheckPoints-5-Folds/1_model_checkpoint'
    # data_path = "/app/Data/IntraRetroPeritoneal/second_5fold_model1_test_set" 
    # data_path = '/app/Data/TMP/Spresto_Preconseption-Nifti-Automated-with_Manual_Correction - Final_complressed'
    model_name = 'ipatrpat'
    applyModel(data_path, model_path, model_name=model_name)
        
    
if __name__ == '__main__':
    main()    
