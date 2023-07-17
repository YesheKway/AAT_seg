#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:09:20 2021

@author: Yeshe Kway
"""
import numpy as np
import torch 

# =============================================================================
#                     Normalization Methods 
# =============================================================================


def normalize(volume,  normalization_type='zeroone'):
    if normalization_type == 'zeromeanunitvariance':
        return normalize_zeroMeanUnitVariance(volume)
    if normalization_type == 'zeroone':
        return normalize_zero_one(volume)
    
def normalize_zero_one(array):        
    """
    normalizes image pixel values to the range 0-1
    """
    return (array - np.min(array))/(np.max(array)-np.min(array)) 

def normalize_zeroMeanUnitVariance(array):
    '''
    zero mean unit variance normalization
    '''
    return (array-array.mean())/array.std()    


# =============================================================================
#                    Volume Processing Methods 
# =============================================================================

# def unpack_label_data(labeldata, n_labels=7):   
#     '''
#     This function extracts labels from the intra and retroperitioneal 
#     segmentation mask of shaped (z, w, h) where labels have numbers from 0-7
#     Note: label 1 (SSAT) and 2 (DSAT) are combined to form SAT 
#     Output: will be in shape (n_labels, z, w, h)
#     '''
#     labels = list()
#     for  i in range(1, n_labels):
#         if i == 2:
#             tmp = labels[0] + labeldata==i
#             label_n = tmp>0
#             labels.pop(0)
#             labels.append(label_n)
#         else:             
#             label_n = labeldata==i            
#             labels.append(label_n)
#     # create a background label
#     bg = 0
#     for file in labels: 
#         bg += file
#     bg = bg == 0
#     labels.insert(0, bg)        
#     return np.stack(labels)


def pad_Volume(volume, new_shape, paddValue=0):
    '''
    Parameters
    ----------
    img : numpy
        DESCRIPTION.
    new_shape :
        DESCRIPTION. (new_hight, new_width).
    col : TYPE, optional
        DESCRIPTION. The default is 'constant'.
    Returns
    -------
    None.
    '''    
    input_shape = volume.shape
    delta_z = new_shape[0] - input_shape[0]
    delta_w = new_shape[1] - input_shape[1]
    delta_h = new_shape[2] - input_shape[2]
    z_top, z_bottom = delta_z//2, delta_z-(delta_z//2)
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    padded_tensor = np.pad(volume, ((z_top, z_bottom), (left,right), (top,bottom)), 'constant', constant_values=(paddValue))
    return padded_tensor


def pad_Volume_aug(volume, new_shape, paddValue=0):
    '''
    Parameters
    ----------
    img : numpy
        DESCRIPTION.
    new_shape :
        DESCRIPTION. (new_hight, new_width).
    col : TYPE, optional
        DESCRIPTION. The default is 'constant'.
    Returns
    -------
    None.
    '''    
    
    volume = np.squeeze(volume)
    input_shape = volume.shape
    delta_z = new_shape[0] - input_shape[0]
    delta_w = new_shape[1] - input_shape[1]
    delta_h = new_shape[2] - input_shape[2]
    z_top, z_bottom = delta_z//2, delta_z-(delta_z//2)
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    padded_tensor = np.pad(volume, ((z_top, z_bottom), (left,right), (top,bottom)), 'constant', constant_values=(paddValue))
    padded_tensor= torch.tensor(padded_tensor, dtype=torch.float)
    padded_tensor = torch.unsqueeze(torch.unsqueeze(padded_tensor, 0), 0)
    return padded_tensor



def pad_Volume_labels_aug(volume, new_shape, paddValue=0):
    '''
    Parameters
    ----------
    img : numpy
        DESCRIPTION.
    new_shape :
        DESCRIPTION. (new_hight, new_width).
    col : TYPE, optional
        DESCRIPTION. The default is 'constant'.
    Returns
    -------
    None.
    '''    
    volume = np.squeeze(volume)    
    count = volume.shape[0]
    labels = []
    for i in range(count):
        v = volume[i]  
        labels.append(pad_Volume_aug(v, new_shape, paddValue))
    final = torch.stack(labels)    
    final = torch.unsqueeze(torch.squeeze(torch.moveaxis(final, 0, 2)), 0)
    return final 


def preprocess_X_patch(X, normalization='zeroone'):
    X = X.cpu().numpy() 
    X_n = []
    for i in range(X.shape[0]):
       X_n.append(normalize(X[i], normalization_type=normalization))
    X_n = np.stack(X_n, 0)
    return torch.tensor(X_n, dtype=torch.float)

def preprocess_X(volume, input_dim=(40, 320,320), normalization='zeroone'):
    '''
    preprocess model input 
    '''
    # --- preprocessing 
    volume = normalize(volume, normalization)  # normalize input 
    volume = pad_Volume(volume, input_dim) # padd input if needed     
    # --- reshape for pytorch model 
    volume = np.expand_dims(volume, axis=0)  # add batch dimension [B, C, H, W]
    volume = volume.astype(np.float32)  # typecasting to float32
    return volume


def preprocess_y(y_volume, input_dim=(40, 320,320)):
    '''
    preprocess model output
    '''
    # padd if needed
    y_volume = pad_Volume(y_volume, input_dim)
    # stack labels to their own dimension 
    # eg. from (z, w, h) --> (n_labels, z, w, h)
    y_volume = unpack_label_data(y_volume) 
    return y_volume.astype(np.uint8)



def disabmle_volume(volume, z_dim):
    '''
    volume will be disambled into patched of network inputs
    note this works only on z dimension 
    '''
    patches = []
    # --- return volume is smaller or same size as model input 
    if volume.shape[0] <= z_dim:
        patches.append(volume) 
        return patches, 0
    # --- define how many patches need to be extracted 
    else: 
        n_patches = int(volume.shape[0] / z_dim)
        residual = volume.shape[0] % z_dim
        # extract n patches 
        start = 0
        for n in range(n_patches):
            start = n*z_dim
            end  = start + z_dim
            patches.append(volume[start:end])
        if residual !=0:
            patches.append(volume[-z_dim:])
    return patches, residual


def disabmle_volume_4dim(volume, z_dim):
    '''
    volume will be disambled into patched of network inputs
    note this works only on z dimension 
    '''
    patches = []
    # ------------------------------------------------------------------------- 
    # return whole volume when if it is smaller or same size as model input 
    if volume.shape[1] <= z_dim:
        patches.append(volume) 
        return patches, 0
    # --- define how many patches need to be extracted 
    else: 
        n_patches = int(volume.shape[1] / z_dim)
        residual = volume.shape[1] % z_dim
        # extract n patches 
        start = 0
        for n in range(n_patches):
            start = n*z_dim
            end  = start + z_dim
            patches.append(volume[:, start:end, :])
        if residual !=0:
            patches.append(volume[:, -z_dim:, :, :])
    return patches, residual



def disabmle_volume(volume, z_dim):
    '''
    volume will be disambled into patched of network inputs
    note this works only on z dimension 
    '''
    patches = []
    # --- return volume is smaller or same size as model input 
    if volume.shape[0] <= z_dim:
        patches.append(volume) 
        return patches, 0
    # --- define how many patches need to be extracted 
    else: 
        n_patches = int(volume.shape[0] / z_dim)
        residual = volume.shape[0] % z_dim
        # extract n patches 
        start = 0
        for n in range(n_patches):
            start = n*z_dim
            end  = start + z_dim
            patches.append(volume[start:end])
        if residual !=0:
            patches.append(volume[-z_dim:])
    return patches, residual


# =============================================================================
#                       Postprocessing 
# =============================================================================


# TODO

# def removePadding(self, volume, original_shape):
#     input_shape = volume.shape
#     c, w, h = input_shape
#     delta_w = input_shape[1] - original_shape[1]
#     delta_h = input_shape[2] - original_shape[2]
#     top, bottom = delta_h//2, delta_h-(delta_h//2)
#     left, right = delta_w//2, delta_w-(delta_w//2)
#     if left != 0:
#         volume = volume[:, left:(w-right), :]
#     if top != 0:
#         volume = volume[:, :, top:h-bottom]
#     return volume
    

# def removePadding_label(self, labels, original_shape):
#     resized = np.zeros((labels.shape[0], original_shape[1], original_shape[2], labels.shape[-1]))
#     for i in range(labels.shape[-1]):
#         resized[:, :, :, i] = self.removePadding(labels[:, :, :, i], original_shape)
#     return resized

def postprocess(volume):
    '''
    postprocess model input 
    '''    
    # volume = argMax_to_axis(volume)
    volume = torch.argmax(volume, dim=1)  # perform argmax to generate 1 channel
    volume = volume.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    volume = np.squeeze(volume)  # remove batch dim and channel dim -> [Z, H, W]
    return volume   



def argMax_to_axis(probs):
    n_classes = probs.shape[1]
    probs = torch.argmax(probs, dim=1)
    probs = probs.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    probs = np.squeeze(probs)  # remove batch dim and channel dim -> [Z, H, W]    
    segmentation = []
    #
    for i in range(n_classes):
        segmentation.append((probs == i).astype(int)[..., np.newaxis])
    x = np.concatenate(segmentation, axis=-1)  
    x = np.moveaxis(x, -1, 0)
    return x

def labels_to_axis(probs):
    n_classes = np.unique(probs)
    segmentation = []
    #
    for i in n_classes:
        segmentation.append((probs == i).astype(int)[..., np.newaxis])
    # # if fat class others was not leared by the model we have to add dummy 
    # if len(segmentation) != 7:
    #     s = segmentation[0].shape
    #     segmentation.append(np.zeros((s[0], s[1], s[2], 1)))
    x = np.concatenate(segmentation, axis=-1)  
    x = np.moveaxis(x, -1, 0)
    return x


def removePadding(volume, original_shape):
    '''
    removes padding to bring volume back to its original shape 
    '''
    input_shape = volume.shape
    z, w, h = input_shape
    delta_z = input_shape[0] - original_shape[0]
    delta_w = input_shape[1] - original_shape[1]
    delta_h = input_shape[2] - original_shape[2]
    top_z, bottom_z = delta_z//2, delta_z-(delta_z//2)
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    if top_z != 0:
        volume = volume[top_z:z-bottom_z, :, :]
    if left != 0:
        volume = volume[:, left:(w-right), :]
    if top != 0:
        volume = volume[:, :, top:h-bottom]
    return volume
    


def removePadding_labels(volume, original_shape):
    '''
    removes padding to bring volume back to its original shape 
    '''
    l = []
    for i in range(volume.shape[0]):
        l.append(removePadding(volume[0],original_shape)[np.newaxis, ...])
    labels = np.concatenate(l, 0)
    return labels 


def removePadding_labels(volume, original_shape):
    '''
    removes padding to bring volume back to its original shape 
    '''
    l = []
    for i in range(volume.shape[0]):
        l.append(removePadding(volume[0],original_shape)[np.newaxis, ...])
    labels = np.concatenate(l, 0)
    return labels 






import os 

def main():
    print('in')
    
    
    # segmentation = []
    # for i in range(3):
    #     iomg = np.ones((40, 320, 320))
    #     segmentation.append(iomg)
    # # if len(segmentation) != 7:
    # #     s = segmentation[0].shape
    # #     dummy = np.zeros(())
    # x = np.concatenate(segmentation, axis=0)  
    # x = np.moveaxis(x, -1, 0)
    # tets = 10
    
    
    # p = '/media/kwaygo/ymk_HDD1/Experiments/Spresto_Applications/SprestoSamples_DSAT_SSAT_ParaRetroPeritoneal/spresto_retroperitoneal_samples/Train_subjects/AllData_subj/SPRESTO_10168'
    # x_path = os.path.join(p, 'Abdo_fat.nii')
    # # get_X_data(x_path)
    
    # y_path = os.path.join(p, 'Abdo_fat_gt.nii')
    # get_y_data(y_path)

if __name__ == '__main__':
    main()    
    



