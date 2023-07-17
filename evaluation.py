#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:11:38 2021
@author: Yeshe Kway
"""

from ImageProcessing import (preprocess_X,
                             preprocess_y,
                             # unpack_label_data,
                             postprocess, 
                             removePadding_labels, 
                             removePadding,
                             disabmle_volume, 
                             labels_to_axis)

from Data_Generator_new import unpack_label_data_tmp
from Utils import (load_Nifti,
                   sort_dir_aphanumerical,
                   process_config_args,
                   load_config,
                   findBestModelFile,
                   load_model)



from evaluate_volumes import get_nifti_spacing
# from apply_model import inference
from ResNetUnet import ResNetU, get_model
from Utils import get_device
from pathlib import Path
from medpy import metric
from tqdm import tqdm
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import yaml
import sys
import os 

from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances


def v_t_c(l):
    return l.cpu().detach().numpy()

def volumes_to_cpu(liste):
    return list(map(v_t_c, liste))


class Model_Evaluator():
    
    def __init__(self):
        
        self.device = get_device()
        self.metric_func_dic = {'fp': self.calc_FalsePositive, 
                                'fn': self.calc_FalseNegative, 
                                'tp': self.calc_TruePositiv, 
                                'tn': self.calc_TrueNegative,
                                'fn_rate': self.calc_FalseNegative_rate,
                                'fp_rate': self.calc_FalsePositive_rate, 
                                'tn_rate': self.calc_TrueNegative_rate,
                                'tp_rate': self.calc_TruePositive_rate, 
                                'precision': self.calc_Precision,
                                'dice': self.calc_Dice, 
                                'HD': self.calc_HausDorff_Distance, 
                                'HD95': self.calc_HausDorff_95, 
                                "DSCS": self.compute_surface_DICE_tolerance}

# =============================================================================
#                          Metrics
# =============================================================================

    def calc_FalseNegative(self, groundTruth, prediction):
       'False Negative (FN): we predict a label of 0 (negative), but the true label is 1.'
       FN = np.sum(np.logical_and(prediction == 0, groundTruth == 1))
       return FN
   
    def calc_FalsePositive(self, groundTruth, prediction):
       'False Positive (FP): we predict a label of 1 (positive), but the true label is 0.'
       FP = np.sum(np.logical_and(prediction == 1, groundTruth == 0))
       return FP

    def calc_TruePositiv(self, groundTruth, prediction):
       'True Positive (TP): we predict a label of 1 (positive), and the true label is 1'
       TP = np.sum(np.logical_and(prediction == 1, groundTruth == 1))       
       return TP
   
    def calc_TrueNegative(self, groundTruth, prediction):
       ' True Negative (TN): we predict a label of 0 (negative), and the true label is 0.'
       TN = np.sum(np.logical_and(prediction == 0, groundTruth== 0))
       return TN

    def calc_FalsePositive_rate(self, groundTruth, prediction):
        FP = self.calc_FalsePositive(groundTruth, prediction)
        TN = self.calc_TrueNegative(groundTruth, prediction)
        return FP/(FP+TN) 

    def calc_FalseNegative_rate(self, groundTruth, prediction):
        FN = self.calc_FalseNegative(groundTruth, prediction)
        TP = self.calc_TruePositiv(groundTruth, prediction)
        return FN/(FN+TP)

    def calc_TrueNegative_rate(self, groundTruth, prediction):
       ' aka specificity'
       TN = self.calc_TrueNegative(groundTruth, prediction)
       FP = self.calc_FalsePositive(groundTruth, prediction)       
       return TN/(TN+FP)

    def calc_TruePositive_rate(self, groundTruth, prediction):
       ' aka sensitivity'
       TP = self.calc_TruePositiv(groundTruth, prediction)
       FN = self.calc_FalseNegative(groundTruth, prediction)
       return TP/(TP+FN)

    def calc_Precision(self, groundTruth, prediction):
       ' aka (Positive predictive value)'
       TP = self.calc_TruePositiv(groundTruth, prediction)
       FP = self.calc_FalsePositive(groundTruth, prediction)
       return TP/(TP+FP)    

    def calc_Pix_Accuracy(self, groundTruth, prediction):
        TP = self.calc_TruePositiv(groundTruth, prediction)
        TN = self.calc_TrueNegative(groundTruth, prediction)
        FP = self.calc_FalsePositive(groundTruth, prediction)
        FN = self.calc_FalseNegative(groundTruth, prediction)
        return (TP + TN) / (TP + TN + FP + FN)

    def calc_FNFPTPTN(self, groundTruth, prediction):
       '''
       :param self:
       :param groundTruth: groundtruth data in 
       :param prediction: prediction array we want to evaluate agains 
                          groundtruth  
       '''
       FN = self.calc_FalseNegative(groundTruth, prediction)
       FP = self.calc_FalsePositive(groundTruth, prediction)
       TP = self.calc_TruePositiv(groundTruth, prediction)
       TN = self.calc_TrueNegative(groundTruth, prediction)
       print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))    


    def calc_Dice(self, groundTruth, prediction, non_seg_score=1.0):
       """
       Computes the Dice coefficient.
       Args:
           true_mask : Array of arbitrary shape.
           pred_mask : Array with the same shape than true_mask.  
       Returns:
           A scalar representing the Dice coefficient between the two 
           segmentations. 
       """
       assert groundTruth.shape == prediction.shape       
       groundTruth = np.asarray(groundTruth).astype(np.bool)
       prediction = np.asarray(prediction).astype(np.bool)       
       # If both segmentations are all zero, the dice will be 1.
       im_sum = groundTruth.sum() + prediction.sum()
       if im_sum == 0:
           return non_seg_score
       # Compute Dice coefficient
       intersection = np.logical_and(groundTruth, prediction)
       return 2. * intersection.sum() / im_sum
    
    
    def calc_HausDorff_Distance(self, groundTruth, prediction, voxelspacing=None): 
        return metric.hd(prediction, groundTruth, voxelspacing=voxelspacing, connectivity=1)    
    
    
    def calc_HausDorff_95(self, groundTruth, prediction, voxelspacing=None):
        return metric.hd95(prediction, groundTruth, voxelspacing=voxelspacing, connectivity=1)
    
    
    def compute_surface_DICE_tolerance(self, groundTruth, prediction, voxel_spacing, t=1):
        return compute_surface_dice_at_tolerance(compute_surface_distances(groundTruth, prediction, voxel_spacing), 1)
    
    
    def compute_metrics(self, gtruth, pred, voxel_spacing=None):
        '''
        Parameters
        ----------
        groundTruth : TYPE
            DESCRIPTION. shape has to be (z, w, h, batch_size)
        prediction : TYPE
            DESCRIPTION. shape has to be (z, w, h, batch_size)
        Returns
        -------
        computed_metrice_dic : TYPE
            DESCRIPTION.
        '''
        
        keys_with_spacing = ["DSCS", "HD", "HD95"]
        
        # gtruth = np.moveaxis(groundTruth, -1, 0)
        # pred = np.moveaxis(prediction, -1, 0)
        # compute all metrices for all labels
        computed_metrice_dic = {}
        for i, label_name in enumerate(self.labelnames):
            for metric_key in self.metric_func_dic:
                if metric_key in keys_with_spacing and voxel_spacing != None:
                    computed_metrice_dic[label_name+ '_' + metric_key] = self.metric_func_dic[metric_key](gtruth[i], pred[i], voxel_spacing)        
                else:
                    computed_metrice_dic[label_name+ '_' + metric_key] = self.metric_func_dic[metric_key](gtruth[i], pred[i])        
        return computed_metrice_dic


# =============================================================================
#                          Help Functions 
# =============================================================================

    # def load_model(self, config, pathToModel):
    #     # load model
    #     model = get_model(config)
    #     # load model state 
    #     # best_model = findBestModelFile(pathToModel)
    #     # print('Best model name: ' + best_model)    
    #     # path_to_bestModel = os.path.join(pathToModel, best_model)
    #     # model.load_state_dict(torch.load(path_to_bestModel))
    #     model.eval()
    #     return model
    
    def inference(self, volume):
        '''
        This functions perfromes preprocessing and applies the model to the 
        given volume 
        '''
        # ----------------------------------------------------------
        # store original shape 
        original_shape = volume.shape 
        # ----------------------------------------------------------
        # check volume is multiple of model input z dimension 
        residual = volume.shape[0]-self.model_input_shape[0]
        # ----------------------------------------------------------
        # extract patches 
        patches, residual = disabmle_volume(volume, self.model_input_shape[0])
        segmentations = [] # storage for argmax output 
        probabilities = [] # storage raw model output (probability maps)
        for patch in patches:
            # -----------------------------------------------------
            # preprocessing 
            X_data = preprocess_X(volume = patch,
                                  input_dim = self.model_input_shape,
                                  normalization=self.config['input_normalization']) 
            # -----------------------------------------------------
            #  make prediction      
            X_data = np.expand_dims(X_data, axis=0)
            X_data = torch.tensor(X_data, dtype=torch.float)
            X_data = X_data.cuda()
            probs = self.model(X_data)
            probabilities.append(torch.squeeze(probs))
            # -----------------------------------------------------
            # postprocess model output 
            out = postprocess(probs) # compute argmax and convert tensor to numpy
            segmentations.append(out)
        # ------------------------------------------------------------
        # if input volume was not evenly devidable to model input shape 
        # extract only relevant part of last processed volume 
        if residual != 0:
            # --------------------------------------------------
            # for processed segmentation maps 
            res_vol = segmentations[-1]
            res_vol = res_vol[-residual:, ...]
            segmentations.pop()
            segmentations.insert(len(segmentations), res_vol)
            # --------------------------------------------------
            # for probability maps 
            res_prob = probabilities[-1]
            res_prob = res_prob[:, -residual:, ...]
            probabilities.pop()
            probabilities.insert(len(probabilities), res_prob)            
        # --------------------------------------------------
        
        probabilities = volumes_to_cpu(probabilities)
        probabilities = np.concatenate(probabilities, 1)
        self.probs = np.squeeze(removePadding_labels(probabilities, original_shape))         
        
        # stack back to one volume 
        segmentations = np.concatenate(segmentations, 0)
        # --- remove zerro padding 
        seg = removePadding(segmentations, original_shape)
        return seg    


    def set_label_names(self, config):
        if config['n_output_channels'] == 7 and config['include_DSAT'] == True:
            self.labelnames = ("bg", "ssat", "dsat", "ipat", "rpat", "psat", "others")
        elif config['n_output_channels'] == 6 and config['include_DSAT'] != True:
            self.labelnames = ("bg", "sat", "ipat", "rpat", "psat", 'others')
        elif config['n_output_channels'] == 6 and config['include_DSAT'] == True:
            self.labelnames = ("bg", "ssat", "dsat", "ipat", "rpat", "psat")

    def set_label_names_default(self):
        self.labelnames = ("bg", "ssat", "dsat", "ipat", "rpat", "psat")


    def evaluateModel(self, config, pathToModel, test_IDs,
                      excel_name='Perfromance_TestSet.xlsx' ,crf=False):
        '''
        evaluate model performance during training 
        '''
        # ---------------------------------------------------------------------
        # set general evaluation settings
        self.config = config
        self.set_label_names(config)
        self.model_input_shape = config['patch_dim']
        self.model = load_model(config, pathToModel, False)
        # self.model = get_model(config) # get/define model 
        
        self.model.eval()
        self.model.cuda()
        pathToTestVolumes = config['data_path']
        # ---------------------------------------------------------------------
        # result storage 
        metrics_raw = pd.DataFrame() 
        metrics_crf = pd.DataFrame() 
        # --------------------------------------------------------------------- 
        # loop through test IDs
        for folder in test_IDs:
            # --- load inputs 
            X_path = os.path.join(pathToTestVolumes, folder, config['x_file_name'])
            y_path = os.path.join(pathToTestVolumes, folder, config['y_file_name'])
            X_data = load_Nifti(X_path) # load volume
            y_gt = load_Nifti(y_path) # load label 
            y_gt = unpack_label_data_tmp(y_gt,include_dsat=config['include_DSAT'])            
            # --- apply model on volume 
            y_pred = labels_to_axis(self.inference(X_data))
            # --- compute metrics
            res = self.compute_metrics(y_gt, y_pred)
            new_row = pd.Series(res)
            new_row.name = folder
            metrics_raw = metrics_raw.append(new_row)
            
            if crf:
                y_pred_crf = crf_2D(X_data, self.probs)
                y_pred_crf = np.moveaxis(y_pred_crf, -1, 0)
                res = self.compute_metrics(y_gt, y_pred_crf)    
                new_row = pd.Series(res)
                new_row.name = folder
                metrics_crf = metrics_crf.append(new_row)             
        if crf:
            self.saveResults(metrics_crf,
                              pathToModel,
                              excelename = 'Performance_crf.xlsx')
        self.saveResults(metrics_raw,
                         pathToModel,
                         excelename=excel_name)    
        
        # return mean values 
        mean_perfrmance = metrics_raw.mean().squeeze()
        mean_perfrmance.name = 'mean'
        return mean_perfrmance


    def compute_evaluation_metrics(self, data_path, dst, mask_name_pred, mask_name_gt, 
                      excel_name='Perfromance_TestSet.xlsx'):
        """
        This function loops through a data set and computes several evaluation metrics
        Each fold needs to contain two segmentation maks with the specified names
        
        Parameters
        ----------
        data_path : str
            Path to data.
        dst : str
            dst where results will be saved as xlsx.
        mask_name_pred : str
        mask_name_gt : str
        excel_name : "name".xlsx, optional
            The default is 'Perfromance_TestSet.xlsx'.

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------
        # result storage
        metrics_raw = pd.DataFrame() 
        # --------------------------------------------------------------------- 
        # loop through test IDs
        for folder in tqdm(os.listdir(data_path)):
            # --- load inputs 
            y_pred = load_Nifti(str(data_path/folder/mask_name_pred)) # load volume
            y_pred = unpack_label_data_tmp(y_pred, include_dsat=True)           
            y_nii = nib.load(str(data_path/folder/mask_name_gt)) # load nifti
            y_gt = load_Nifti(str(data_path/folder/mask_name_gt)) # load volume
            y_gt  = unpack_label_data_tmp(y_gt , include_dsat=True)           
            voxel_spacing = get_nifti_spacing(y_nii)
            
            # --- compute metrics
            res = self.compute_metrics(y_gt, y_pred, voxel_spacing)
            new_row = pd.Series(res)
            new_row.name = folder
            metrics_raw = metrics_raw.append(new_row)
        self.saveResults(metrics_raw,
                         dst,
                         excelename=excel_name)    
        
    # =========================================================================
    #               Post Evaluation Code for SAMS data -- START     
    # =========================================================================    

    def post_evaluateModel(self, config, pathToModel, data_path, dst,
                      excel_name='Perfromance_TestSet.xlsx'):
        '''
        Evaluate model on data after training
        '''
        # ---------------------------------------------------------------------
        # set general evaluation settings
        self.config = config
        self.set_label_names(config)
        self.model_input_shape = config['patch_dim']
        self.model = load_model(config, pathToModel, False)
        self.model.eval()
        self.model.cuda()
        # ---------------------------------------------------------------------
        # result storage 
        metrics_raw = pd.DataFrame() 
        # --------------------------------------------------------------------- 
        # loop through test IDs
        for folder in os.listdir(data_path):
            # --- load inputs 
            X_path = os.path.join(data_path, folder, config['x_file_name'])
            y_path = os.path.join(data_path, folder, config['y_file_name'])
            X_data = load_Nifti(X_path) # load volume
            y_gt = load_Nifti(y_path) # load label 
            y_gt = unpack_label_data_tmp(y_gt, include_dsat=config['include_DSAT'])            
            # --- apply model on volume 
            y_pred = labels_to_axis(self.inference(X_data))
            # --- compute metrics
            res = self.compute_metrics(y_gt, y_pred)
            new_row = pd.Series(res)
            new_row.name = folder
            metrics_raw = metrics_raw.append(new_row)
            
        self.saveResults(metrics_raw,
                         dst,
                         excelename=excel_name)    
        # return mean values 
        mean_perfrmance = metrics_raw.mean().squeeze()
        mean_perfrmance.name = 'mean'
        return mean_perfrmance
    
    def postcompute_kfold_validation_external_data(self, kfoldspath, data_path, dst):
        # loop through all k-fold models 
        config_path = load_config(kfoldspath)
        for model_folder in tqdm(os.listdir(kfoldspath)):
            folder_path = kfoldspath/model_folder
            if os.path.isdir(folder_path):
                self.post_evaluateModel(config_path, str(folder_path), data_path, dst,
                      excel_name= model_folder + '.xlsx')
        
    # =========================================================================
    #               Post Evaluation Code for SAMS data -- END
    # =========================================================================          
    
    
    
    # =========================================================================
    #               Post Evaluation Code for SPRESTO data -- START     
    # =========================================================================    

    def post_evaluateModel_SPRESTO(self, config, k_fold_path, data_path, dst,
                      excel_name='Perfromance_TestSet.xlsx'):
        '''
        Evaluate model on data after training
        '''
        # ---------------------------------------------------------------------
        # set general evaluation settings
        self.config = config
        self.set_label_names(config)
        self.model_input_shape = config['patch_dim']
        self.model = load_model(config, str(k_fold_path), False)
        self.model.eval()
        self.model.cuda()
        # ---------------------------------------------------------------------
        # result storage 
        metrics_raw = pd.DataFrame() 
        # --------------------------------------------------------------------- 
        # loop through test IDs
        test_ids = pd.read_excel(str(k_fold_path/"testing_ids.xlsx"), index_col=1).index.to_list()
        
        for folder in test_ids:
            # --- load inputs 
            X_path = os.path.join(data_path, folder, config['x_file_name'])
            y_path = os.path.join(data_path, folder, config['y_file_name'])
            X_data = load_Nifti(X_path) # load volume
            y_gt = load_Nifti(y_path) # load label 
            y_gt = unpack_label_data_tmp(y_gt, include_dsat=config['include_DSAT'])            
            # --- apply model on volume 
            y_pred = labels_to_axis(self.inference(X_data))
            # --- compute metrics
            res = self.compute_metrics(y_gt, y_pred)
            new_row = pd.Series(res)
            new_row.name = folder
            metrics_raw = metrics_raw.append(new_row)
            
        self.saveResults(metrics_raw,
                          dst,
                          excelename=excel_name)    
        # return mean values 
        mean_perfrmance = metrics_raw.mean().squeeze()
        mean_perfrmance.name = 'mean'
        return mean_perfrmance
    
    
    def get_folder_names(self, directory):
        folder_names = []
        # Iterate over all items in the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            # Check if the item is a directory/folder
            if os.path.isdir(item_path):
                folder_names.append(item)
        return folder_names    
    
    
    def postcompute_kfold_validation_SPRESTO(self, kfoldspath, data_path, dst):
        # loop through all k-fold models 
        config_path = load_config(kfoldspath)
        for model_folder in tqdm(self.get_folder_names(kfoldspath)):
            self.post_evaluateModel_SPRESTO(config_path, kfoldspath/model_folder, data_path, dst,
                      excel_name= model_folder + '.xlsx')
        
    # =========================================================================
    #               Post Evaluation Code for SPRESTO data -- END
    # =========================================================================       
    
    
    
    
    def saveResults(self, model_performance, dst, excelename):
        means = model_performance.mean()
        means.name = 'mean'
        stds = model_performance.std()
        stds.name = 'std'    
        model_performance = model_performance.append([means, stds])
        # tmp 
        # model_performance = model_performance[list(map(add_dice, list(self.labelnames)))]
        model_performance.to_excel(os.path.join(dst, excelename))      

def add_dice(name):
    return name + '_dice'


def format_res(mean, std):
    return (str(round(mean*100, 2)) + ' ± '+ str(round(std*100, 2)))
            

def compute_mean_table(path, dst):
    fat_names = ['ssat', 'dsat', 'ipat', 'rpat', 'psat']
    suffixes = {'_dice':'dice',
                '_fp_rate': 'fp',
                '_fn_rate': 'fn',
                '_precision': 'precision',
                '_sensitivity': 'sensitivity', 
                '_HD95': "HD95"
                }
    res_df = pd.DataFrame(index= fat_names ,columns=suffixes.values())
    for file in os.listdir(path): # loop through folds 
        tmp_res = pd.read_excel(path/file, index_col=0) # get data 
        for f_name in fat_names: # loop through fat
            for suffix in suffixes.keys(): # loop through metrics
                if suffix== '_sensitivity':
                    t = tmp_res.drop(['mean', 'std'])
                    sens = t[f_name+'_tp']/(t[f_name+'_tp']+t[f_name+'_fn'])
                    mean = np.mean(sens) 
                    std = np.std(sens)
                else:
                    mean = tmp_res.loc['mean', f_name+suffix] 
                    std = tmp_res.loc['std', f_name+suffix]                  
                
                if suffix!="_HD95":
                    res_df.loc[f_name, suffixes[suffix]] = format_res(mean, std)
                else:
                    res_df.loc[f_name, suffixes[suffix]] = (str(round(mean, 2)) + ' ± '+ str(round(std, 2)))
                    
    res_df.to_excel(dst)          
    


def main():
    # # -------------------------------------------------------------------------
    # # server paths 
    # pathToModel = "/app/Code/PyTorch/3D_segmentation/TrainedModels/5_fold_ex1_model1_excludeothers/model_type_dynamic_unet_3D_7_bz2LR_0.0001_patience_40_n_input_channels1_n_out_channels5AUG_1loss_BCEDiceLoss_n_layer6_n_filter24_input_normalizationzeroone"
    # # pathToData = "//app/Data/IntraRetroPeritoneal/AllVolumes"    
    # # test_id_path = '/app/Code/PyTorch/3D_segmentation/TrainedModels/5_fold_ex1_model1_excludeothers/testing_ids.xlsx'
    # test_id_path = '/media/kwaygo/BackupPlus/Data/Extracted/SAMS/SAMS_DATA_Experiment/Abdominal_Volumes_SAMS_ROI_VAT_corrections.xlsx'
    # # -------------------------------------------------------------------------
    # # local test paths
    # # pathToModel = "//home/kwaygo/Documents/Projects/Torch/tmp/tmp"
    # # pathToData = "//media/kwaygo/ymk_HDD3/Experiments/Spresto_Applications/SprestoSamples_DSAT_SSAT_ParaRetroPeritoneal/spresto_retroperitoneal_samples/Test_subjects/O"    
    
    # # -------------------------------------------------------------------------
    # test_IDs = pd.read_excel(test_id_path, index_col=0)[0].tolist()
    # config = load_config(pathToModel)
    # ME = Model_Evaluator()
    # ME.evaluateModel(config, pathToModel, test_IDs, crf=False)    


    # =========================================================================
    #     Compute evaluation metics for nifti masks
    # =========================================================================
    # data_path = Path(r'/media/yeshe/BackupPlus/Backup - Coding/ModellingCode - Final/3D_AbdoFat-segmentation-kfold-training-EntireCohort/Data/Intra - Rater - Set/Set_2/All_Subjects')
    # dst = Path(r'/media/yeshe/BackupPlus/Backup - Coding/ModellingCode - Final/3D_AbdoFat-segmentation-kfold-training-EntireCohort/Data/Intra - Rater - Set/Set_2')
    # ME = Model_Evaluator()
    # ME.set_label_names_default()
    # ME.compute_evaluation_metrics(data_path,
    #                           dst,
    #                           mask_name_pred='Abdo_fat_gt.nii.gz',
    #                           mask_name_gt='Abdo_fat_gt2.nii.gz', 
    #                           excel_name='Test_hausdorf.xlsx')
    
    # =========================================================================
    #               Post-compute evaluation metics for k-folds 
    # =========================================================================
    # data_path = Path('/app/Data/SAMS/Final_Train_Data')
    # # dst = '/app/Code/PyTorch/3D_segmentation-kfold-training-EntireCohort/Results/External_validation/No_aug'
    # dst = '/app/Code/PyTorch/3D_segmentation-kfold-training-EntireCohort/Results/External_validation/50samples/no_aug'
    # k_fold_path = Path('/app/Code/PyTorch/3D_segmentation-kfold-training-EntireCohort/Results/No_Augmentation_2/CheckPoints-5-Folds')
    # ME = Model_Evaluator()
    # ME.postcompute_kfold_validation_external_data(k_fold_path, data_path, dst)
    
    
    # -------------------------------------------------------------------------
    #               Post-compute avarage performance
    # -------------------------------------------------------------------------
    path = Path(r'E:\Backup - Coding\ModellingCode - Final\3D_AbdoFat-segmentation-kfold-training-EntireCohort\Results\Review_Answer\Main_model_results\K_fold_test_sets')
    dst = Path(r'E:\Backup - Coding\ModellingCode - Final\3D_AbdoFat-segmentation-kfold-training-EntireCohort\Results\Review_Answer\Main_model_results\K_fold_test_sets\summary_holdout.xlsx')
    compute_mean_table(path, dst)
    
    
    # =========================================================================
    #               Post-compute evaluation metics for k-folds 
    # =========================================================================
    # data_path = Path('/media/yeshe/BackupPlus/Backup - Coding/ModellingCode - Final/3D_AbdoFat-segmentation-kfold-training-EntireCohort/TEST_pipeline/TMP_data')
    # # dst = '/app/Code/PyTorch/3D_segmentation-kfold-training-EntireCohort/Results/External_validation/No_aug'
    # dst = '/media/yeshe/BackupPlus/Backup - Coding/ModellingCode - Final/3D_AbdoFat-segmentation-kfold-training-EntireCohort/TEST_pipeline/test'
    # k_fold_path = Path('/media/yeshe/BackupPlus/Backup - Coding/ModellingCode - Final/3D_AbdoFat-segmentation-kfold-training-EntireCohort/TEST_pipeline/CheckPoints-5-Folds')
    # ME = Model_Evaluator()
    # ME.postcompute_kfold_validation_SPRESTO(k_fold_path, data_path, dst)
    
    
    

if __name__ == '__main__':
    main()    
    
