#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jul 12 08:12:07 2021
@author: yeshe kway 
"""
from ImageProcessing import pad_Volume_aug, pad_Volume_labels_aug, preprocess_X_patch
from Data_Generator_new import (Data_Generator_builder,
                                get_xy_data, 
                                get_xy_data_patch)
from Utils import get_configurations, get_device, define_start_end_random_patch
from ResNetUnet import get_model, print_model_parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from callbacks import LRScheduler, EarlyStopping
from Server_help import delete_models_keep_best
from evaluation import Model_Evaluator
from loss_functions import get_loss
from shutil import copyfile
import pandas as pd
import numpy as np
import itertools    
import random
import torch 
import time 
import re 
import os 


# =============================================================================
#                           Help Functions 
# =============================================================================

def copy_config_file(config, dst_folder):
    # copy config file to folder path 
    copyfile(config['config_path'],
             os.path.join(dst_folder, os.path.basename(config['config_path'])))  

def print_memory_usage(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



def categories_to_columns(df):
    '''
    1. Creates a extra column to identify categiries (categorie label is 
                            extracted from first caracter of folder name 
    2. extract all subjects of a categirie shuffels the list and 
        create a column for each categorie                             
    '''
    
    # ------------------------------------------------------------------------
    # create category column 
    df['category'] = 0
    for index, row in df.iterrows():
        df.loc[index,'category'] = df.loc[index,'subject_ID'][0]
    # get unique categories 
    categories = df['category'].unique()
    # -------------------------------------------------------------------------
    # create column for each categorie 
    sorted_df = {}
    for c in categories:
        s = df.loc[df['category'] == c]  # get all subj IDs in categiry 'c'
        l = s['subject_ID'].values.tolist()   
        random.shuffle(l) # shuffle the list 
        sorted_df[c] = l
    new_df = pd.DataFrame.from_dict(sorted_df)
    return new_df 


def list_to_df(liste):
    df = pd.DataFrame(liste)
    df.columns = ['subject_ID']
    df = categories_to_columns(df)
    return df

    
def define_kfold_sets(data_path, k):
    '''
    Define k-folds from data path and returns panda df with ids for the folds
    this functions stratifies the folds by checking first character of foldername
    returns: fold IDs
    '''
    folders = os.listdir(data_path)
    df_c = list_to_df(folders)
    stratified_samples_per_fold = 2
    df2 = df_c[df_c.index % stratified_samples_per_fold == 0].reset_index(drop=True)  # Selects every nth row starting from 0    
    df1 = df_c[df_c.index % stratified_samples_per_fold != 0].reset_index(drop=True) # Excludes every nth row starting from 0
    d = pd.concat([df1, df2], 1)
    return d


# def create_folds(df, n_folds):
    

def define_kfold_sets_1(data_path, k):
    '''
    Define k-folds from data path and returns panda df with ids for the folds
    this functions stratifies the folds by checking first character of foldername
    returns: fold IDs
    '''
    folders = os.listdir(data_path)
    df_c = list_to_df(folders)
    df1 = df_c.iloc[0:5, :].reset_index(drop=True)  
    df2 = df_c.iloc[5:10, :].reset_index(drop=True)  
    df3 = df_c.iloc[10:15, :].reset_index(drop=True)  
    d = pd.concat([df1, df2, df3], 1)
    return d
    
    
# =============================================================================
#                           Mode Trainer  
# =============================================================================        


class Trainer():
    
    def __init__(self, config):
        
        # ---------------------------------------------------------------------
        # set general training settings
        self.config = config
        self.input_normalization = config['input_normalization']
        self.patch_training = config['patch_training']
        self.input_dim = config['patch_dim']
        self.device = get_device() # select device 
        self.epoch_verbose = config['epoch_verbose']
        self.loss_f = get_loss(config) # define loss function
        self.n_epochs = config['n_epochs']
        self.dst_path = self.create_dst(config) # destination at which model checkpoints will be saved at 

    # =========================================================================
    #                      Getter Methods 
    # =========================================================================


    # def get_data_generators(self, config):
    #     # define train and validatio lists
    #     self.train_ids, self.val_ids = get_ID_lists(config['data_path'])
    #     # instantiate Data Generators 
    #     dgb = Data_Generator_builder(config, self.train_ids, self.val_ids)
    #     if config['patch_training']:
    #         train_Gen, val_Gen = dgb.get_train_val_generators_patch()
    #     else:
    #         train_Gen, val_Gen = dgb.get_train_val_generators()
    #     return train_Gen, val_Gen


    def get_optimizer(self, config):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        return optimizer

    # =========================================================================
    #                       Help Training Methods 
    # =========================================================================
    
    def set_callbacks(self, config):
        # --- initialize  early_stopping object
        self.early_stopping = EarlyStopping(patience=config['patience'],
                                            delta=config['es_delta'], 
                                            path=self.model_checkpoint_dst, 
                                            verbose=config['es_verbose'])
        # --- initialize learning rate decay
        self.lr_decay = config['lr_decay']
        if self.lr_decay:
            self.lr_shedule = ReduceLROnPlateau(optimizer=self.optimizer,
                                                mode='max',
                                                factor=0.1,
                                                min_lr=0.001,
                                                patience=10,
                                                verbose=True)

    # def save_checkpoint(self, epoch, loss):
    #     model_path = os.path.join(self.dst_path, str(epoch) + '_best-model-parameters-' 'loss_'+ str(round(loss, 4)) +'.pt')
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'loss': loss,
    #         }, model_path
    #     )


    def create_dst(self, config):
        # define new dir
        # folder_name = self.define_foldername(config)
        # dst_path = os.path.join(config['dst'], folder_name)
        dst_path = os.path.join(config['dst'], 'CheckPoints-' + str(config['kfolds']) + '-Folds')
        # make folder 
        os.mkdir(dst_path)
        # additionally copy config file 
        copy_config_file(config, dst_path)
        return dst_path


    def define_foldername(self, config):
        aug =  1 if config['augmentation'] else 0
        folder_name =   'model_type_' + config['model_type'] + \
                        '_bz' + str(config['batch_size']) + \
                        'LR_' + str(config['learning_rate']) + \
                        '_patience_' + str(config['patience']) + \
                        '_n_input_channels' + str(config['n_input_channels']) + \
                        '_n_out_channels' + str(config['n_output_channels']) + \
                        'AUG_'+ str(aug) +\
                        'loss_' + config['loss_function'] + \
                        '_n_layer' + str(config['n_layer']) + \
                        '_n_filter' + str(config['n_filter']) + \
                        '_input_normalization' + config["input_normalization"]
        return folder_name 


    def print_train_setting_info(self, ids_train, ids_eval, ids_test):
        print('='*50)
        print('Training settings:')
        print('Training on: ', self.device)
        print_memory_usage(self.device)
        print('='*50)
        print('Model information')
        print('input shape', self.input_dim)
        print_model_parameters(self.model)
        print('Training on:', len(ids_train))
        print('Validation on:', len(ids_eval))
        print('Testing on:', len(ids_test))
        print('='*50)
    
    
    def clear_mem(self):
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()
        
    # =========================================================================
    #                           Training Function
    # =========================================================================


        # def train(self):
    #     # set model to device 
    #     self.model.to(device=self.device)
    #     # initialize empty list to track losses
    #     train_losses, val_losses = [], []
    #     # valid_loss_min = np.Inf
    #     # start timer 
    #     start = time.time()
    #     # print training information 
    #     self.print_train_setting_info()
        
    #     # Loop over epochs
    #     for epoch in range(self.n_epochs):
    #         # --------------------- training ----------------------------------
    #         t_loss = self.run_epoch_training(self.training_generator)
    #         # --------------------- validation --------------------------------
    #         v_loss = self.run_epoch_validation(self.validation_generator) 
    #         # --------------------- Print epoch information -------------------
    #         print_msg = (f'Epoch {epoch+1} of {self.n_epochs} ' +
    #                       f'train_loss: {t_loss:.5f} ' +
    #                       f'valid_loss: {v_loss:.5f}')
    #         print(print_msg)
    #         # --------------------- callbacks ---------------------------------
    #         # --- check early stopping criteria
    #         self.early_stopping(v_loss, self.model, epoch)
    #         if self.early_stopping.early_stop:
    #             print('Early stopping')
    #             break     
    #         # --- check lr decay
    #         if self.lr_decay: 
    #             self.lr_shedule.step(v_loss)
            
    #         # # save model if validation loss has decreased
    #         # if v_loss < valid_loss_min:
    #         #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,v_loss))
    #         #     # save checkpoint as best model
    #         #     self.save_checkpoint(epoch, v_loss)
    #         #     valid_loss_min = v_loss
            
    #         # ------------- save loss values for train histroy ----------------
    #         train_losses.append(t_loss)
    #         val_losses.append(v_loss)
    #     end = time.time()
    #     print('='*50)
    #     print('Training completed')
    #     print(f"Training time: {(end-start)/60:.3f} minutes")
    #     self.clear_mem()



            
    
    
    
    def train_model(self):
        
        # set model to device 
        self.model.to(device=self.device)
        # initialize empty list to track losses
        train_losses, val_losses = [], []
        # valid_loss_min = np.Inf
        # start timer 
        start = time.time()
        
        # Loop over epochs
        for epoch in range(self.n_epochs):
            # --------------------- training ----------------------------------
            t_loss = self.run_epoch_training(self.train_Gen)
            # --------------------- validation --------------------------------
            v_loss = self.run_epoch_validation(self.val_Gen) 
            # --------------------- Print epoch information -------------------
            if self.epoch_verbose:
                print_msg = (f'Epoch {epoch+1} of {self.n_epochs} ' +
                              f'train_loss: {t_loss:.5f} ' +
                              f'valid_loss: {v_loss:.5f}')
                print(print_msg)
            # --------------------- callbacks ---------------------------------
            # --- check early stopping criteria
            self.early_stopping(v_loss, self.model, epoch)
            if self.early_stopping.early_stop:
                print('Early stopping')
                break     
            # --- check lr decay
            if self.lr_decay: 
                self.lr_shedule.step(v_loss)
            
            # ------------- save loss values for train histroy ----------------
            train_losses.append(t_loss)
            val_losses.append(v_loss)
        
        end = time.time()
        self.save_train_curves(train_losses, val_losses, end-start)        
        # print('='*50)
        # print('Training completed')
        # print(f"Training time: {(end-start)/60:.3f} minutes")


    def save_train_curves(self, train_loss, val_loss, execution_time):
        '''
        data to Dataframe and save as excel 
        '''
        
        excel_name = 'training_curves.xlsx'
        d = {'training_loss': train_loss, 'validation_loss': val_loss, 
             'execution_time':execution_time}
        train_hist = pd.DataFrame.from_dict(d)
        train_hist.to_excel(os.path.join(self.model_checkpoint_dst, excel_name))

    def run_epoch_training(self, data_generator):
        # set training mode 
        self.model.train()
        running_loss = 0.0
        # Training   
        for count, data in enumerate(data_generator, 1): # loop through all samples/batches 
            # Transfer volumes to GPU
            data = get_xy_data_patch(data, self.config['input_normalization'])  if self.patch_training  else get_xy_data(data) 
            X_data, y_data = data[0].to(self.device, dtype=torch.float), data[1].to(self.device, dtype=torch.float)        
            # reset gradients
            self.optimizer.zero_grad()
            # forward propagation
            out = self.model(X_data)
            # compute lossn
            loss = self.loss_f(out, y_data)
            # backpropagation
            loss.backward() 
            #update the parameters
            self.optimizer.step()
            # save loss
            running_loss += loss.item()
            del X_data
            del y_data
            del out
        epoch_loss = running_loss / count
        return epoch_loss 
    
    
    def run_epoch_validation(self, data_generator):
        # set evaluation mode
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            # loop through all samples/batches
            for count, data in enumerate(data_generator, 1): 
                data = get_xy_data_patch(data, self.config['input_normalization']) if self.patch_training  else get_xy_data(data) 
                # Transfer volumes to GPU
                X_data, y_data = data[0].to(self.device, dtype=torch.float), data[1].to(self.device, dtype=torch.float)        
                # forward data through model 
                out = self.model(X_data)
                # calculate the loss
                loss = self.loss_f(out, y_data)
                running_loss += loss.item()
                del X_data
                del y_data
                del out
        epoch_loss = running_loss/count
        return epoch_loss     
    
   # def train_kfold(self, config):
      
   #    # ---------------------------------------------------------------------
   #    # define k-fold stratified sets 
   #    fold_id = define_kfold_sets_1(config['data_path'], config['kfolds'])
   #    fold_id.to_excel(os.path.join(self.dst_path, 'fold_Ids.xlsx')) # save id info to excel 
   #    # define the k-fold train test set indices
       # kf = KFold(n_splits=config['kfolds'])
   #    index_gen = kf.split(fold_id.to_numpy())
   #    # instantiate Data Generator Builder (this class will help us later to create data generators)
   #    dgb = Data_Generator_builder(config)
   #    # instantiate model evaluater
   #    modelEvaluater = Model_Evaluator()
   #    # storage to save k evaluation results on test set 
   #    mean_metrics_test = pd.DataFrame()
      
   #    # ---------------------------------------------------------------------
   #    # perform k-fold training 
   #    for k, data in enumerate(index_gen,1):
          
   #        print('='*50)
   #        print('Training fold ' + str(k) + ' of ' + str(config['kfolds']))
          
   #        # -----------------------------------------------------------------
   #        # create checkpoint folder for nth fold
   #        self.model_checkpoint_dst = os.path.join(self.dst_path, str(k) + '_model_checkpoint')
   #        os.mkdir(self.model_checkpoint_dst)
          
   #        # -----------------------------------------------------------------
   #        # get train and test IDs
   #        strat_batch_train_IDs, strat_batch_test_IDs = data
   #        # get ID lists and also save them 
   #        ids_train, ids_eval, ids_test = self.get_train_test_evaluation_Id_lists(fold_id,
   #                                                                                strat_batch_train_IDs,
   #                                                                                strat_batch_test_IDs,
   #                                                                                self.model_checkpoint_dst)
   #        # -----------------------------------------------------------------
   #        # instantiate data Generators 
   #        self.train_Gen = dgb.get_data_generator(ids_train, validation=False)
   #        self.val_Gen = dgb.get_data_generator(ids_eval,validation=True)
   #        # self.test_Gen = dgb.get_data_generator(ids_test, validation=True)

   #        # -----------------------------------------------------------------
   #        # define model , set optimiser and callbacks
   #        self.model = get_model(config) # get/define model 
   #        self.optimizer = self.get_optimizer(config) # define optimizer
   #        self.set_callbacks(config)            
  
   #        # -----------------------------------------------------------------
   #        # start training 
   #        # self.print_train_setting_info(ids_train, ids_eval, ids_test)  # print training information 
   #        self.train_model()
          
   #        # -----------------------------------------------------------------
   #        # delete all model checkpoints but the best 
   #        # evaluate the best model on Test set 
   #        print('-'*30)
   #        print('Testing model on test data N=' + str(len(ids_test)))
   #        mean_results = modelEvaluater.evaluateModel(config, self.model_checkpoint_dst, ids_test)  
   #        mean_results.name = str(k) + '-fold-'+ mean_results.name
   #        mean_metrics_test = mean_metrics_test.append(mean_results)
   #        # -----------------------------------------------------------------
   #        # clear GPU mem 
   #        self.clear_mem()
      
   #    # save mean result over k folds 
   #    mean_metrics_test.to_excel(os.path.join(self.dst_path, 'OverallPerformance.xlsx')) 


    def add_cohort_BMI_info_to_df(self, all_ids):
        # load cohort data to get BMI info for stratified grouping
        indexs = list(map(get_number, all_ids))
        cohort_data = get_cohort_data('./spresto_cohort_latest.sav', indexs)
        
        # save IDs and BMI info in DataFrame
        ID_df = pd.DataFrame(data={'IDs':indexs}, index=indexs)
        ID_df['Weight_cats'] = cohort_data['pcv1_bmi_categories'].astype('category').cat.codes
        
        # recode BMI info into Nomralweight=0, Obese=1, Overweight=2, Underweight=3   
        ID_df.loc[ID_df['Weight_cats']==1, 'Weight_cats'] = 0 # add undefined (no BMI measuredm, N=4) to normal group  
        ID_df.loc[ID_df['Weight_cats']==2, 'Weight_cats'] = 1
        ID_df.loc[ID_df['Weight_cats']==3, 'Weight_cats'] = 2
        ID_df.loc[ID_df['Weight_cats']==4, 'Weight_cats'] = 3
        print('='*30)
        print(np.unique(ID_df['Weight_cats'], return_counts=True))        
        return ID_df
    
    
    def extract_external_testIDs(self, df):
        '''
        this function return the training IDs for cross-validation
        '''    
        all_ids = df.index.to_list()
        
        # select 89 for external testing and 300 IDs for cross-validation 
        test_IDs = random.sample(all_ids, 89)
        train_IDs = list(set(all_ids) - set(test_IDs))
        
        # test = df.groupby('Weight_cats', group_keys=False).apply(lambda x: x.sample(25))
        # print(np.unique(test['Weight_cats'], return_counts=True))        
        # test = 10
        
        # to DF
        test_IDs_df = pd.DataFrame(data={'subjectID': test_IDs}, index=test_IDs)
        train_IDs_df = pd.DataFrame(data={'subjectID': train_IDs}, index=train_IDs)
        # save as excels
        test_IDs_df.to_excel(os.path.join(self.config['dst'], 'all_test_ids.xlsx'))
        train_IDs_df.to_excel(os.path.join(self.config['dst'], 'all_train_ids.xlsx'))        
        
        print('='*30)
        print('Total Data for Training: ', len(train_IDs_df))
        print('External Testing Data: ', len(test_IDs_df))
        print('='*30)
        
        return train_IDs_df, test_IDs_df
        
        
    def save_IDs(self, train, test, val, dst):
        train.to_excel(os.path.join(dst, 'training_ids.xlsx'))
        test.to_excel(os.path.join(dst, 'validation_ids.xlsx'))
        val.to_excel(os.path.join(dst, 'testing_ids.xlsx'))

        
    def train_kfold(self, config):
        
        # ---------------------------------------------------------------------
        # Select Training Data for 5-fold-cross-validation experiment
        # get all subject ids from folder names
        all_ids = os.listdir(config['data_path'])
        # Add BM cat for stratification
        # ID_df = self.add_cohort_BMI_info_to_df(all_ids)
        
        all_train_IDs, ex_test_IDs = self.extract_external_testIDs(pd.DataFrame(all_ids, index=all_ids)) # this function only return the training IDs for cross-validation
        ex_test_id = ex_test_IDs.index.to_list() 
        
        # ---------------------------------------------------------------------
        # create the folds 
        kf = KFold(n_splits=config['kfolds'])
        index_gen = kf.split(all_train_IDs)
        
        # ---------------------------------------------------------------------
        # create class objects
        dgb = Data_Generator_builder(config)
        modelEvaluater = Model_Evaluator()
        
        # ---------------------------------------------------------------------
        # Results storage
        mean_metrics_test = pd.DataFrame()
        mean_results_exTest = pd.DataFrame()
        
        for k, data in enumerate(index_gen, 1):
        
            # -----------------------------------------------------------------
            # create checkpoint folder for kth fold
            self.model_checkpoint_dst = os.path.join(self.dst_path, str(k) + '_model_checkpoint')
            os.mkdir(self.model_checkpoint_dst)    
        
            # -----------------------------------------------------------------
            # split training data further into train and validation set 
            
            train_index, test_index = data
            fold_train_ids_all = all_train_IDs.iloc[train_index]
            ids_test = all_train_IDs.iloc[test_index]
            # split training IDs into train and validation set 
            ids_train, ids_val = train_test_split(fold_train_ids_all, test_size=0.2)
            # save all Id sets for reference 
            self.save_IDs(ids_train, ids_test, ids_val, self.model_checkpoint_dst)
            
            
            ids_train, ids_val = ids_train.index.to_list(), ids_val.index.to_list()
            ids_test = ids_test.index.to_list() 
            
            # -----------------------------------------------------------------
            # instantiate data Generators 
            self.train_Gen = dgb.get_data_generator(ids_train, validation=False)
            self.val_Gen = dgb.get_data_generator(ids_val,validation=True)
            # self.test_Gen = dgb.get_data_generator(ids_test, validation=True)

            # -----------------------------------------------------------------
            # define model , set optimiser and callbacks
            self.model = get_model(config) # get/define model 
            self.optimizer = self.get_optimizer(config) # define optimizer
            self.set_callbacks(config)            
    
            # -----------------------------------------------------------------
            # start training 
            # self.print_train_setting_info(ids_train, ids_eval, ids_test)  # print training information 
            self.train_model()
            
            # -----------------------------------------------------------------
            # evaluate the best model on K-fold Test set 
            print('-'*30)
            print('Testing model on K-fold test data N=' + str(len(ids_test)))
            mean_results = modelEvaluater.evaluateModel(config,
                                                        self.model_checkpoint_dst,
                                                        ids_test)  
            # save model mean result
            mean_results.name = str(k) + '-fold-'+ mean_results.name
            mean_metrics_test = mean_metrics_test.append(mean_results)  
            
            # -----------------------------------------------------------------
            # evaluate best model on external Test set 
            print('-'*30)
            print('Testing model on external test data N=' + str(len(ex_test_id)))
            mean_results_ex = modelEvaluater.evaluateModel(config,
                                                         self.model_checkpoint_dst,
                                                         ex_test_id, 
                                                         excel_name='Perfromance_TestSet_external.xlsx')  
            # save model mean result
            mean_results_ex.name = str(k) + '-fold-'+ mean_results_ex.name
            mean_results_exTest = mean_results_exTest.append(mean_results_ex)
            
            # -----------------------------------------------------------------
            # clear GPU mem 
            self.clear_mem()
        
        # save mean result over k folds of test sets and external test set 
        mean_metrics_test.to_excel(os.path.join(self.dst_path, 'OverallPerformance_TestSets.xlsx')) 
        mean_results_exTest.to_excel(os.path.join(self.dst_path, 'OverallPerformance_ExternalTestSet.xlsx')) 
  
        
        
        
    def get_train_test_evaluation_Id_lists(self, dataframe, train_IDs, test_IDs, dst=''):
        
        # ---------------------------------------------------------------------
        # extract testing IDs to list 
        testing_Ids_list = dataframe.iloc[test_IDs].squeeze()
        # ---------------------------------------------------------------------
        # extract all training IDs to list 
        training = dataframe.loc[train_IDs] 
        # extract first stritified sample as evaluation set
        evaluation_Ids_list = dataframe.loc[0] # select  
        training.drop(index=training.index[0], axis=0, inplace=True) # drop the evaluation row from training frame 
        # ---------------------------------------------------------------------
        # concatenate remaining training Ids to one list 
        training_Ids_list =[] # Create an empty list for training ids 
        for index, row in training.iterrows(): # Iterate over each row in training IDs
            my_list = row.to_list()
            training_Ids_list.append(my_list)
        training_Ids_list = list(itertools.chain.from_iterable(training_Ids_list)) # concatenate all lists 
        
        training_Ids_list = pd.Series(training_Ids_list)
        # save lists if path is given 
        if os.path.isdir(dst):
            training_Ids_list.to_excel(os.path.join(dst, 'training_ids.xlsx'))
            evaluation_Ids_list.to_excel(os.path.join(dst, 'validation_ids.xlsx'))
            testing_Ids_list.to_excel(os.path.join(dst, 'testing_ids.xlsx'))
        return training_Ids_list.tolist(), evaluation_Ids_list.tolist(), testing_Ids_list.tolist()
    
    
def get_cohort_data(spss_path, IDs):
    data = pd.read_spss(spss_path)
    s = data['subjectid']
    selected = data[data['subjectid'].isin(IDs)] # filter for imaging data 
    selected = selected.set_index('subjectid')
    selected = selected.reindex(IDs)    
    return selected


def get_number(string_el):
    temp = re.findall(r'\d+', string_el)    
    return int(temp[0])
    
    
from sklearn.model_selection import KFold

# =============================================================================
#                           **** Main **** 
# =============================================================================    
    
def main():
    
    
    # data_path = '/media/kwaygo/Backup Plus/Data/Extracted/SPRESTO/Women/Preconception/Precon - Intra - Retro - Peritoneal - Modelling - Data/Experiment - 2 - 60 samples/All - Volumes Ex1 - compressed'
    # data_path = "/media/kwaygo/Backup Plus/Data/Extracted/SPRESTO/Women/Preconception/Precon - Intra - Retro - Peritoneal - Modelling - Data/Experiment - 2 - 60 samples/All - Volumes Ex2 - compressed"
    # fold_id = define_kfold_sets_1(data_path, 5)
    # kf = KFold(n_splits=5)
    # index_gen = kf.split(fold_id.to_numpy())
    # for k, data in enumerate(index_gen,1):
    #     print('='*30)
    #     print('train', data[0])
    #     print('test', data[1])
    #     print('='*30)
    #     test = 10
    # test = 10
    
    print('='*50)
    print('Starting program ...')
    # load config file with training and model configurations  
    config = get_configurations()
    # reproducability setting 
    random.seed(config['numpy_seed'])
    np.random.seed(config['numpy_seed'])
    torch.manual_seed(config['pytroch_seed'])
    # torch.use_deterministic_algorithms(True)
    # create model trainer 
    modelTrainer = Trainer(config)
    if config['kfolds'] != 0:
        modelTrainer.train_kfold(config)
    else: # start training 
        modelTrainer.train()
    
    
if __name__ == '__main__':#
    main()
    
    




    