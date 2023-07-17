#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 07:35:33 2021
@author: Yeshe Kway
"""
import nibabel as nib
from pathlib import Path
from vis_help import plot_img, plot_masks, disp_raw_label
from apply_model import transformTensorForIMG
from Utils import load_Nifti, get_configurations
from ImageProcessing import (preprocess_X,
                             preprocess_y,
                             pad_Volume_aug,
                             preprocess_X_patch)

from torch.utils.data import DataLoader
import numpy as np
import torchio as tio
import torch
import random
import os
import csv
import re


# =============================================================================
#        Help function to unpack produced data during training load
# =============================================================================


def get_xy_data(data):
    X_data = data['mri']['data']
    y_data = data['seg']['data']
    return X_data, y_data


def get_xy_data_patch(data, input_normalization):
    X_data = data['mri']['data']
    X_data = preprocess_X_patch(X_data, input_normalization)
    y_data = data['seg']['data']
    y_data = custom_unpack_label_data_patch(y_data)
    return X_data, y_data


def unpack_label_data_tmp(labeldata, n_labels=7, include_dsat=False):
    '''
    This function extracts labels from the intra and retroperitioneal 
    segmentation mask of shaped (z, w, h) where labels have numbers from 0-7
    Note: label 1 (SSAT) and 2 (DSAT) are combined to form SAT 
    Output: will be in shape (n_labels, z, w, h)
    '''
    labels = list()
    for i in range(1, n_labels):
        if include_dsat:
            label_n = labeldata == i
            labels.append(label_n)
        else:
            if i == 2:
                tmp = labels[0] + labeldata == i
                label_n = tmp > 0
                labels.pop(0)
                labels.append(label_n)
            else:
                label_n = labeldata == i
                labels.append(label_n)
    # create a background label
    bg = 0
    for file in labels:
        bg += file
    bg = bg == 0
    labels.insert(0, bg)
    return np.stack(labels)


def custom_unpack_label_data_patch(tensor):
    '''
    This function extracts labels from the intra and retroperitioneal 
    segmentation mask of shaped (z, w, h) where labels have numbers from 0-7
    Note: label 1 (SSAT) and 2 (DSAT) are combined to form SAT 
    Output: will be in shape (n_labels, z, w, h)
    '''
    if is_label_mask(tensor):
        n_labels = 7
        labels = list()
        for i in range(1, n_labels):
            if i == 2:
                tmp = labels[0] + tensor == i
                label_n = tmp > 0
                labels.pop(0)
                labels.append(label_n)
            else:
                label_n = tensor == i
                labels.append(label_n)
        # create a background label
        bg = 0
        for file in labels:
            bg += file
        bg = bg == 0
        labels.insert(0, bg)
        ls = np.squeeze(np.stack(labels, 1))
        t = torch.tensor(ls, dtype=torch.double)
        return t
    else:
        return tensor

# =============================================================================
#               Function to create training and validation set  split
#               and save IDs as .csv list
# =============================================================================


def get_ID_lists(dataPath):
    '''
    Loads train and validatio ID list when saved from 'dataPath'
    Parameters
    ----------
    dataPath : TYPE
        DESCRIPTION.
    Returns
    -------
    trainIDs : TYPE
        DESCRIPTION.
    valIDs : TYPE
        DESCRIPTION.
    '''
    # validation_Ids = ('SPRESTO_10407', 'SPRESTO_10971', 'SPRESTO_10764', 'SPRESTO_10165')
    allIds = os.listdir(dataPath)
    validation_Ids = allIds
    train_ids = allIds
    # train_ids = set(allIds) - set(validation_Ids)
    # p_trainIDs = os.path.join(dataPath, 'training_IDs.csv')
    # p_valIDs = os.path.join(dataPath, 'validation_IDs.csv')
    # trainIDs = readCSVToList(p_trainIDs)
    # valIDs = readCSVToList(p_valIDs)
    return list(train_ids), list(validation_Ids)


def CreateTrainValLists():
    '''
    This fucntion extracts all the IDs from the files in a given path 
    and creates train and validation split, saving Ids in CSV 
    lists named 'training_Ids.csv' and 'validation_Ids.csv' respectively 
    '''
    pathToFiles = '//media/kwaygo/ymk_HDD1/Experiments/Kidney/extracted'
    dst = '//media/kwaygo/ymk_HDD1/Experiments/Kidney/'
    # train data ratio
    train_ratio = 80
    selectTrainValIDs(pathToFiles, dst, train_ratio, 100)


def selectTrainValIDs(path, dst, ratio=70, seed=30):
    '''
     this function takes a path to the training data and extracts the all 
     filenames and splits them by defined 'ratio' into train and 
     validation set. 
     Args: 
         :path :absolute path to data (raw data or label data)
         :ratio: spliting ratio (trainSetInPercentage) default 70         
         dst: where list will be saved to 
    '''
    filenames = os.listdir(path)
    nrOfFiles = len(filenames)
    random.Random(seed).shuffle(filenames)

    # split data into train and validations set
    split = int((ratio/100) * nrOfFiles)
    print("-splitting: " + str(nrOfFiles) + " Subjects" " (train:" +
          str(ratio)+"%" + " val:" + str(100-ratio)+"%)")
    names_train = filenames[:split]
    names_val = filenames[split:]
    print("   -->" + str(len(names_train)) + " training subjects")
    print("   -->" + str(len(names_val)) + " validation subjects")
    print("-save split infomration in CSV")

    # save validation subject ids in CSV
    writeListToCSV(extractIDs(names_train), "training_IDs", dst)
    # save training subject ids in CSV
    writeListToCSV(extractIDs(names_val), "validation_IDs", dst)


def writeListToCSV(lisT, csvName, dst):
    """
    this function saves a list into a CSV file and saves it in destination dst
    Args: 
        dst: destination where CSV should be saved to
        csvName: name of CSV file 
        lisT: type list         
    """
    filename = csvName + ".csv"
    with open(os.path.join(dst, filename), 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(lisT)
    print("CSV file saved as: " + csvName)


def extractIDs(self, lisT):
    """
    this function takes a list of strings and extracts only the numbers found in 
    the stings which are our IDs.
    """
    return list(map(lambda x: re.findall(r"\d+", x), lisT))


# =============================================================================
#                    Define Custom Transformation Fucnctions
# =============================================================================

count = 0
start = 0


def get_random_number(s):
    global count
    global start
    global end
    if count != 2:
        count += 1
        return start
    else:
        count = 1
        start = np.random.randint(0, s)
        return start


def define_start_end_random_patch(volume, input_dim):
    '''
    defines start and end number to extract a patch from a volume based in 
    model input shape 'input_dim'
    '''
    max_start_position = volume.shape[-1]-input_dim[0]
    if max_start_position < 2:
        start = 0
    else:
        start = get_random_number(max_start_position)
    end = start + input_dim[0]
    return start, end


def padd_volume(tensor):
    tensor = pad_Volume_aug(tensor, (320, 320, 40))  # padd input if needed
    tensor = torch.reshape(
        tensor, (1, tensor.shape[2], tensor.shape[3], tensor.shape[4]))
    return tensor


def custom_tran(tensor):
    tensor = torch.rot90(tensor, dims=(1, 2))
    start, end = define_start_end_random_patch(tensor, (40, 320, 320))
    t = tensor[..., start:end]
    t = padd_volume(t)
    return t


def custom_reshape(tensor):
    tensor = torch.moveaxis(tensor, -1, 1)
    return tensor


def is_label_mask(tensor):
    return False if len(torch.unique(tensor)) > 8 else True


my_trans = tio.Lambda(custom_tran)
custom_reshape_1 = tio.Lambda(custom_reshape)

# =============================================================================
#                       Define Transformations
# =============================================================================

# motion_or_ghost = {
#     tio.RandomGhosting(intensity=3, num_ghosts=30, axes=(1,)): 0.5,
#     tio.RandomMotion(degrees=10, translation=(0, 0.1), num_transforms=1): 0.5
# }

vol_transformation = {
    tio.RandomElasticDeformation(num_control_points=6,
                                 max_displacement=(20, 20, 0)): 0.4,
    tio.RandomAffine(degrees=0,
                     scales=(0.2, 0.2, 0),
                     translation=(0, 0.1)): 0.6
}

# aug_intensities = {
#     tio.RandomNoise():0.5,
#     tio.RandomBiasField(coefficients=0.2,  p=0.1):0.5
# }
# aug_artifacts = {
#     tio.RandomGhosting(intensity=1.5, num_ghosts=30, axes=(0,)) :0.5,
#     tio.RandomMotion(degrees=2, translation=3, num_transforms=1) : 0.5,
# }


# =============================================================================
#                           Data Gernerator
# =============================================================================


class Data_Generator_builder():

    def __init__(self, config):

        self.define_custom_transformations()
        self.shuffle_subjects = config['shuffle_subjects']
        self.shuffle_patches = config['shuffle_patches']
        self.samples_per_volume = config['samples_per_volume']
        self.validation_batch_size_factor = config['validation_batch_size_factor']
        self.max_queue_length = config['max_queue_length']
        self.patch_size = config['patch_size']
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.num_workers = config['n_workers']
        self.pathToData = config['data_path']
        self.x_file_name = config['x_file_name']
        self.y_file_name = config['y_file_name']
        self.input_dim = config['patch_dim']
        self.patch_dim = config['patch_dim']
        self.input_normalization = config['input_normalization']
        self.n_output_channels = config['n_output_channels']
        self.dsat = config['include_DSAT']

        # self.training_transform = tio.Compose([
        #     my_trans,
        #     # tio.OneOf(aug_intensities, p=0.2),
        #     # tio.OneOf(aug_artifacts, p=0.2),
        #     tio.RandomNoise(p=0.1),
        #     tio.RandomBiasField(coefficients=0.2,  p=0.2),
        #     # tio.OneOf(motion_or_ghost, p=0.3),
        #     tio.RandomGhosting(intensity=1.5, num_ghosts=30, axes=(0,), p=0.2),
        #     tio.RandomMotion(degrees=2, translation=3, num_transforms=1, p=0.2),
        #     tio.OneOf(vol_transformation, p=0.4),
        #     tio.RescaleIntensity((0, 1)),
        #     label_process,
        #     custom_reshape_1,
        # ])

        # ---------------------------------------------------------------------
        # Transformations applied during Training
        self.training_transform = tio.Compose([
            my_trans,
            # tio.OneOf(aug_intensities, p=0.2),
            # tio.OneOf(aug_artifacts, p=0.2),
            tio.RandomNoise(p=0.1),
            tio.RandomBiasField(coefficients=0.2,  p=0.1),
            # tio.OneOf(motion_or_ghost, p=0.3),
            tio.RandomGhosting(intensity=1.5, num_ghosts=30, axes=(0,), p=0.1),
            tio.RandomMotion(degrees=2, translation=3,
                              num_transforms=1, p=0.1),
            tio.OneOf(vol_transformation, p=0.2),
            tio.RescaleIntensity((0, 1)),
            self.label_process,
            custom_reshape_1,
        ])



        # self.training_transform = tio.Compose([
        #     my_trans,
        #     # tio.OneOf(aug_intensities, p=0.2),
        #     # tio.OneOf(aug_artifacts, p=0.2),
        #     # tio.RandomNoise(p=1),
        #     # tio.RandomBiasField(coefficients=0.2,  p=1),
        #     # tio.RandomGhosting(intensity=1.5, num_ghosts=30, axes=(0,), p=1),
        #     # tio.RandomMotion(degrees=2, translation=3, num_transforms=1, p=1),
        #     # tio.OneOf(vol_transformation, p=0.2),
            
        #     tio.RandomElasticDeformation(num_control_points=6, max_displacement=(20, 20, 0), p=1),
        #     # tio.RandomAffine(degrees=0, scales=(0.2, 0.2, 0), translation=(0, 0.1), p=1),
            
        #     tio.RescaleIntensity((0, 1)),
        #     self.label_process,
        #     custom_reshape_1,
        # ])




        # ---------------------------------------------------------------------
        # Transformations applied during validation
        self.validation_transform = tio.Compose([
            my_trans,
            tio.RescaleIntensity((0, 1)),
            self.label_process,
            custom_reshape_1,
        ])

        # ---------------------------------------------------------------------
        # Transformations applied during model application
        self.application_transform = tio.Compose([
            self.padd_patch,
            tio.RescaleIntensity((0, 1)),
            self.custom_reshape_app,
        ])

    def get_X_data(self, data):
        '''
        get model input data 
        '''
        X_data = []
        for file in self.x_file_name:
            X_data.append(data[file]['data'])
        X_data = torch.tensor(np.squeeze(np.stack(X_data, 1)),
                              dtype=torch.double)
        return X_data

    def get_xy_data(self, data, application=False):
        # ---------------------------------------------------------------------
        # get X data / Input data
        X_data = data['mri']['data']
        if application == True:
            return X_data
        # ---------------------------------------------------------------------
        # get lable data
        y_data = data['seg']['data']
        return X_data, y_data
    # =========================================================================
    # TIO transformation functions
    # =========================================================================

    def custom_reshape_appl(self, tensor):
        tensor = torch.moveaxis(tensor, -1, 1)
        tensor = np.rot90(np.squeeze(tensor), axes=(1, 2))
        tensor = torch.from_numpy(tensor.copy())
        tensor = torch.unsqueeze(tensor, dim=0)
        return tensor

    def padd_volume(self, tensor):
        '''
        only padds in width and hight
        '''
        # ---------------------------------------------------------------------
        # padd input if needed
        t_z = tensor.shape[-1]
        z = int(t_z) if t_z >= self.patch_dim[0] else self.patch_dim[0]
        tensor = pad_Volume_aug(tensor, (self.patch_dim[2],
                                         self.patch_dim[1],
                                         z))
        tensor = torch.reshape(
            tensor, (1, tensor.shape[2], tensor.shape[3], tensor.shape[4]))
        return tensor

    def define_custom_transformations(self):
        self.label_process = tio.Lambda(self.custom_unpack_label_data)
        # self.roi_extraction = tio.Lambda(self.patch_extraction)
        # self.custom_reshape_1 = tio.Lambda(self.custom_reshape)
        self.padd_patch = tio.Lambda(
            self.padd_volume)  # used in apply generator
        self.custom_reshape_app = tio.Lambda(self.custom_reshape_appl)

    def custom_unpack_label_data(self, tensor):
        '''
        This function extracts labels from the intra and retroperitioneal 
        segmentation mask of shaped (z, w, h) where labels have numbers from 0-7
        Note: label 1 (SSAT) and 2 (DSAT) are combined to form SAT 
        Output: will be in shape (n_labels, z, w, h)
        '''
        if is_label_mask(tensor):
            labels = list()
            for i in range(1, self.n_output_channels):
                if self.dsat:
                    label_n = tensor == i
                    labels.append(label_n)
                else:
                    if i == 2:
                        tmp = labels[0] + tensor == i
                        label_n = tmp > 0
                        labels.pop(0)
                        labels.append(label_n)
                    else:
                        label_n = tensor == i
                        labels.append(label_n)
            # create a background label
            bg = 0
            for file in labels:
                bg += file
            bg = bg == 0
            labels.insert(0, bg)
            ls = np.squeeze(np.stack(labels, 1))
            t = torch.tensor(ls, dtype=torch.double)
            return t
        else:
            return tensor

    def create_Torchio_sbjects(self, id_list):
        subjects = []
        for Id in id_list:
            # define paths
            subj_path = os.path.join(self.pathToData, Id)
            image_path = os.path.join(subj_path, self.x_file_name)
            label_path = os.path.join(subj_path, self.y_file_name)
            # create object
            subject = tio.Subject(
                mri=tio.ScalarImage(image_path),
                seg=tio.LabelMap(label_path)
            )
            subjects.append(subject)
        return subjects

    def get_train_val_generators_old(self):
        # create tio subjects
        training_subjects = self.create_Torchio_sbjects(self.train_ids)
        validation_subjects = self.create_Torchio_sbjects(self.val_ids)
        # create tio datasets
        training_dataset = tio.SubjectsDataset(training_subjects,
                                               transform=self.training_transform)
        validation_dataset = tio.SubjectsDataset(validation_subjects,
                                                 transform=self.validation_transform)
        # create troch dataloader
        train_gen = torch.utils.data.DataLoader(training_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=self.shuffle,
                                                num_workers=self.num_workers)
        val_gen = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              num_workers=self.num_workers)
        return train_gen, val_gen

    def get_data_generator(self, Id_list, validation=False):
        # ---------------------------------------------------------------------
        # there will be no augmentation transformation for evaluations and testing
        transformations = self.validation_transform if validation else self.training_transform
        # ---------------------------------------------------------------------
        # create tio subjects
        subjects = self.create_Torchio_sbjects(Id_list)
        # ---------------------------------------------------------------------
        # create tio datasets
        dataset = tio.SubjectsDataset(subjects,
                                      transform=transformations)
        # ---------------------------------------------------------------------
        # create troch dataloader
        data_gen = torch.utils.data.DataLoader(dataset,
                                               batch_size=self.batch_size,
                                               shuffle=self.shuffle,
                                               num_workers=self.num_workers)
        return data_gen

    def get_train_val_generators_patch(self):
        # create tio subjects
        training_subjects = self.create_Torchio_sbjects(self.train_ids)
        validation_subjects = self.create_Torchio_sbjects(self.val_ids)

        # --- create tio datasets

        validation_batch_size = self.validation_batch_size_factor * self.batch_size

        sampler = tio.data.UniformSampler(
            (self.input_dim[1], self.input_dim[2], self.input_dim[0]))

        training_dataset = tio.Queue(subjects_dataset=training_subjects,
                                     max_length=self.max_queue_length,
                                     samples_per_volume=self.samples_per_volume,
                                     sampler=sampler,
                                     num_workers=0,
                                     shuffle_subjects=self.shuffle_subjects,
                                     shuffle_patches=self.shuffle_patches,)
        validation_dataset = tio.Queue(subjects_dataset=validation_subjects,
                                       max_length=self.max_queue_length,
                                       samples_per_volume=self.samples_per_volume,
                                       sampler=sampler,
                                       num_workers=0,
                                       shuffle_subjects=self.shuffle_subjects,
                                       shuffle_patches=self.shuffle_patches,)
        # --- create troch dataloader
        train_gen = torch.utils.data.DataLoader(training_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=self.shuffle,
                                                num_workers=self.num_workers)
        val_gen = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=validation_batch_size,
                                              shuffle=self.shuffle,
                                              num_workers=self.num_workers)
        return train_gen, val_gen

    def get_data_generator_model_application(self, data_path, names, evaluation=False):
        '''
        This Data Genrator can be used to load volumes for model application
        (only preprocessing no additional transformations added)

        if arg: evaluation is set, then label data (name as defined in train config file) 
        will be loaded as well 
        '''
        # ---------------------------------------------------------------------
        # set data path
        self.pathToData = data_path
        # ---------------------------------------------------------------------
        # there will be no augmentation and patch extraction
        transformations = self.application_transform
        # ---------------------------------------------------------------------
        # create tio subjects
        subjects = self.create_Torchio_sbjects_applcation(names, evaluation)
        # ---------------------------------------------------------------------
        # create tio datasets
        dataset = tio.SubjectsDataset(subjects, transform=transformations)
        # ---------------------------------------------------------------------
        # create troch dataloader
        data_gen = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0)
        return data_gen

    def create_Torchio_sbjects_applcation(self, id_list, evalu=False):

        subjects = []
        for Id in id_list:
            # -----------------------------------------------------------------
            # define paths
            subj_path = os.path.join(self.pathToData, str(Id))
            # -----------------------------------------------------------------
            # create tio subject object
            if evalu:
                label_path = os.path.join(subj_path, self.y_file_name)
                subject = tio.Subject(mri=tio.ScalarImage(os.path.join(subj_path, self.x_file_name)),
                                      seg=tio.LabelMap(label_path))
            else:
                subject = tio.Subject(img1=tio.ScalarImage(
                    os.path.join(subj_path, self.x_file_name)))
            # -----------------------------------------------------------------
            # add to subject list
            subjects.append(subject)
        return subjects


# =============================================================================
#                               TMP
# =============================================================================

    def create_Torchio_sbjects_help(self, path):
        subjects = []
        for Id in os.listdir(path):
            # define paths
            subj_path = os.path.join(path, Id)
            image_path = os.path.join(subj_path, self.x_file_name)
            label_path = os.path.join(subj_path, self.y_file_name)
            # create object
            subject = tio.Subject(
                mri=tio.ScalarImage(image_path),
                seg=tio.LabelMap(label_path)
            )
            subjects.append(subject)

        # training_dataset = tio.SubjectsDataset(subjects,
        #                                        transform=self.training_transform)
        # train_gen = torch.utils.data.DataLoader(training_dataset,
        #                                              batch_size=self.batch_size,
        #                                              shuffle=self.shuffle,
        #                                              num_workers=self.num_workers)
        return subjects


# =============================================================================
#                           Test Data Generator
# =============================================================================


def torch_to_cpu(tensor):
    return np.squeeze(tensor.cpu().numpy())


def torch_tensor_to_nifti(tensor, affine, dst, name, mask=False):

    affine = torch_to_cpu(affine)
    tensor = torch_to_cpu(tensor)

    if mask:
        tensor = np.argmax(tensor, 0)

    # print('before ', affine)
    # print('after ', affine*np.eye(4))

    X = tensor
    print('='*30)
    print('Volume max', np.max(X))
    print('Volume min', np.min(X))
    print('Volume std', np.std(X))
    print('Volume mean', np.mean(X))
    print('='*30)

    r = transformTensorForIMG(X)
    new_image = nib.Nifti1Image(r.astype(np.float32), affine=affine)
    nib.save(new_image, os.path.join(dst, name))


def test_dataGen():
    # -------------------------------------------------------------------------
    # get config
    config = get_configurations()
    # -------------------------------------------------------------------------
    # define train and validatio lists
    train_ids, val_ids = get_ID_lists(config['data_path'])
    # -------------------------------------------------------------------------
    # create object
    dg = Data_Generator_builder(config)
    # -------------------------------------------------------------------------
    # patch generator
    # train_gen, val_gen = dg.get_train_val_generators_patch()
    # -------------------------------------------------------------------------
    # test normal generator
    train_gen = dg.get_data_generator(train_ids)
    #  get a sample
    train_batch = next(iter(train_gen))
    X, y = get_xy_data(train_batch)
    y = np.squeeze(y.cpu().numpy())
    y = np.moveaxis(y, 1, 0)
    X = np.squeeze(X.cpu().numpy())
    img = X
    disp_raw_label(img[0], y[0], 'Data_Gen')


def save_dataGen_samples():
    # -------------------------------------------------------------------------
    # get config
    config = get_configurations()
    # -------------------------------------------------------------------------
    # define train and validatio lists
    train_ids, val_ids = get_ID_lists(config['data_path'])
    # -------------------------------------------------------------------------
    # create object
    dg = Data_Generator_builder(config)
    # -------------------------------------------------------------------------
    # patch generator
    # train_gen, val_gen = dg.get_train_val_generators_patch()
    # -------------------------------------------------------------------------
    # test normal generator
    train_gen = dg.get_data_generator(train_ids)
    #  get a sample
    train_batch = next(iter(train_gen))

    X_data = train_batch['mri']

    sub_name = os.path.basename(str(Path(X_data['path'][0]).parents[0]))

    dst = r'/home/kwaygo/Documents/Projects/Torch/3D_AbdoFat-segmentation-kfold-training-EntireCohort/Augmentation_Examples/transformed'
    sub_path = os.path.join(dst, sub_name)
    os.mkdir(sub_path)

    t = np.fliplr(np.moveaxis(np.moveaxis(np.squeeze(X_data['data'].cpu().numpy()), 0, -1), 0, 1))
    im = nib.Nifti1Image(t, X_data['affine'][0].cpu().numpy())
    nib.save(im, os.path.join(sub_path, 'trans.nii.gz'))
    # y = np.squeeze(y.cpu().numpy())
    # y = np.moveaxis(y, 1, 0)
    # X = np.squeeze(X.cpu().numpy())
    # img = X
    # disp_raw_label(img[0], y[0], 'Data_Gen')


def test_application_gen():
    # -------------------------------------------------------------------------
    # get config
    config = get_configurations()
    # -------------------------------------------------------------------------
    # define train and validatio lists
    train_ids, test_IDs = get_ID_lists(config['data_path'])
    # -------------------------------------------------------------------------
    # create object
    dg = Data_Generator_builder(config)
    # -------------------------------------------------------------------------
    # application generator
    data_generator = dg.get_data_generator_model_application(config['data_path'],
                                                             test_IDs,
                                                             evaluation=True)
    #  get a sample
    train_batch = next(iter(data_generator))
    X, y = dg.get_xy_data(train_batch)
    y = np.squeeze(y.cpu().numpy())
    y = np.moveaxis(y, 1, 0)
    X = np.squeeze(X.cpu().numpy())
    img = X
    disp_raw_label(img[0], y[0], 'Data_Gen')


def main():

    print('test')
    # test_dataGen()

    save_dataGen_samples()

    # test_application_gen()
    # training_transform = tio.Compose([
    #   tio.RandomNoise(p=0.1),
    #   tio.RandomBiasField(coefficients=0.2,  p=0.1),
    #   # tio.OneOf(motion_or_ghost, p=0.3),
    #   tio.RandomGhosting(intensity=1.5, num_ghosts=30, axes=(0,), p=0.1),
    #   tio.RandomMotion(degrees=2, translation=3, num_transforms=1, p=0.1),
    #   tio.OneOf(vol_transformation, p=0.2),
    #   tio.RescaleIntensity((0, 1)),
    #   ])
    # training_transform


if __name__ == '__main__':
    main()
