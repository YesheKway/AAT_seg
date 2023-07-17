#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 08:12:20 2021

@author: yeshe kway
"""
from ResNetUnet import get_model
import nibabel as nib
import numpy as np
import argparse
import random
import torch
import yaml
import re
import os


# =============================================================================
#                           Help Functions 
# =============================================================================


def get_configurations():
    parser = argparse.ArgumentParser(prog='TrainingParameters')
    parser.add_argument('-config',
                        '--config_path',
                        type=str, 
                        default='./configs/default-local.yaml', 
                        help='Configuration file defining training and evaluation parameters'
    )
    args = parser.parse_args()
    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['config_path'] = args.config_path
    config = process_config_args(config)    
    return config


def process_config_args(config):
    '''
    convert variables from config file to int of float, just to make sure
    that variables are castet correctly 
    '''
    config["n_output_channels"] = int(config["n_output_channels"])
    config["n_input_channels"]  = int(config["n_input_channels"])
    config["batch_size"]  = int(config["batch_size"])
    config["dropout"]  = float(config["dropout"])
    config["learning_rate"]  = float(config["learning_rate"])
    config["patience"]  = int(config["patience"])
    config["n_epochs"]  = int(config["n_epochs"])            
    return config

def define_start_end_random_patch(volume, input_dim):
    '''
    defines start and end number to extract a patch from a volume based in 
    model input shape 'input_dim'
    '''
    max_start_position = volume.shape[0]-input_dim[0]
    start = 0 if max_start_position<2 else random.randint(0, max_start_position-1)
    end = start + input_dim[0]    
    return start, end

def sort_dir_aphanumerical(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def findBestModelFile(main_dir):
     model_list = list()
     for file in os.listdir(main_dir):
         if file.endswith(".pt"):
             model_list.append(file)
     return sort_dir_aphanumerical(model_list)[-1]


# =============================================================================
#                           Load Functions
# =============================================================================

def load_Nifti(path):
    '''
    Load nifti volume and reorientates it 
    '''
    volume = np.moveaxis(np.rot90(nib.load(path).get_fdata()), -1, 0)
    return volume

def load_config(path):
    # --- find config file
    for file in os.listdir(path):
       if file.endswith(".yaml"):    
           config_file = file 
           break
    pathToconfig = os.path.join(path, config_file)
    # --- open/load config file 
    with open(pathToconfig) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config = process_config_args(config)    
    return config


def load_model(config, pathToModel, debug=False):
    # load model
    model = get_model(config)
    # load model state 
    if debug == False:
        best_model = findBestModelFile(pathToModel)
        print('Best model name: ' + best_model)    
        path_to_bestModel = os.path.join(pathToModel, best_model)
        model.load_state_dict(torch.load(path_to_bestModel))
    return model
    
