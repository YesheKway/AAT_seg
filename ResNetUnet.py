#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:14:10 2021

@author: yeshe kway
"""
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch 

from torchsummary import summary
from torchviz import make_dot
import hiddenlayer as hl

# =============================================================================
#                         DictionarysModule Dictionaries
# =============================================================================

def norm_layer(normalization):
    if normalization=='instance':
        return nn.InstanceNorm3d
    elif normalization=='batch':
        return nn.BatchNorm3d
    elif normalization=='none':
        return nn.Identity()


def activation_func(activation):
    return  nn.ModuleDict([
        ['softmax', nn.Softmax(dim=1)],
        ['sigmoid', nn.Sigmoid()],
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

# =============================================================================
#                           Additional Layers
# =============================================================================

# class Upsampling3D(nn.Module):
#     def __init__(self, in_shape, scale_factors, mode='trilinear'):
#         super().__init__()
#         new_shape = (in_shape[0] * scale_factors[0], in_shape[1] * scale_factors[1], in_shape[2] * scale_factors[2]) 
#         # x_data = torch.tensor([1, 2, 2])
#         # new_shape = torch.tensor(interpolation1(x).shape[-3:])*x_data
#         # print(torch.tensor(interpolation1(x).shape[-3:])*x_data)
#         # new_shape
#         self.ups = nn.Upsample(size=new_shape, mode=mode, align_corners=False)
#     def forward(self, x):
#         x = self.ups(x)
#         return x


class Upsampling3D(nn.Module):
    def __init__(self, scale_factors, mode='trilinear'):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=scale_factors, mode=mode, align_corners=False)
    def forward(self, x):
        x = self.ups(x)
        return x

# =============================================================================
#                           Convolutional Layers
# =============================================================================



class conv3D_norm(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, bias=True, activation='leaky_relu', normalization='instance'):
        super().__init__()
        
        self.activation = activation
        self.normalization = normalization
        
        self.conv = nn.Conv3d(in_c,
                          out_c,
                          kernel_size=kernel,
                          stride=stride,
                          padding=padding, 
                          bias=bias
        )
        self.norm = norm_layer(normalization)(out_c)
        self.act = activation_func(activation)
    
    def forward(self, x):
        x = self.conv(x)
        if self.normalization != 'none':
            x = self.norm(x)        
        if self.activation != 'none':
            x = self.act(x)
        return x

# =============================================================================
#                           Main Building Blocks 
# =============================================================================

class up_block(nn.Module):
    
    def __init__(self, in_c=0, out_c=0, kernel=3, sample_factors=(2,2,2), stride=1, padding=1, activation='leaky_relu', normalization='instance', upsampling='trilinear'):
        super().__init__()
        
        self.up_sample = Upsampling3D(scale_factors=sample_factors)
        self.conv = conv3D_norm(in_c,
                         out_c,
                         kernel=kernel,
                         stride=1,
                         padding=1, 
                         activation=activation,
                         normalization=normalization
        )        
        self.res_block = res_block(in_c,
                                   out_c,
                                   kernel=3, 
                                   stride=stride,
                                   padding=1,
                                   activation=activation,
                                   normalization=normalization
        )
        
        
    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        x = self.conv(x)
        x = torch.cat([x, skip_connection], axis=1)
        x = self.res_block(x)
        return x
    
    
class res_block(nn.Module):
    
    def __init__(self, in_c, out_c, kernel=3,  stride=1, padding=1, activation='leaky_relu', normalization='instance'):
        super().__init__()
        
        # residual conv (1x1x1)
        self.skip_conv = nn.Conv3d(in_c, out_c, kernel_size=1, padding=0, stride=stride)
        # conv norm activation
        self.conv_1 = conv3D_norm(in_c,
                         out_c,
                         kernel=kernel,
                         stride=stride,
                         padding=padding, 
                         activation=activation,
                         normalization=normalization)
        # conv norm without activation
        self.conv_2 = conv3D_norm(out_c,
                          out_c,
                          kernel=kernel,
                          stride=1,
                          padding=padding, 
                          activation='none',
                          normalization='none')
        self.normalization = norm_layer(normalization)(out_c)
        self.activation = activation_func(activation)
        

    def forward(self, x):
        # extract residual 
        residual = self.skip_conv(x)
        # conv norm activation  
        x = self.conv_1(x)
        # second conv norm 
        x = self.conv_2(x)
        # add residual to output and apply data normalization 
        # followed by activation
        x = self.activation(self.normalization(x+residual))
        return x

# =============================================================================
#                           Build Resnet Unet 
# =============================================================================

class ResNetU(nn.Module):
    
    def __init__(self, in_shape, in_c, n_filters, n_layers, activation='leaky_relu',
                 normalization='instance', final_activation='softmax', n_classes=1):
        super(ResNetU, self).__init__()
        list_kernels, list_strides = get_kernels_and_strides(in_shape, n_layers)
        self.layer_kernels = list_kernels
        self.strides = list_strides        
        self.n_layers = n_layers
        self.n_filters = n_filters
        
        # -------------------- define encoder ---------------------------------
        self.encoder_layers = list()
        for layer in range(self.n_layers):
            # define filter numbers
            out_c = 2**layer * n_filters
            in_c = in_c if layer==0 else int(out_c/2)
            strides = self.strides[layer]
            
            self.encoder_layers.append(res_block(in_c,
                                                out_c,
                                                kernel=3, 
                                                stride=strides,
                                                padding=1,
                                                activation=activation,
                                                normalization=normalization))
            
        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        # --------------------  bottle neck layer -----------------------------         
        # self.bottleneck = res_block(2**(n_layers-1)* n_filters,
        #                             2**(n_layers-1)* n_filters,
        #                             kernel=3, 
        #                             stride=1,
        #                             padding=1,
        #                             activation=activation,
        #                             normalization=normalization) 
        
        # -------------------- define decoder ---------------------------------
        self.decoder_layers = list()
        for layer in range(n_layers-1, 0, -1):
            # define filter numbers
            test_in_c= (2**layer * n_filters)
            test_out_c = int(test_in_c/2)
            all_strides = self.strides
            test_stride = self.strides[layer] #TODO
            self.decoder_layers.append(up_block(test_in_c,
                                                test_out_c,
                                                kernel=3, 
                                                sample_factors=self.strides[layer],
                                                padding=1,
                                                activation=activation,
                                                normalization=normalization)
        )
        self.decoder = torch.nn.Sequential(*self.decoder_layers)
        self.final_conv = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)
        self.final_act = activation_func(final_activation)
        
        
    def forward(self, inputs):
        x = inputs
        
        # ------------------ list to save skip connections --------------------
        skip_connections = list()
        # ------------------ build encoder ------------------------------------
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            # save skip connections to connect them with 
            # the decoder 
            if i < self.n_layers-1:
                skip_connections.append(x)
        # ------------------  build decoder -----------------------------------
        for i, decoder_block in enumerate(self.decoder):
            test = skip_connections[(i+1)*-1]           
            x = decoder_block(x, test)            
        # ---------------------final output -----------------------------------
        x = self.final_conv(x)        
        x = self.final_act(x)
        return x




class ResNetU_2(nn.Module):
    
    
    def __init__(self, in_shape, in_c, n_filters, n_layers, activation='leaky_relu',
                 normalization='instance', final_activation='softmax', n_classes=1):
        super(ResNetU_2, self).__init__()
        list_kernels, list_strides = get_kernels_and_strides(in_shape, n_layers)
        self.layer_kernels = list_kernels
        self.strides = list_strides        
        self.n_layers = n_layers
        self.n_filters = n_filters
        # -------------------- define encoder ---------------------------------
        self.encoder_layers = list()
        for layer in range(self.n_layers):
            # -----------------------------------------------------------------
            # define filter numbers stride and padding 
            out_c = 2**layer * n_filters
            in_c = in_c if layer==0 else int(out_c/2)
            strides = self.strides[layer]
            kernel = 5 if layer==0 else 5
            padding = 2 if layer==0 else 2
            # kernels = self.layer_kernels[layer]    
            
            self.encoder_layers.append(res_block(in_c,
                                                out_c,
                                                kernel=kernel, 
                                                stride=strides,
                                                padding=padding,
                                                activation=activation,
                                                normalization=normalization)
            )
        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        # --------------------  bottle neck layer -----------------------------         
        # self.bottleneck = res_block(2**(n_layers-1)* n_filters,
        #                             2**(n_layers-1)* n_filters,
        #                             kernel=3, 
        #                             stride=1,
        #                             padding=1,
        #                             activation=activation,
        #                             normalization=normalization) 
        
        # -------------------- define decoder ---------------------------------
        self.decoder_layers = list()
        for layer in range(n_layers-1, 0, -1):
            # define filter numbers
            test_in_c= (2**layer * n_filters)
            test_out_c = int(test_in_c/2)
            all_strides = self.strides
            test_stride = self.strides[layer] #TODO
            self.decoder_layers.append(up_block(test_in_c,
                                                test_out_c,
                                                kernel=3, 
                                                sample_factors=self.strides[layer],
                                                padding=1,
                                                activation=activation,
                                                normalization=normalization)
        )
        self.decoder = torch.nn.Sequential(*self.decoder_layers)
        self.final_conv = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)
        self.final_act = activation_func(final_activation)
        
        
    def forward(self, inputs):
        x = inputs
        
        # ------------------ list to save skip connections --------------------
        skip_connections = list()
        # ------------------ build encoder ------------------------------------
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            # save skip connections to connect them with 
            # the decoder 
            if i < self.n_layers-1:
                skip_connections.append(x)
        # ------------------  build decoder -----------------------------------
        for i, decoder_block in enumerate(self.decoder):
            test = skip_connections[(i+1)*-1]           
            x = decoder_block(x, test)            
        # ---------------------final output -----------------------------------
        x = self.final_conv(x)        
        x = self.final_act(x)
        return x




# n_layers = 3
# for l in range(n_layers):
#     print(l)
# print('='*30)
# for layer in range(n_layers-1, 0, -1):
#     print(layer)


def get_kernels_and_strides(volume_shape, n_layers, kernel_shape=(5, 5, 5), stride_shape=(2, 2, 2)):
    '''
    Parameters
    ----------
    volume_shape : TYPE
        DESCRIPTION. shape has to be (z, w, h)
    Returns
    -------
    list of kernels 
    '''
    z, w, h = volume_shape
    # w = w/2
    # h = h/2
    
    # ratio = z *(2**x) = w
    import math
    ratio = math.log((w/z), 2)
    
    # ratio = (w/z)/2
    kernels = list()
    strides = list()
    for l in range(1, n_layers+1):             
        z_kernel = kernel_shape[0] if l>ratio else 1
        kernel = (z_kernel, kernel_shape[0], kernel_shape[0])
        # print("kernel tuple: %s" % (kernel,))
        z_stride = stride_shape[0] if l>ratio else 1
        stride = (z_stride, stride_shape[0], stride_shape[0])
        # print("stride tuple: %s" % (stride,))
        kernels.append(kernel)
        strides.append(stride)
    # first resblock will just extract features no stride 2 for downsampling         
    strides.insert(0, (1,1,1)) 
    kernels.insert(0, (5,5,5))
    # print('stride list: ')
    # print(strides)        
    return kernels, strides


def vis_model(model):
    summary(model, input_size=(3, 128, 128, 128), device='cpu')
    in_ = torch.randn(1, 3, 128, 128, 128)
    o_m = model(in_)
    # res = make_dot(o_m)
    # res.render(filename='g1.dot')
    import hiddenlayer as hl
    r = hl.build_graph(model, torch.zeros([1, 3, 128, 128, 128]))
    tets = 10


def print_model_parameters(model, input_shape=(1, 40, 320, 320)):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tmp = torch.randn(1, 1, 40, 320, 320)
    # o_m = model(tmp)
    # print('model out: ' + str(o_m.shape))
    
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} total trainable parameters.")    
    # del model
    # del tmp
    # del o_m
    
    
def get_model(configs):
    
    # if DSAT is included we will keep output dim
    # if DSAT is excluded total SAT will be segmented and  SSAT and DSAT will be combined 
    if configs['include_DSAT']:
        out_channels = configs['n_output_channels']
    else:
        out_channels = configs['n_output_channels'] - 1
    
    # define model input parameters 
    model_params = {'in_shape':         configs['patch_dim'], 
                    'in_c':             configs['n_input_channels'],
                    'n_filters':        configs['n_filter'],
                    'n_layers':         configs['n_layer'],
                    'activation':       configs['filter_activation'],
                    'normalization':    configs['normalization_type'],
                    'final_activation': configs['final_activation'],
                    'n_classes':        out_channels
    }
    # get model 
    if configs['model_type'] == 'dynamic_unet_3D':
        model = ResNetU(**model_params) 
    if configs['model_type'] == 'dynamic_unet_3D_5x5':
        model = ResNetU_2(**model_params) 
    return model    
    
    
# =============================================================================
#                   sanity checks for layer components 
# =============================================================================


def check_conv3D_norm():
    print('start')
    in_ = torch.randn(1, 3, 128, 128, 128)
    c = conv3D_norm(3, 16, stride=2)
    o = c(in_)
    print(o.shape)


def test_up_block():
    # sample input     
    in_ = torch.randn(1, 64, 40, 40, 40)
    # sample skip connection 
    s = torch.randn(1, 32, 40, 80, 80)
    s_2 = torch.randn(1, 16, 40, 160, 160)
    # perform 2 upsampling layers 
    up_layer = up_block(64, 32, 3, sample_factors=(1, 2, 2))
    up_layer_2 = up_block(32, 16, 3, sample_factors=(1, 2, 2))
    out = up_layer(in_,s)
    out2 = up_layer_2(out, s_2)
    print('input layer shape: '), print(in_.shape)
    print('first upsampling: '), print(out.shape)
    print('first upsampling: '), print(out2.shape)
    # print(up_layer)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*50)
    print('Device selected: ', device)
    return device


def main():    
    # -------------------- implementation checks ------------------------------
    # check_moduleList()
    # test_up_block()
    kernels = get_kernels_and_strides(volume_shape=(80, 320, 320), n_layers=6)

    model = ResNetU(in_shape=(40, 320, 320), in_c=1, n_filters=4, n_layers=6)
    # in_ = torch.randn(1, 1, 40, 320, 320)
    # model
    # in_
    # o_m = model(in_)
    # print('='*30)
    # print('model out: ' + str(o_m.shape))
    # print('='*30)
    # print(model)
    
    # # print_model_parameters(model)
    # torch.cuda.empty_cache()
    
    # # path = '/home/kwaygo/Documents/Projects/Torch/trainedModels/model.pth'
    # # torch.save(model.state_dict(), path)

    summary(model, input_size=(1, 40, 320, 320), device='cpu')
    
    # # r = hl.build_graph(model, torch.zeros(1, 3, 32, 256, 256))
    # # r.save('/home/kwaygo/Documents/Projects/Torch/3D_segmentation/model_viz/model_2.pdf')
    # test = 0
    
    
    
if __name__ == '__main__':
    main()


# Total params: 1,172,481
# Trainable params: 1,172,481
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 15.62
# Forward/backward pass size (MB): 23062.50
# Params size (MB): 4.47
# Estimated Total Size (MB): 23082.60
# ----------------------------------------------------------------

