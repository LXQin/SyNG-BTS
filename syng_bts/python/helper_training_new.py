#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:14:37 2022

@author: yunhui, Cindi
"""

#%%

from .helper_utils_new import set_all_seeds, plot_recons_samples, plot_new_samples, plot_training_loss, plot_multiple_training_losses
from . import helper_train_new as ht
from .helper_models_new import AE, VAE, CVAE, GAN
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import copy
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter
import math
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import random_split


#%%
def training_AEs(savepath,             # path to save reconstructed samples
                 savepathnew,          # path to save newly generated samples
                 rawdata,              # raw data tensor with samples in row, features in column
                 rawlabels,            # labels for each sample, n_samples * 1, will not be used in AE or VAE
                 colnames,             # colnames saved
                 batch_size,           # batch size
                 random_seed,
                 modelname,            # choose from "AE","VAE","CVAE"
                 num_epochs,           # maxminum number of training epochs if early stopping does not triggered
                 learning_rate,        # learning rate
                 val_ratio = 0.2,      # validation ratio
                 pre_model = None,     # load pre-trained model from transfer learning
                 save_model = None,    # save model for transfer learning
                 kl_weight = 1,        # specify for VAE and CVAE
                 early_stop = True,    # whether or not using early stopping rule: best loss does not get improved in the future early_stop_num epochs.
                 early_stop_num = 30,  # stop training if loss does not improve for early_stop_num epochs
                 loss_fn = "MSE",      # choose from MSE or WMSE, do not use WMSE if you do not know the weights
                 save_recons = False,  # wheter to save the reconstructed data 
                 new_size = None,      # how many new samples you want to generate, for AE there is no new size so use None
                 save_new = False,     # whether to save the newly generated samples
                 plot = False,
                
                 use_scheduler = False,# scheduler parameters
                 step_size = 10,
                 gamma = 0.5):        # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones
        
    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    labels_squeezed = rawlabels.squeeze(1).long()  # shape: (n,)
    num_classes = len(torch.unique(labels_squeezed))
    data = TensorDataset(rawdata,rawlabels)
    
    if modelname == "CVAE":
        model = CVAE(num_features,num_classes)
        colnames = list(colnames) 
        colnames.append("groups")
    elif modelname == "VAE":
        model = VAE(num_features)
    elif modelname == "AE":
        model = AE(num_features)
    else: 
        raise ValueError("modelname is not supported by train_AEs funcion.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    # train_loader = DataLoader(data, batch_size = batch_size, shuffle=True, drop_last=True)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size
    
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False, drop_last = False)
    
    # transfer learning 
    if pre_model is not None:
        model.load_state_dict(torch.load(pre_model))
    
    if new_size is None:
        new_size = rawdata.shape[0]
    
    if modelname == "CVAE":
        log_dict, best_model = ht.train_CVAE(num_epochs = num_epochs,
                                             model = model,
                                             loss_fn = loss_fn,
                                             optimizer = optimizer, 
                                             train_loader = train_loader,
                                             val_loader = val_loader,
                                             early_stop = early_stop,
                                             early_stop_num = early_stop_num,
                                             skip_epoch_stats = True,
                                             reconstruction_term_weight = 1,
                                             kl_weight = kl_weight,
                                             logging_interval = 50,
                                             save_model = save_model)
        plot_training_loss(log_dict['val_reconstruction_loss_per_batch'], num_epochs, custom_label = " (reconstruction)")
        plt.show()
        plot_training_loss(log_dict['val_kl_loss_per_batch'], num_epochs, custom_label = " (KL)")
        plt.show()
        plot_training_loss(log_dict['val_combined_loss_per_batch'], num_epochs, custom_label = " (combined)")
        plt.show()
        
        if early_stop:
            if save_recons:
                # Plot generated data
                plot_recons_samples(savepath = savepath, data_loader = train_loader, model = best_model, n_features = num_features, modelname = "CVAE", plot = plot)       
                plt.show()
            if save_new:
                # plot and save new generated data
                plot_new_samples(model = best_model, savepathnew = savepathnew, latent_size = 32, modelname = "CVAE", num_images = new_size, plot = plot, colnames = colnames)
                plt.show()
        else:
            if save_recons:
                # Plot generated data
                plot_recons_samples(savepath = savepath, data_loader = train_loader, model = model, n_features = num_features, modelname = "CVAE", plot = plot)       
                plt.show()
            if save_new:
                # plot and save new generated data
                plot_new_samples(model = model, savepathnew = savepathnew, latent_size = 32, modelname = "CVAE", num_images = new_size, plot = plot, colnames = colnames)
                plt.show()
    elif modelname=="VAE":
        log_dict, best_model = ht.train_VAE(num_epochs = num_epochs,
                                         model = model,
                                         loss_fn = loss_fn,
                                         optimizer = optimizer, 
                                         train_loader = train_loader,
                                         early_stop = early_stop,
                                         early_stop_num = early_stop_num,
                                         skip_epoch_stats = True,
                                         reconstruction_term_weight = 1,
                                         kl_weight = kl_weight,
                                         logging_interval = 50,
                                         save_model = save_model,
                                         scheduler = scheduler)
        
        plot_training_loss(log_dict['train_reconstruction_loss_per_batch'], num_epochs, custom_label = " (reconstruction)")
        plt.show()
        plot_training_loss(log_dict['train_kl_loss_per_batch'], num_epochs, custom_label = " (KL)")
        plt.show()
        plot_training_loss(log_dict['train_combined_loss_per_batch'], num_epochs, custom_label = " (combined)")
        plt.show()
        
        if early_stop:
            if save_recons:
                # Plot generated data
                plot_recons_samples(savepath = savepath, data_loader = train_loader, model = best_model, n_features = num_features, modelname = "VAE", plot = plot)       
                plt.show()
            else:
                plot_recons_samples(savepath = None, data_loader = train_loader, model = best_model, n_features = num_features, modelname = "VAE", plot = plot)       
                plt.show()
            if save_new:
                # plot and save new generated data
                plot_new_samples(model = best_model, savepathnew = savepathnew, latent_size = 32, modelname = "VAE", num_images = new_size, plot = plot, colnames = colnames)
                plt.show()
            else:
                plot_new_samples(model = best_model, savepathnew = None, latent_size = 32, modelname = "VAE", num_images = new_size, plot = plot)
                plt.show()
        else:
            
            if save_recons:
                # Plot generated data
                plot_recons_samples(savepath = savepath, data_loader = train_loader, model = model, n_features = num_features, modelname = "VAE", plot = plot)       
                plt.show()
            else:
                plot_recons_samples(savepath = None, data_loader = train_loader, model = model, n_features = num_features, modelname = "VAE", plot = plot)       
                plt.show()
            if save_new:
                # plot and save new generated data
                plot_new_samples(model = model, savepathnew = savepathnew, latent_size = 32, modelname = "VAE", num_images = new_size, plot = plot, colnames = colnames)
                plt.show()
            else:
                plot_new_samples(model = model, savepathnew = None, latent_size = 32, modelname = "VAE", num_images = new_size, plot = plot)
                plt.show()
        
            
    else:
        log_dict, best_model = ht.train_AE(num_epochs = num_epochs,
                                        model = model,
                                        loss_fn = loss_fn,
                                        optimizer = optimizer, 
                                        train_loader = train_loader,
                                        early_stop = early_stop,
                                        early_stop_num = early_stop_num,
                                        skip_epoch_stats = True,
                                        logging_interval = 50,
                                        save_model = save_model)
        plot_training_loss(log_dict['train_loss_per_batch'], num_epochs, custom_label = " loss")
        plt.show()
        
        if early_stop :
            if save_recons:
                # Plot generated data
                plot_recons_samples(savepath = savepath, data_loader = train_loader, model = best_model, n_features = num_features, modelname = "AE", plot = plot)       
                plt.show()       
        else:
            
            if save_recons:
                # Plot generated data
                plot_recons_samples(savepath = savepath, data_loader = train_loader, model = model, n_features = num_features, modelname = "AE", plot = plot)       
                plt.show()
    return log_dict



def training_GANs(savepathnew,         # path to save newly generated samples
                  rawdata,             # raw data matrix with samples in row, features in column
                  rawlabels,           # labels for each sample, n_samples * 1, will not be used in AE or VAE
                  batch_size,          # batch size
                  random_seed,
                  modelname,           # choose from "GAN","WGAN","WGANGP"
                  num_epochs,          # maxminum number of training epochs if early stopping does not triggered
                  learning_rate,
                  new_size,            # how many new samples you want to generate
                  pre_model = None,    # load pre-trained model from transfer learning                  
                  save_model = None,   # save model for transfer learning
                  early_stop = True,   # whether or not using early stopping rule: best loss does not get improved in the future early_stop_num epochs.
                  early_stop_num = 30, # stop training if loss does not improve for early_stop_num epochs
                  save_new = False,    # whether to save the newly generated samples
                  plot = False):       # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones
    
    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    data = TensorDataset(rawdata,rawlabels)
    train_loader = DataLoader(data, batch_size = batch_size, shuffle=True, drop_last = True)
    latent_dim = 32

    model = GAN(num_features = num_features, latent_dim = latent_dim)

    optim_gen = torch.optim.Adam(model.generator.parameters(),
                                 betas=(0.5, 0.999),
                                 lr=learning_rate)
    optim_discr = torch.optim.Adam(model.discriminator.parameters(),
                                   betas=(0.5, 0.999),
                                   lr=learning_rate)
        # transfer learning 
    if pre_model is not None:
        model.load_state_dict(torch.load(pre_model))
 
    
    if modelname == "GAN":
        log_dict = ht.train_GAN(num_epochs = num_epochs,
                             model = model, 
                             optimizer_gen = optim_gen,
                             optimizer_discr = optim_discr,
                             latent_dim = latent_dim,
                             train_loader = train_loader,
                             early_stop = early_stop,
                             early_stop_num = early_stop_num,
                             logging_interval = 100,
                             save_model = save_model)
    elif modelname == "WGAN":
        log_dict, best_model = ht.train_WGAN(num_epochs = num_epochs,
                                          model = model, 
                                          optimizer_gen = optim_gen,
                                          optimizer_discr = optim_discr,
                                          latent_dim = latent_dim,
                                          train_loader = train_loader,
                                          early_stop = early_stop,
                                          early_stop_num = early_stop_num,
                                          logging_interval = 100,
                                          save_model = save_model)
    elif modelname == "WGANGP":
        log_dict, best_model = ht.train_WGANGP(num_epochs = num_epochs,
                                            model = model, 
                                            optimizer_gen = optim_gen, 
                                            optimizer_discr = optim_discr, 
                                            latent_dim = latent_dim,
                                            train_loader = train_loader,
                                            early_stop = early_stop,
                                            early_stop_num = early_stop_num,
                                            discr_iter_per_generator_iter = 5,
                                            logging_interval = 100, 
                                            gradient_penalty = True,
                                            gradient_penalty_weight = 10,
                                            save_model = save_model)
            
    plot_multiple_training_losses(
    losses_list=(log_dict['train_discriminator_loss_per_batch'],
                 log_dict['train_generator_loss_per_batch']),
    num_epochs=num_epochs, 
    custom_labels_list=(' -- Discriminator', ' -- Generator')
    )
    plt.show()
    
    if early_stop:
         if save_new:
             # plot and save new generated data
            plot_new_samples(model = best_model, savepathnew = savepathnew, latent_size = latent_dim, modelname = "GANs", num_images = new_size, plot = plot)
            plt.show()
         else:
            plot_new_samples(model = best_model, savepathnew = None, latent_size = latent_dim, modelname = "GANs", num_images = new_size, plot = plot)
            plt.show()
    else:
         if save_new:
             # plot and save new generated data
            plot_new_samples(model = model, savepathnew = savepathnew, latent_size = latent_dim, modelname = "GANs", num_images = new_size, plot = plot)
            plt.show()
         else:
            plot_new_samples(model = model, savepathnew = None, latent_size = latent_dim, modelname = "GANs", num_images = new_size, plot = plot)
            plt.show()        
        
   
    return log_dict
        
def training_iter(iter_times,          # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
                  savepathextend,      # save final (extended) dataset 
                  rawdata,             # pilot data
                  rawlabels,           # pilot labels
                  random_seed,
                  modelname,           # choose from AE, VAE
                  num_epochs,          # maxminum number of training epochs if early stopping does not triggered
                  batch_size,          # batch size
                  learning_rate,       # learning rate
                  early_stop = False,  # whether use early stopping rule
                  early_stop_num = 30, # training will stop if the loss does not improve for early_stop_num epochs
                  kl_weight = 1,       # only take effect for training VAE
                  loss_fn = "MSE",     # choose WMSE only if you know the weight, MSE by default
                  replace = False,     # whether to replace the failure features in each reconstruction
                  saveextend = False,  # whether to save the extended dataset, if True, savepathextend must be provided
                  plot = False):       # whether to plot the heatmaps of reconstructed dataset
    
    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    data = TensorDataset(rawdata,rawlabels)
    
    if modelname == "AE":
        model = AE(num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
        feed_data = rawdata
        feed_set = data
        for i in range(iter_times):
            # batch_size = round(feed_data.shape[0] * 0.1)
            feed_loader = DataLoader(feed_set, batch_size = batch_size, shuffle = True, drop_last = True)
            log_dict, best_model = ht.train_AE(num_epochs = num_epochs,
                                            model = model,
                                            loss_fn = loss_fn,
                                            optimizer = optimizer, 
                                            train_loader = feed_loader,
                                            early_stop = early_stop,
                                            early_stop_num = early_stop_num,
                                            skip_epoch_stats = True,
                                            logging_interval = 50,
                                            save_model = None)
            # Loss 
            plot_training_loss(log_dict['train_loss_per_batch'], num_epochs, custom_label = " (combined)")
            plt.show()
            if early_stop:
                feed_data_gen, feed_labels = plot_recons_samples(savepath = None, data_loader = feed_loader, model = best_model, n_features = num_features, modelname = "AE", plot = plot)
            else:
                feed_data_gen, feed_labels = plot_recons_samples(savepath = None, data_loader = feed_loader, model = model, n_features = num_features, modelname = "AE", plot = plot)
            print(feed_data_gen.shape)
            if replace:
                new_sample_range = range(int(feed_data_gen.shape[0]/2), feed_data_gen.shape[0])
                num_failures = 0
                for i_feature in range(feed_data_gen.shape[1]):
                    if( torch.std(feed_data_gen[new_sample_range,i_feature]) == 0 ) & ( torch.mean(feed_data_gen[new_sample_range,i_feature]) == 0 ):
                        feed_data_gen[new_sample_range,i_feature] = feed_data[:, i_feature]
                        num_failures += 1
                print("replace " + str(num_failures) +" zero features")
            feed_data = feed_data_gen
            feed_set = TensorDataset(feed_data, feed_labels)


    elif modelname=="VAE":
        model = VAE(num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        feed_data = rawdata
        feed_set = data
        for i in range(iter_times):
            batch_size = round(feed_data.shape[0] * 0.1)
            feed_loader = DataLoader(feed_set, batch_size = batch_size, shuffle = True, drop_last = True)
            log_dict, best_model = ht.train_VAE(num_epochs = num_epochs,
                                             model = model,
                                             loss_fn = loss_fn,
                                             optimizer = optimizer, 
                                             train_loader = feed_loader,
                                             early_stop = early_stop,
                                             early_stop_num = early_stop_num,
                                             skip_epoch_stats = True,
                                             reconstruction_term_weight = 1,
                                             kl_weight = kl_weight,
                                             logging_interval = 50,
                                             save_model = None)

            # Loss 
            plot_training_loss(log_dict['train_reconstruction_loss_per_batch'], num_epochs, custom_label = " (reconstruction)")
            plot_training_loss(log_dict['train_kl_loss_per_batch'], num_epochs, custom_label = " (KL)")
            plot_training_loss(log_dict['train_combined_loss_per_batch'], num_epochs, custom_label = " (combined)")
            plt.show()
            if early_stop:
                feed_data_gen, feed_labels = plot_recons_samples(savepath = None, data_loader = feed_loader, model = best_model, n_features = num_features, modelname = "VAE", plot = plot)     
            else:
                feed_data_gen, feed_labels = plot_recons_samples(savepath = None, data_loader = feed_loader, model = model, n_features = num_features, modelname = "VAE", plot = plot)     
            print(feed_data_gen.shape)
            if replace:
                new_sample_range = range(int(feed_data_gen.shape[0]/2), feed_data_gen.shape[0])
                num_failures = 0
                for i_feature in range(feed_data_gen.shape[1]):
                    if( torch.std(feed_data_gen[new_sample_range,i_feature]) == 0 ) & ( torch.mean(feed_data_gen[new_sample_range,i_feature]) == 0 ):
                        feed_data_gen[new_sample_range,i_feature] = feed_data[:, i_feature]
                        num_failures += 1
                print("replace " + str(num_failures) +" zero features")
            feed_data = feed_data_gen
            feed_set = TensorDataset(feed_data, feed_labels)
    if saveextend:
        np.savetxt(savepathextend, torch.cat((feed_data, feed_labels), dim = 1).detach().numpy(), delimiter = ",") 
        
    return feed_data, feed_labels

def training_flows(savepathnew, rawdata, batch_frac, valid_batch_frac, random_seed,
                   modelname,
                   num_blocks,
                   num_epoches,
                   learning_rate,
                   new_size,
                   num_hidden,
                   early_stop,  # whether use early stopping rule
                   early_stop_num,
                   # stop training if loss does not improve for early_stop_num epochs
                   pre_model,  # load pre-trained model from transfer learning
                   save_model=None,
                   plot=False):
    set_all_seeds(random_seed)
    device = torch.device("cpu")
    num_inputs = rawdata.shape[1]
    num_samples = rawdata.shape[0]

    # ## With validation
    #
    # N_validate = int(0.15 * num_samples)
    # valid_dataset = rawdata[-N_validate:]
    # train_dataset = rawdata[:-N_validate]
    # print(train_dataset)
    # valid_dataset = TensorDataset(valid_dataset)
    # train_dataset = TensorDataset(train_dataset)
    # train_batch_size = round(batch_frac * (num_samples - N_validate))
    # valid_batch_size = round(valid_batch_frac * N_validate)
    # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last = True)
    # valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, drop_last = True)

    ## Without validation
    train_dataset = TensorDataset(rawdata)
    train_batch_size = round(batch_frac * num_samples)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last = True)



    act = 'tanh'

    modules = []
    if modelname == 'glow':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                ht.BatchNormFlow(num_inputs),
                ht.LUInvertibleMM(num_inputs),
                ht.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs=None,
                    s_act='tanh', t_act='relu')
            ]
            mask = 1 - mask
    elif modelname == 'realnvp':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                ht.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs=None,
                    s_act='tanh', t_act='relu'),
                ht.BatchNormFlow(num_inputs)
            ]
            mask = 1 - mask
    elif modelname == 'maf':
        for _ in range(num_blocks):
            modules += [
                ht.MADE(num_inputs, num_hidden, num_cond_inputs=None, act=act),
                ht.BatchNormFlow(num_inputs),
                ht.Reverse(num_inputs)
            ]
    elif modelname == 'maf-split':
        for _ in range(num_blocks):
            modules += [
                ht.MADESplit(num_inputs, num_hidden, num_cond_inputs=None,
                              s_act='tanh', t_act='relu'),
                ht.BatchNormFlow(num_inputs),
                ht.Reverse(num_inputs)
            ]
    elif modelname == 'maf-split-glow':
        for _ in range(num_blocks):
            modules += [
                ht.MADESplit(num_inputs, num_hidden, num_cond_inputs=None,
                              s_act='tanh', t_act='relu'),
                ht.BatchNormFlow(num_inputs),
                ht.InvertibleMM(num_inputs)
            ]

    model = ht.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    # transfer learning
    if pre_model is not None:
        model.load_state_dict(torch.load(pre_model))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    writer = SummaryWriter(comment=modelname)
    global_step = 0

    def train(epoch, global_step, writer):
        # global global_step, writer
        model.train()
        train_loss = 0
        # samples = np.empty(shape=298)

        pbar = tqdm(total=len(train_loader.dataset))
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(device)
                else:
                    cond_data = None

                data = data[0]
            # import pdb
            # pdb.set_trace()
            data = data.to(device)
            optimizer.zero_grad()
            loss = -model.log_probs(data, cond_data).mean()
            train_loss += loss.item()
            # samples = np.vstack((samples, (model.sample(42).detach().numpy().reshape(42, -1))))
            # print(samples)
            loss.backward()
            optimizer.step()

            pbar.update(data.size(0))
            pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
                -train_loss / (batch_idx + 1)))

            writer.add_scalar('training/loss', loss.item(), global_step)
            global_step += 1


        # mysamples = pow(2, np.array(samples)) - 1
        # save_file_path = 'outputs/SKCM/MAF/epoch_%d.txt' % (global_step)
        # print((mysamples[1:453, ]).shape)
        # np.savetxt(save_file_path, mysamples[1:453, ])

        pbar.close()

        for module in model.modules():
            if isinstance(module, ht.BatchNormFlow):
                module.momentum = 0

            with torch.no_grad():
                model(train_loader.dataset.tensors[0].to(data.device))

        for module in model.modules():
            if isinstance(module, ht.BatchNormFlow):
                module.momentum = 1

        return global_step, train_loss/len(train_loader.dataset)

    def validate(epoch, model, loader, global_step, writer, prefix='Validation'):
        # global global_step, writer

        model.eval()
        val_loss = 0

        pbar = tqdm(total=len(loader.dataset))
        pbar.set_description('Eval')
        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(device)
                else:
                    cond_data = None

                data = data[0]
            data = data.to(device)
            with torch.no_grad():
                val_loss += -model.log_probs(data, cond_data, save=True,
                                             save_step=epoch).sum().item()  # sum up batch loss
            pbar.update(data.size(0))
            pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
                -val_loss / pbar.n))

        writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

        pbar.close()
        return val_loss / len(loader.dataset)

    # ## With validation version
    # best_validation_loss = float('inf')
    # best_validation_epoch = 0
    # best_model = model

    # Without validation version
    best_train_loss = float('inf')
    best_train_epoch = 0
    best_model = model

    for epoch in range(num_epoches):
        print('\nEpoch: {}'.format(epoch))

        global_step, train_loss = train(epoch, global_step, writer)



        # # With validation version
        # validation_loss = validate(epoch, model, valid_loader, global_step, writer)
        # if epoch - best_validation_epoch >= 30:
        #     break
        #
        # if validation_loss < best_validation_loss:
        #     best_validation_epoch = epoch
        #     best_validation_loss = validation_loss
        #     best_model = copy.deepcopy(model)
        #
        # print(
        #     'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        #         format(best_validation_epoch, -best_validation_loss))


        ## Without validation version

        # if epoch >= 70:
        #     plot_new_samples(model=model, savepathnew=savepathnew.replace('.csv', f'_epoch_{epoch}.csv'), latent_size=num_hidden, modelname=modelname,
        #                      num_images=new_size, plot=plot)
        if early_stop:
                if (epoch - best_train_epoch >= early_stop_num) or (np.isnan(train_loss)) or (math.isinf(train_loss)):
                    #import pdb
                    # pdb.set_trace()
                    break

        if (train_loss < best_train_loss) and not (math.isnan(train_loss)) and not (math.isinf(train_loss)):
            best_train_epoch = epoch
            best_train_loss = train_loss
            best_model = copy.deepcopy(model)


        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
                format(best_train_epoch, -best_train_loss))

    if save_model is not None:
        torch.save(best_model.state_dict(), save_model)
    # plot and save new generated data
    plot_new_samples(model=best_model, savepathnew=savepathnew, latent_size=num_hidden, modelname=modelname,
                     num_images=new_size, plot=plot)
    plt.show()