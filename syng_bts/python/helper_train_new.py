#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:04:48 2022

@author: yunhui, xinyi
"""

import torch
import time
import copy
import math
import types

import numpy as np
import scipy as sp
import scipy.linalg
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


#%%

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
    return curr_loss
    
def WMSE(output, target, reduction = "none"):
    weights = 1/(torch.mean(target,0)+1)
    loss = weights*(output - target)**2
    return loss

def calculate_latent_satistics(model, train_loader):
    model.eval()
    latent_means = defaultdict(list)
    latent_vars = defaultdict(list)

    with torch.no_grad():
        for features, lab in train_loader:
            if lab.dim() == 1:
                lab = lab.unsqueeze(1)
            lab = lab.float()

            _, z_mean, z_log_var, _ = model(features, lab, deterministic = True)
            z_var = torch.exp(z_log_var)

            for label_value in torch.unique(lab):
                mask = (lab[:, 0] == label_value)
                if mask.sum() > 0:
                    latent_means[label_value.item()].append(z_mean[mask])
                    latent_vars[label_value.item()].append(z_var[mask])

    epoch_latent_stats = {}
    for label_value in latent_means.keys():
        means_cat = torch.cat(latent_means[label_value], dim=0)
        vars_cat = torch.cat(latent_vars[label_value], dim=0)

        mu_y = means_cat.mean(dim=0)
        sigma_y = vars_cat.mean(dim=0).sqrt()

        epoch_latent_stats[label_value] = {
            "mu": mu_y,
            "sigma": sigma_y
        }
    return epoch_latent_stats

#%%
def train_AE(num_epochs, 
             model,
             optimizer, 
             train_loader,
             early_stop,
             early_stop_num,
             loss_fn = "MSE",
             logging_interval = 100, 
             skip_epoch_stats = False,
             save_model = None):
    
    log_dict = {'train_loss_per_batch': [],
                'train_combined_loss_per_epoch': []}

    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE
        

    start_time = time.time()
    best_loss = float('inf')
    best_epoch = 0
    best_model = model
    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()
        for batch_idx, (features,_) in enumerate(train_loader):
            # FORWARD AND BACK PROP
            encoded,  decoded = model(features)

            batchsize = features.shape[0]
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
                
            loss = pixelwise 
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
    
            loss.backward()
    
            # UPDATE MODEL PARAMETERS
            optimizer.step()
    
            # LOGGING
            log_dict['train_loss_per_batch'].append(pixelwise.item())                
                
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                         len(train_loader), loss))
    
        if not skip_epoch_stats:
            model.eval()
                
            with torch.set_grad_enabled(False):  # save memory during inference
                    
                train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                          epoch+1, num_epochs, train_loss))
                log_dict['train_combined_loss_per_epoch'].append(train_loss.item())
    
        train_loss = sum(epoch_loss) / len(epoch_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))     
        # for early stopping
        if early_stop & (epoch - best_epoch >= early_stop_num):
            print('Training for early stopping stops at epoch '+str(best_epoch) + " with best loss " + str(best_loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            break
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(best_model.state_dict(), save_model)
        
    return log_dict, best_model
#%%

def train_VAE(num_epochs, 
              model, 
              optimizer, 
              train_loader, 
              early_stop,
              early_stop_num,
              loss_fn = "MSE",
              logging_interval = 100, 
              skip_epoch_stats = False,
              reconstruction_term_weight = 1,
              kl_weight = 1,
              save_model = None,
              scheduler = None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': [],
                'latent_statistics_per_batch': []  # 新增
               }

    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE
        

    start_time = time.time()
    best_loss = float('inf')
    best_epoch = 0
    best_model = model
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr'] # Check Learning Rate (with scheduler)
        print(f"\n Epoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.6f}")
        
        epoch_loss = []
        model.train()
        for batch_idx, (features, lab) in enumerate(train_loader):
            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features)
                
            # total loss = reconstruction loss + KL divergence
            #kl_divergence = (0.5 * (z_mean**2 + 
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                  - z_mean**2 
                                  - torch.exp(z_log_var), 
                                  axis=1) # sum over latent dimension
    
            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
        
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
                
            loss = reconstruction_term_weight*pixelwise + kl_weight*kl_div
            epoch_loss.append(loss.item())
                
            optimizer.zero_grad()
    
            loss.backward()
    
            # UPDATE MODEL PARAMETERS
            optimizer.step()
    
            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            log_dict['latent_statistics_per_batch'].append({})
                
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                         len(train_loader), loss))

        if scheduler is not None:
            scheduler.step()
        
        if not skip_epoch_stats:
            model.eval()
                
            with torch.set_grad_enabled(False):  # save memory during inference
                    
                train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                          epoch+1, num_epochs, train_loss))
                log_dict['train_combined_loss_per_epoch'].append(train_loss.item())

        # ---------- 新增：统计 latent 分布 ----------

        # epoch_latent_stats = calculate_latent_satistics(model, train_loader)
        # log_dict['latent_statistics_per_batch'].pop()
        # log_dict['latent_statistics_per_batch'].append(epoch_latent_stats)

        # ---------- 继续训练流程 ----------
        
        train_loss = sum(epoch_loss) / len(epoch_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))     
        # for early stopping
        if early_stop & (epoch - best_epoch >= early_stop_num):
            print('Training for early stopping stops at epoch '+str(best_epoch) + " with best loss " + str(best_loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            break
            
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(best_model.state_dict(), save_model)
    
    return log_dict, best_model


#%%


def train_CVAE(num_epochs,
               model, 
               optimizer, 
               train_loader, 
               val_loader,
               early_stop,
               early_stop_num,
               loss_fn = "MSE",
               logging_interval = 100, 
               skip_epoch_stats = False,
               reconstruction_term_weight = 1,
               kl_weight = 1,
               save_model = None):
    
    log_dict = {'train_combined_loss_per_epoch': [],
                'train_combined_loss_per_batch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': [],
                'val_combined_loss_per_batch': [],
                'val_reconstruction_loss_per_batch': [],
                'val_kl_loss_per_batch': [],
                'latent_statistics_per_batch': [],  # 新增
               }

    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE

    start_time = time.time()
    best_loss = float('inf')
    best_epoch = 0
    best_model = model

    val_features, val_labels = next(iter(val_loader))
    if val_labels.dim() == 1:
        val_labels = val_labels.unsqueeze(1)
    val_labels = val_labels.float()
    
    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_val_loss = []
        for batch_idx, (features, lab) in enumerate(train_loader):
            model.train()
            if lab.dim() == 1: # label (dim = 1)
                lab = lab.unsqueeze(1)
            lab = lab.float()

                # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features,lab)
                # total loss = reconstruction loss + KL divergence
                #kl_divergence = (0.5 * (z_mean**2 + 
                #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                          - z_mean**2 
                                          - torch.exp(z_log_var), 
                                          axis=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
    
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
            
            train_loss = reconstruction_term_weight*pixelwise + kl_weight*kl_div
            epoch_train_loss.append(train_loss.item())

            optimizer.zero_grad()

            train_loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(train_loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())

            # --------- NEW: compute val loss after this batch ----------
            model.eval()
            with torch.no_grad():
                _, z_mean_val, z_log_var_val, decoded_val = model(val_features, val_labels, deterministic = True)
                val_pixelwise = loss_fn(decoded_val, val_features, reduction='none')
                val_pixelwise = val_pixelwise.view(val_features.size(0), -1).sum(axis=1)
                val_pixelwise = val_pixelwise.mean()

                kl_div_val = -0.5 * torch.sum(1 + z_log_var_val 
                                                - z_mean_val**2 
                                                - torch.exp(z_log_var_val),
                                                axis=1
                                             )
                kl_div_val = kl_div_val.mean()

                val_loss = (reconstruction_term_weight * val_pixelwise 
                                  + kl_weight * kl_div_val)

            epoch_val_loss.append(val_loss.item())

            # LOGGING
            log_dict['val_combined_loss_per_batch'].append(val_loss.item())
            log_dict['val_reconstruction_loss_per_batch'].append(val_pixelwise.item())
            log_dict['val_kl_loss_per_batch'].append(kl_div_val.item())

            log_dict['latent_statistics_per_batch'].append({})
            
            print('Epoch: %03d/%03d | Batch %04d/%04d | Val-Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                         len(train_loader), val_loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                 # Problem for CVAE: compute_epoch_loss_autoencoder() uses model(features), not for CVAE
                 train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn)
                 print('***Epoch: %03d/%03d | Loss: %.3f' % (
                        epoch+1, num_epochs, train_loss))
                 log_dict['train_combined_loss_per_epoch'].append(train_loss.item())

        # ---------- latent parameters ----------

        epoch_latent_stats = calculate_latent_satistics(model, train_loader)
        log_dict['latent_statistics_per_batch'].pop()
        log_dict['latent_statistics_per_batch'].append(epoch_latent_stats)

        # ---------- best model / early stop ----------
        
        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            
        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            print('Training for early stopping stops at epoch '+str(best_epoch) + " with best loss " + str(best_loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            break
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    for key in log_dict:
        log_dict[key].append(best_epoch)

    if save_model is not None:
        torch.save(best_model.state_dict(), save_model)
    
    return log_dict, best_model

#%% 
def train_GAN(num_epochs,
              model, 
              optimizer_gen,
              optimizer_discr, 
              latent_dim, 
              train_loader,
              early_stop = None,
              early_stop_num = None, # loss for GAN are not meaningful, so early stopping rule is not applied.
              logging_interval = 100, 
              save_model = None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': []}

    loss_fn = F.binary_cross_entropy_with_logits


    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)  
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size) # fake label = 0
            flipped_fake_labels = real_labels # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                  % (epoch+1, num_epochs, batch_idx, len(train_loader), gener_loss.item(), discr_loss.item()))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

#%%
def train_WGAN(num_epochs, 
               model, 
               optimizer_gen,
               optimizer_discr, 
               latent_dim, 
               train_loader, 
               early_stop,
               early_stop_num,
               logging_interval = 100, 
               save_model = None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': []}

    # if loss == 'regular':
    #     loss_fn = F.binary_cross_entropy_with_logits
    # elif loss == 'wasserstein':
    #     def loss_fn(y_pred, y_true):
    #         return -torch.mean(y_pred * y_true)
    # else:
    #     raise ValueError('This loss is not supported.')
    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    start_time = time.time()
    best_loss = float('inf')
    best_epoch = 0
    best_model = model
    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)  
            fake_images = model.generator_forward(noise)
            
            # if loss == 'regular':
            #     fake_labels = torch.zeros(batch_size) # fake label = 0
            # elif loss == 'wasserstein':
            #     fake_labels = -real_labels # fake label = -1    
            fake_labels = -real_labels # fake label = -1    
            flipped_fake_labels = real_labels # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()
            
            # if loss == 'wasserstein':
            #     for p in model.discriminator.parameters():
            #         p.data.clamp_(-0.01, 0.01)
            for p in model.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            epoch_loss.append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                  % (epoch+1, num_epochs, batch_idx, len(train_loader), gener_loss.item(), discr_loss.item()))


        train_loss = sum(epoch_loss) / len(epoch_loss)
        if (abs(train_loss) < abs(best_loss)) & (epoch >= 10):
            best_loss = train_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))     
        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            print('Training for early stopping stops at epoch '+str(best_epoch) + " with best loss " + str(best_loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            break
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict, best_model


def train_WGANGP(num_epochs, 
                 model, 
                 optimizer_gen, 
                 optimizer_discr, 
                 latent_dim, 
                 train_loader,
                 early_stop,
                 early_stop_num,
                 discr_iter_per_generator_iter = 5,
                 logging_interval = 100, 
                 gradient_penalty = True,
                 gradient_penalty_weight = 10,
                 save_model = None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': []}

    if gradient_penalty:
        log_dict['train_gradient_penalty_loss_per_batch'] = []

    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    start_time = time.time()
    
    
    skip_generator = 1
    best_loss = float('inf')
    best_epoch = 0
    best_model = model
    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()
        
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)  # format NCHW
            fake_images = model.generator_forward(noise)
            
            fake_labels = -real_labels # fake label = -1    
            flipped_fake_labels = real_labels # here, fake label = 1

    
            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()
            
            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)

            ###################################################
            # gradient penalty
            if gradient_penalty:

                alpha = torch.rand(batch_size, 1, 1, 1)

                interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                interpolated.requires_grad = True

                discr_out = model.discriminator_forward(interpolated)

                grad_values = torch.ones(discr_out.size())
                gradients = torch.autograd.grad(
                    outputs=discr_out,
                    inputs=interpolated,
                    grad_outputs=grad_values,
                    create_graph=True,
                    retain_graph=True)[0]

                gradients = gradients.view(batch_size, -1)

                # calc. norm of gradients, adding epsilon to prevent 0 values
                epsilon = 1e-13
                gradients_norm = torch.sqrt(
                    torch.sum(gradients ** 2, dim=1) + epsilon)

                gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * gradient_penalty_weight
                discr_loss += gp_penalty_term
                
                log_dict['train_gradient_penalty_loss_per_batch'].append(gp_penalty_term.item())
            #######################################################
            
            discr_loss.backward()

            optimizer_discr.step()
            
            # Use weight clipping (standard Wasserstein GAN)
            if not gradient_penalty:
                for p in model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            
            if skip_generator <= discr_iter_per_generator_iter:
                
                # --------------------------
                # Train Generator
                # --------------------------

                optimizer_gen.zero_grad()

                # get discriminator loss on fake images with flipped labels
                discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
                gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
                gener_loss.backward()

                optimizer_gen.step()
                
                skip_generator += 1
                
            else:
                skip_generator = 1
                gener_loss = torch.tensor(0.)

            # --------------------------
            # Logging
            # --------------------------   
            epoch_loss.append(discr_loss.item())
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                    % (epoch+1, num_epochs, batch_idx, len(train_loader), gener_loss.item(), discr_loss.item()))


        train_loss = sum(epoch_loss) / len(epoch_loss)
        if (abs(train_loss) < abs(best_loss)) & (epoch >= 10):
            best_loss = train_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))     
        # for early stopping
        if early_stop and (epoch - best_epoch >= early_stop_num):
            print('Training for early stopping stops at epoch '+str(best_epoch) + " with best loss " + str(best_loss))
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            break
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict, best_model
##%
## Flow models start here.....
def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        # import pdb
        # pdb.set_trace()
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADESplit(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 pre_exp_tanh=False):
        super(MADESplit, self).__init__()

        self.pre_exp_tanh = pre_exp_tanh

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

        input_mask = get_mask(num_inputs, num_hidden, num_inputs,
                              mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs, num_inputs,
                               mask_type='output')

        act_func = activations[s_act]
        self.s_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.s_trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs,
                                                   output_mask))

        act_func = activations[t_act]
        self.t_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.t_trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.s_joiner(inputs, cond_inputs)
            m = self.s_trunk(h)

            h = self.t_joiner(inputs, cond_inputs)
            a = self.t_trunk(h)

            if self.pre_exp_tanh:
                a = torch.tanh(a)

            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.s_joiner(x, cond_inputs)
                m = self.s_trunk(h)

                h = self.t_joiner(x, cond_inputs)
                a = self.t_trunk(h)

                if self.pre_exp_tanh:
                    a = torch.tanh(a)

                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                  nn.MaskedLinear(num_hidden, num_hidden,
                                                  hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))
        # self.trunk = nn.Sequential(act_func(),
        #                            nn.MaskedLinear(num_hidden, num_inputs * 2,
        #                                            output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            # import pdb
            # pdb.set_trace()
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1, keepdim=True)


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, 'inverse')
        else:
            return super(Logit, self).forward(inputs, 'direct')


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.register_buffer("perm", torch.randperm(num_inputs))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_inputs))
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_inputs))

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_inputs))
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_inputs))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        # import pdb
        # pdb.set_trace()
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet


        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None, save = None, save_step = None):
        u, log_jacob = self(inputs, cond_inputs)

        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)

        # if save:
        #     # import pdb
        #     # pdb.set_trace()
        #     save_path = 'outputs/SKCM'
        #     #u_npy = pow(2, u.numpy())-1
        #     u_npy = u.numpy()
        #     save_file_path = 'outputs/SKCM/epoch_%d.txt' % save_step
        #     np.savetxt(save_file_path, u_npy)
        #     # with open(save_file_path, 'w') as f:
        #     #     for i in range(h):
        #     #         for j in range(w):
        #     #             f.write(''str(log_probs_npy[i][j]))
        #     #         f.write('\n')
        #     # f.close()

        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples
