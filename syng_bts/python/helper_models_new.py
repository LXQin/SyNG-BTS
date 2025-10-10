#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:37:49 2022

@author: yunhui
"""
#%%
import torch
import torch.nn as nn

#%%

class AE(nn.Module):

    def __init__(self, num_features):
        
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Linear(int(num_features), int(256)),
                nn.ReLU(True),
                nn.Linear(int(256), int(128)),
                nn.ReLU(True),
                nn.Linear(int(128), int(64)),
                nn.ReLU(True)
                )    
       
        self.decoder = nn.Sequential(
                nn.Linear(int(64), int(128)),
                nn.ReLU(True),
                nn.Linear(int(128), int(256)),
                nn.ReLU(True),
                nn.Linear(int(256), int(num_features)),                
                nn.ReLU(True)
                )
        

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
#%%

class VAE(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.encoder = nn.Sequential(
                nn.Linear(int(num_features), int(256)),
                nn.ReLU(True),
                nn.Linear(int(256), int(128)),
                nn.ReLU(True),
                nn.Linear(int(128), int(64)),
                nn.ReLU(True)
                )    
        
        self.z_mean = torch.nn.Linear(64, 32)
        self.z_log_var = torch.nn.Linear(64, 32)
        
        self.decoder = nn.Sequential(
                nn.Linear(int(32), int(64)),
                nn.ReLU(True),
                nn.Linear(int(64), int(128)),
                nn.ReLU(True),
                nn.Linear(int(128), int(256)),                
                nn.ReLU(True),
                nn.Linear(int(256), int(num_features)),                
                nn.ReLU(True)
                )
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        #.to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    
    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
    
    
#%%
class CVAE(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        # encoder input: features + label(dim = 1)
        # nn.BatchNorm1D() added
        self.encoder = nn.Sequential(
            nn.Linear(num_features + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )

        self.z_mean = nn.Linear(64, 32)
        self.z_log_var = nn.Linear(64, 32)

        self.decoder = nn.Sequential(
            nn.Linear(32 + 1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, num_features),
            nn.ReLU(True)
        )

    def reparameterize(self, z_mu, z_log_var, deterministic=False):
        if deterministic:
            return z_mu
        else:
            eps = torch.randn_like(z_mu)
            z = z_mu + eps * torch.exp(z_log_var / 2.)
            return z

    def encoding_fn(self, x, y, deterministic=False):
        x = self.encoder(torch.cat((x,y),dim=1))
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var, deterministic)
        return z_mean, z_log_var, encoded

    def decoding_fn(self, encoded, y):
        encoded = torch.cat((encoded, y),dim=1)
        decoded = self.decoder(encoded)
        return decoded

    def forward(self, x, y, deterministic=False):
        z_mean, z_log_var, encoded = self.encoding_fn(x, y, deterministic)
        decoded = self.decoding_fn(encoded, y)
        return encoded, z_mean, z_log_var, decoded


#%%
class GAN(torch.nn.Module):

    def __init__(self, num_features, latent_dim = 32):
        super().__init__()
        
        self.num_features = num_features
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, num_features),
            nn.ReLU(inplace = True)
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 1)
        )
    
    def generator_forward(self, z): # z is input low dimension noise
        img = self.generator(z)
        return img
    
    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits
