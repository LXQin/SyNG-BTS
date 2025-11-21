#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:22:40 2022

@author: yunhui, xinyi
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import torch.nn.functional as F


def preprocessinglog2(dataset):
    # log2 pre-processing of count data
    return torch.log2(dataset + 1)


def set_all_seeds(seed):
    # set random seed
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_labels(n_samples, groups=None):
    # create binary labels and blurry labels for training two-group data
    set_all_seeds(10)  # randomness only for blur labels generation.
    if groups is None:
        labels = torch.zeros([n_samples, 1])
        blurlabels = labels
    else:
        base = groups[0]
        labels = torch.zeros([n_samples, 1]).to(torch.float32)
        labels[groups != base, 0] = 1
        blurlabels = torch.zeros([n_samples, 1]).to(torch.float32)
        blurlabels[groups != base, 0] = (10 - 9) * torch.rand(sum(groups != base)) + 9
        blurlabels[groups == base, 0] = (1 - 0) * torch.rand(sum(groups == base)) + 0
    return labels, blurlabels
    
def create_labels_mul(n_samples, groups=None):
    set_all_seeds(10)

    if groups is None:
        labels = torch.zeros([n_samples, 1], dtype=torch.float32)
        blurlabels = labels.clone()
        return labels, blurlabels

    groups_cat = groups.astype("category")
    codes = groups_cat.cat.codes
    group_tensor = torch.from_numpy(codes.copy().values)
    labels = group_tensor.float().unsqueeze(1) 
    blurlabels = labels + torch.rand_like(labels)
    return labels, blurlabels

def draw_pilot(dataset, labels, blurlabels, n_pilot, seednum):
    # draw pilot datasets
    set_all_seeds(
        seednum
    )  # each draw has its own seednum, so guaranteed that 25 replicated sets are not the same
    n_samples = dataset.shape[0]
    if torch.unique(labels).shape[0] == 1:
        shuffled_indices = torch.randperm(n_samples)
        pilot_indices = shuffled_indices[-n_pilot:]
        rawdata = dataset[pilot_indices, :]
        rawlabels = labels[pilot_indices, :]
        rawblurlabels = blurlabels[pilot_indices, :]
    else:
        base = labels[0, :]
        n_pilot_1 = n_pilot
        n_pilot_2 = n_pilot
        n_samples_1 = sum(labels[:, 0] == base)
        n_samples_2 = sum(labels[:, 0] != base)
        dataset_1 = dataset[labels[:, 0] == base, :]
        dataset_2 = dataset[labels[:, 0] != base, :]
        labels_1 = labels[labels[:, 0] == base, :]
        labels_2 = labels[labels[:, 0] != base, :]
        blurlabels_1 = blurlabels[labels[:, 0] == base, :]
        blurlabels_2 = blurlabels[labels[:, 0] != base, :]
        shuffled_indices_1 = torch.randperm(n_samples_1)
        pilot_indices_1 = shuffled_indices_1[-n_pilot_1:]
        rawdata_1 = dataset_1[pilot_indices_1, :]
        rawlabels_1 = labels_1[pilot_indices_1, :]
        rawblurlabels_1 = blurlabels_1[pilot_indices_1, :]
        shuffled_indices_2 = torch.randperm(n_samples_2)
        pilot_indices_2 = shuffled_indices_2[-n_pilot_2:]
        rawdata_2 = dataset_2[pilot_indices_2, :]
        rawlabels_2 = labels_2[pilot_indices_2, :]
        rawblurlabels_2 = blurlabels_2[pilot_indices_2, :]
        rawdata = torch.cat((rawdata_1, rawdata_2), dim=0)
        rawlabels = torch.cat((rawlabels_1, rawlabels_2), dim=0)
        rawblurlabels = torch.cat((rawblurlabels_1, rawblurlabels_2), dim=0)
    return rawdata, rawlabels, rawblurlabels


def Gaussian_aug(rawdata, rawlabels, multiplier):
    # Gaussian augmentation
    # This function performs offline augmentation by adding gaussian noise to the
    # log2 counts, rawdata is the data generated from draw_pilot(), so does rawlabels,
    # multiplier specifies the number of samples for each kind of label, must be a list if
    # unique labels > 1. This function generates rawdata and rawlabels again but with
    # gaussian augmented data with size multiplier*n_rawdata

    oriraw = rawdata
    orirawlabels = rawlabels
    for all_mult in multiplier:
        for mult in list(range(all_mult)):
            rawdata = torch.cat(
                (
                    rawdata,
                    oriraw
                    + torch.normal(
                        mean=0, std=1, size=(oriraw.shape[0], oriraw.shape[1])
                    ),
                ),
                dim=0,
            )
            rawlabels = torch.cat((rawlabels, orirawlabels), dim=0)

    return rawdata, rawlabels


def plot_training_loss(
    minibatch_losses, num_epochs, averaging_iterations=100, custom_label=""
):
    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        range(len(minibatch_losses)),
        (minibatch_losses),
        label=f"Minibatch Loss{custom_label}",
    )
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")

    if len(minibatch_losses) < 1001:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([0, np.max(minibatch_losses[num_losses:]) * 1.5])

    ax1.plot(
        np.convolve(
            minibatch_losses,
            np.ones(
                averaging_iterations,
            )
            / averaging_iterations,
            mode="valid",
        ),
        label=f"Running Average{custom_label}",
    )
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


def plot_recons_samples(
    savepath, model, modelname, data_loader, n_features, plot=False
):
    # plot reconstructed samples heatmap and save reconstructed samples as .csv file

    orig_all = torch.zeros([1, n_features])
    decoded_all = torch.zeros([1, n_features])
    labels = torch.zeros(0, dtype=torch.long)

    for batch_idx, (features, lab) in enumerate(data_loader):
        # compatible with two types of labels:
        # - (N, C) one-hot -> use argmax
        # - (N, 1) single column/real number 0/1 -> directly squeeze to (N,)
        if isinstance(lab, torch.Tensor):
            if lab.dim() == 2:
                if lab.size(1) > 1:
                    labels_batch = torch.argmax(lab, dim=1)  # shape: (batch_size,)
                else:
                    labels_batch = lab.squeeze(1).long()
            else:
                labels_batch = lab.long()
        else:
            # revert: convert non-tensor labels to tensor
            labels_batch = torch.as_tensor(lab).long()
        labels = torch.cat((labels, labels_batch), dim=0)

        with torch.no_grad():
            if modelname == "CVAE":
                encoded, z_mean, z_log_var, decoded_images = model(features, lab)
            elif modelname == "VAE":
                encoded, z_mean, z_log_var, decoded_images = model(features)
            else:
                encoded, decoded_images = model(features)

        orig_all = torch.cat((orig_all, features), dim=0)
        decoded_all = torch.cat((decoded_all, decoded_images), dim=0)

    orig_all = orig_all[1:]
    decoded_all = decoded_all[1:]
    
    if modelname == "CVAE":
        labels = labels.unsqueeze(1).float()  # shape: (N,1)
        orig_all = torch.cat((orig_all, labels), dim=1)
        decoded_all = torch.cat((decoded_all, labels), dim=1)
    if plot:
        sns.heatmap(
            torch.cat((orig_all, decoded_all), dim=0).detach().numpy(), cmap="YlGnBu"
        )
        plt.show()

    if savepath is not None:
        components = savepath.split("/")
        directory = "/".join(savepath.split("/")[:2])
        os.makedirs(directory, exist_ok=True)
        for i in range(2, len(components) - 1):
            directory = directory + "/" + components[i]
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)

        file_path = savepath
        # used to be savepath instead of file_path
        np.savetxt(
            file_path,
            torch.cat((orig_all, decoded_all), dim=0).detach().numpy(),
            delimiter=",",
        )
    else:
        return torch.cat((orig_all, decoded_all), dim=0).detach(), labels


# def plot_latent_space_with_labels(num_classes, data_loader, encoding_fn):
#     d = {i:[] for i in range(num_classes)}

#     with torch.no_grad():
#         for i, (features,targets) in enumerate(data_loader):
#             embedding = encoding_fn(features)
#             for i in range(num_classes):
#                 if i in targets:
#                     mask = targets == i
#                     d[i].append(embedding[mask].numpy())

#     colors = list(mcolors.TABLEAU_COLORS.items())
#     for i in range(num_classes):
#         d[i] = np.concatenate(d[i])
#         plt.scatter(
#             d[i][:, 0], d[i][:, 1],
#             color=colors[i][1],
#             label=f'{i}',
#             alpha=0.5)

#     plt.legend()


def plot_new_samples(
    model, modelname, savepathnew, latent_size, num_images, plot=False, colnames = None
):
    # plot new samples heatmap and save new samples as .csv file

    with torch.no_grad():
        ##########################
        ###### RANDOM SAMPLE #####
        ##########################
        if len(num_images) == 1:
            num_images = num_images[0]
            rand_features = torch.randn(num_images, latent_size)
            if modelname == "CVAE":
                num_classes = model.num_classes
                base = num_images // num_classes
                rem = num_images % num_classes
                counts = [base]*num_classes
                for i in range(rem):
                    counts[i] += 1
                labels_list = []
                for class_id, n_c in enumerate(counts):
                    ids = torch.full((n_c,), fill_value=class_id, dtype=torch.float32)
                    labels_list.append(ids)
                one_group_labels = torch.cat(labels_list)
                labels = one_group_labels.unsqueeze(1)  # shape = [N, 1]
                
                rand_features = torch.cat((rand_features, labels), dim=1)
                new_images = model.decoder(rand_features)
                new_images = torch.cat((new_images, labels), dim=1)
            elif modelname == "AE":
                new_images = model.decoder(rand_features)
            elif modelname == "VAE":
                new_images = model.decoder(rand_features)
            elif modelname == "GANs":
                new_images = model.generator(rand_features)
            elif modelname == "glow":
                new_images = model.sample(num_images)
            elif modelname == "realnvp":
                new_images = model.sample(num_images)
            elif modelname == "maf":
                new_images = model.sample(num_images)
        else:
            # if new_size = num_images = [n_for_0, n_for_1, ... , n_for_(num_classes-1), replicate]
            counts = num_images[:-1]
            repli = num_images[-1]
            num_images_repe = sum(counts)
            num_images = num_images_repe * repli
            rand_features = torch.randn(num_images, latent_size)
            if modelname == "CVAE":
                if len(num_images) != num_classes + 1:
                    raise ValueError("num_images should have length num_classes+1")

                labels_list = []
                for class_id, n_c in enumerate(counts):
                    ids = torch.full((n_c,), fill_value=class_id, dtype=torch.float32)
                    labels_list.append(ids)
                one_group_labels = torch.cat(labels_list)
                labels = one_group_labels.repeat(repli).unsqueeze(1)  # shape = [N, 1]

                rand_features = torch.cat((rand_features, labels), dim=1)
                new_images = model.decoder(rand_features)
                new_images = torch.cat((new_images, labels), dim=1)
            elif modelname == "AE":
                new_images = model.decoder(rand_features)
            elif modelname == "VAE":
                new_images = model.decoder(rand_features)
            elif modelname == "GANs":
                new_images = model.generator(rand_features)
            elif modelname == "glow":
                new_images = model.sample(num_images)
            elif modelname == "realnvp":
                new_images = model.sample(num_images)
            elif modelname == "maf":
                new_images = model.sample(num_images)

        ##########################
        ### VISUALIZATION
        ##########################
        # last column of saved data is the labels: 0 for MXF, 20 for PMFH
        # either generated for VAE or setted for CVAE
        if plot:
            sns.heatmap(new_images.detach().numpy(), cmap="YlGnBu")
            plt.show()

        if savepathnew is not None:
            if isinstance(savepathnew, Path):
                path_components = savepathnew.parts
            else:
                path_components = str(savepathnew).split("/")
            directory = "/".join(path_components[:2])
            os.makedirs(directory, exist_ok=True)
            for i in range(2, len(path_components) - 1):
                directory = directory + "/" + path_components[i]
                os.makedirs(directory, exist_ok=True)
                print("Directory created: " + directory)
            # used to be savepath instead of file_path
            np.savetxt(savepathnew, new_images.detach().numpy(), delimiter=",")
        else:
            return new_images


def plot_multiple_training_losses(
    losses_list, num_epochs, averaging_iterations=100, custom_labels_list=None
):
    for i, _ in enumerate(losses_list):
        if not len(losses_list[i]) == len(losses_list[0]):
            raise ValueError(
                "All loss tensors need to have the same number of elements."
            )

    if custom_labels_list is None:
        custom_labels_list = [str(i) for i, _ in enumerate(custom_labels_list)]

    iter_per_epoch = len(losses_list[0]) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    for i, minibatch_loss_tensor in enumerate(losses_list):
        ax1.plot(
            range(len(minibatch_loss_tensor)),
            (minibatch_loss_tensor),
            label=f"Minibatch Loss{custom_labels_list[i]}",
        )
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")

        ax1.plot(
            np.convolve(
                minibatch_loss_tensor,
                np.ones(
                    averaging_iterations,
                )
                / averaging_iterations,
                mode="valid",
            ),
            color="black",
        )

    if len(losses_list[0]) < 1000:
        num_losses = len(losses_list[0]) // 2
    else:
        num_losses = 1000
    maxes = [np.max(losses_list[i][num_losses:]) for i, _ in enumerate(losses_list)]
    ax1.set_ylim([0, np.max(maxes) * 1.5])
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
