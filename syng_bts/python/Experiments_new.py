#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:09:21 2022

@author: yunhui, xinyi
"""

# %% Import libraries
import torch
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from .helper_utils_new import *
from .helper_training_new import *
import re

sns.set()

import importlib.resources as pkg_resources


# %% Define pilot experiments functions
def PilotExperiment(
    dataname,
    pilot_size,
    model,
    batch_frac,
    learning_rate,
    epoch,
    early_stop_num=30,
    off_aug=None,
    AE_head_num=2,
    Gaussian_head_num=9,
    pre_model=None,
):
    r"""
    This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP with several pilot sizes given data, model, batch_size, learning_rate, epoch, off_aug and pre_model.
    For each pilot size, there will be 5 random draws from the original dataset.
    For each draw, the pilot data is served as the input to the model training, and the generated data has sample size equal to 5 times the original sample size.

    Parameters
    ----------
    dataname : string
               pure data name without .csv. Eg: SKCMPositive_4
    pilot_size : list
                 a list including potential pilot sizes
    model : string
            name of the model to be trained
    batch_frac : float
                  batch fraction
    learning_rate : float
            learning rate
    epoch : int
                            choose from None (early_stop), or any integer, if choose None, early_stop_num will take effect
    early_stop_num : int
          if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == “None”
    off_aug : string (AE_head or Gaussian_head or None)
                        choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
    AE_head_num : int
                how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
    Gaussian_head_num : int
            how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
    pre_model : string
                    transfer learning input model. If pre_model == None, no transfer learning

    """
    # read in data

    path = "../RealData/" + dataname + ".csv"

    # just use an if statement for datasets that are already built in
    if dataname == "SKCMPositive_4" and not os.path.exists(path):
        with pkg_resources.open_text(
            "syng_bts.RealData", "SKCMPositive_4.csv"
        ) as data_file:
            df = pd.read_csv(data_file)
    else:
        df = pd.read_csv(path, header=0)
    dat_pd = df
    data_pd = dat_pd.select_dtypes(include=np.number)
    oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)

    # log2 transformation
    oridata = preprocessinglog2(oridata)
    n_samples = oridata.shape[0]

    # get group information if there is or is not
    if "groups" in dat_pd.columns:
        groups = dat_pd["groups"]
    else:
        groups = None

    # create 0-1 labels, this function use the first element in groups as 0.
    # also create blurlabels.
    orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)

    print("1. Read data, path is " + path)

    # get model name and kl_weight if modelname is some autoencoder
    if len(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)) > 1:
        kl_weight = int(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[4])
        modelname = re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[1]
    else:
        modelname = model
        kl_weight = 1

    print("2. Determine the model is " + model + " with kl-weight = " + str(kl_weight))

    # decide batch fraction in file name
    model = "batch" + str(batch_frac).replace(".", "") + "_" + model

    # decide epochs
    if epoch is not None:
        num_epochs = epoch
        early_stop = False
        epoch_info = str(epoch)
        model = "epoch" + epoch_info + "_" + model
    else:
        num_epochs = 1000
        early_stop = True
        epoch_info = "early_stop"
        model = "epochES_" + model

    # decide offline augmentation
    if off_aug == "AE_head":
        AE_head = True
        Gaussian_head = False
        off_aug_info = off_aug
    elif off_aug == "Gaussian_head":
        Gaussian_head = True
        AE_head = False
        off_aug_info = off_aug
    else:
        AE_head = False
        Gaussian_head = False
        off_aug_info = "No"

    print(
        "3. Determine the training parameters are epoch = "
        + epoch_info
        + " off_aug = "
        + off_aug_info
        + " learing rate = "
        + str(learning_rate)
        + " batch_frac = "
        + str(batch_frac)
    )

    random_seed = 123
    repli = 5

    if (len(torch.unique(orilabels)) > 1) & (
        int(sum(orilabels == 0)) != int(sum(orilabels == 1))
    ):
        new_size = [int(sum(orilabels == 0)), int(sum(orilabels == 1)), repli]
    else:
        new_size = [repli * n_samples]

    if pre_model is not None:
        model = model + "_transfrom" + re.search(r"from([A-Z]+)_", pre_model).group(1)

    print("4. Pilot experiments start ... ")
    for n_pilot in pilot_size:
        for rand_pilot in [1, 2, 3, 4, 5]:
            print(
                "Training for data="
                + dataname
                + ", model="
                + model
                + ", pilot size="
                + str(n_pilot)
                + ", for "
                + str(rand_pilot)
                + "-th draw"
            )

            # get pilot_size real samples as seeds for DGM. For two cancers, the first n_pilot are from group 0, the second n_pilot are from group 1
            rawdata, rawlabels, rawblurlabels = draw_pilot(
                dataset=oridata,
                labels=orilabels,
                blurlabels=oriblurlabels,
                n_pilot=n_pilot,
                seednum=rand_pilot,
            )

            # for training of two cancers without CVAE, we use blurlabels as an additional feature to train
            if (modelname != "CVAE") and (torch.unique(rawlabels).shape[0] > 1):
                rawdata = torch.cat((rawdata, rawblurlabels), dim=1)

            savepath = (
                "../ReconsData/"
                + dataname
                + "_"
                + model
                + "_"
                + str(n_pilot)
                + "_Draw"
                + str(rand_pilot)
                + ".csv"
            )
            savepathnew = (
                "../GeneratedData/"
                + dataname
                + "_"
                + model
                + "_"
                + str(n_pilot)
                + "_Draw"
                + str(rand_pilot)
                + ".csv"
            )
            losspath = (
                "../Loss/"
                + dataname
                + "_"
                + model
                + "_"
                + str(n_pilot)
                + "_Draw"
                + str(rand_pilot)
                + ".csv"
            )

            # whether or not add Gaussian_head augmentation
            if Gaussian_head:
                rawdata, rawlabels = Gaussian_aug(
                    rawdata, rawlabels, multiplier=[Gaussian_head_num]
                )
                savepath = (
                    "../ReconsData/"
                    + dataname
                    + "_Gaussianhead_"
                    + model
                    + "_"
                    + str(n_pilot)
                    + "_Draw"
                    + str(rand_pilot)
                    + ".csv"
                )
                savepathnew = (
                    "../GeneratedData/"
                    + dataname
                    + "_Gaussianhead_"
                    + model
                    + "_"
                    + str(n_pilot)
                    + "_Draw"
                    + str(rand_pilot)
                    + ".csv"
                )
                losspath = (
                    "../Loss/"
                    + dataname
                    + "_"
                    + model
                    + "_Gaussianhead_"
                    + str(n_pilot)
                    + "_Draw"
                    + str(rand_pilot)
                    + ".csv"
                )
                print("Gaussian head is added.")

            # if AE_head = True, for each pilot size, 2 iterative AE reconstruction will be conducted first
            # resulting in n_pilot * 4 samples, and the extended samples will be input to the model specified by modelname
            if AE_head:
                savepath = (
                    "../ReconsData/"
                    + dataname
                    + "_AEhead_"
                    + model
                    + "_"
                    + str(n_pilot)
                    + "_Draw"
                    + str(rand_pilot)
                    + ".csv"
                )
                savepathnew = (
                    "../GeneratedData/"
                    + dataname
                    + "_AEhead_"
                    + model
                    + "_"
                    + str(n_pilot)
                    + "_Draw"
                    + str(rand_pilot)
                    + ".csv"
                )
                savepathextend = (
                    "../ExtendData/"
                    + dataname
                    + "_AEhead_"
                    + model
                    + "_"
                    + str(n_pilot)
                    + "_Draw"
                    + str(rand_pilot)
                    + ".csv"
                )
                losspath = (
                    "../Loss/"
                    + dataname
                    + "_AEhead_"
                    + model
                    + "_"
                    + str(n_pilot)
                    + "_Draw"
                    + str(rand_pilot)
                    + ".csv"
                )
                print("AE reconstruction head is added, reconstruction starting ...")
                feed_data, feed_labels = training_iter(
                    iter_times=AE_head_num,  # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
                    savepathextend=savepathextend,  # save path of the extended dataset
                    rawdata=rawdata,  # pilot data
                    rawlabels=rawlabels,  # pilot labels
                    random_seed=random_seed,
                    modelname="AE",  # choose from AE, VAE
                    num_epochs=1000,  # maximum number of epochs if early stop is not triggered, default value for AEhead is 1000
                    batch_size=round(
                        rawdata.shape[0] * 0.1
                    ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
                    learning_rate=0.0005,  # learning rate, default value for AEhead is 0.0005
                    early_stop=False,  # AEhead by default does not utilize early stopping rule
                    early_stop_num=30,  # won't take effect since early_stop == False
                    kl_weight=1,  # only take effect if model name is VAE, default value is 1
                    loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
                    replace=True,  # whether to replace the failure features in each reconstruction
                    saveextend=False,  # whether to save the extended dataset, if true, savepathextend must be provided
                    plot=False,
                )  # whether or not plot the heatmap of extended data

                rawdata = feed_data
                rawlabels = feed_labels
                print("Reconstruction finish, AE head is added.")
            # Training
            if "GAN" in modelname:
                log_dict = training_GANs(
                    savepathnew=savepathnew,  # path to save newly generated samples
                    rawdata=rawdata,  # raw data matrix with samples in row, features in column
                    rawlabels=rawlabels,  # labels for each sample, n_samples * 1, will not be used in AE or VAE
                    batch_size=round(
                        rawdata.shape[0] * batch_frac
                    ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
                    random_seed=random_seed,
                    modelname=modelname,  # choose from "GAN","WGAN","WGANGP"
                    num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
                    learning_rate=learning_rate,
                    new_size=new_size,  # how many new samples you want to generate
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,  # save model for transfer learning, specify the path if want to save model
                    save_new=True,  # whether to save the newly generated samples
                    plot=False,
                )  # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones

                print("GAN model training for one pilot size one draw finished.")

                log_pd = pd.DataFrame(
                    {
                        "discriminator": log_dict["train_discriminator_loss_per_batch"],
                        "generator": log_dict["train_generator_loss_per_batch"],
                    }
                )
                # create directory if not exists
                components = losspath.split("/")
                directory = "/".join(losspath.split("/")[:2])
                # print("Directory created: " + directory)
                os.makedirs(directory, exist_ok=True)
                for i in range(2, len(components) - 1):
                    directory = directory + "/" + components[i]
                    os.makedirs(directory, exist_ok=True)
                    print("Directory created: " + directory)
                log_pd.to_csv(Path(losspath), index=False)

            elif "AE" in modelname:
                log_dict = training_AEs(
                    savepath=savepath,  # path to save reconstructed samples
                    savepathnew=savepathnew,  # path to save newly generated samples
                    rawdata=rawdata,  # raw data tensor with samples in row, features in column
                    rawlabels=rawlabels,  # abels for each sample, n_samples * 1, will not be used in AE or VAE
                    batch_size=round(rawdata.shape[0] * batch_frac),  # batch size
                    random_seed=random_seed,
                    modelname=modelname,  # choose from "VAE", "AE"
                    num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
                    learning_rate=learning_rate,
                    kl_weight=kl_weight,  # only take effect if model name is VAE, default value is
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,  # save model for transfer learning, specify the path if want to save model
                    loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
                    save_recons=False,  # whether save reconstructed data, if True, savepath must be provided
                    new_size=new_size,  # how many new samples you want to generate
                    save_new=True,  # whether save new samples, if True, savepathnew must be provided
                    plot=False,
                )  # whether plot reconstructed samples' heatmap

                print("VAEs model training for one pilot size one draw finished.")
                log_pd = pd.DataFrame(
                    {
                        "kl": log_dict["train_kl_loss_per_batch"],
                        "recons": log_dict["train_reconstruction_loss_per_batch"],
                    }
                )
                # create directory if not exists
                components = losspath.split("/")
                directory = "/".join(losspath.split("/")[:2])
                # print("Directory created: " + directory)
                os.makedirs(directory, exist_ok=True)
                for i in range(2, len(components) - 1):
                    directory = directory + "/" + components[i]
                    os.makedirs(directory, exist_ok=True)
                    print("Directory created: " + directory)
                log_pd.to_csv(Path(losspath), index=False)
            elif "maf" in modelname:
                training_flows(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epoches=num_epochs,
                    learning_rate=learning_rate,
                    new_size=new_size,
                    num_hidden=226,
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,
                    # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,
                    plot=False,
                )
            elif "realnvp" in modelname:
                training_flows(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epoches=num_epochs,
                    learning_rate=learning_rate,
                    new_size=new_size,
                    num_hidden=226,
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,
                    # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,
                    plot=False,
                )

            elif "glow" in modelname:
                training_flows(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epoches=num_epochs,
                    learning_rate=learning_rate,
                    new_size=new_size,
                    num_hidden=226,
                    early_stop=early_stop,  # whether use early stopping rule
                    early_stop_num=early_stop_num,
                    # stop training if loss does not improve for early_stop_num epochs
                    pre_model=pre_model,  # load pre-trained model from transfer learning
                    save_model=None,
                    plot=False,
                )

            else:
                print("wait for other models")


# %% Define application of experiment
def ApplyExperiment(
    path,
    dataname,
    apply_log,
    new_size,
    model,
    batch_frac,
    learning_rate,
    epoch,
    val_ratio = 0.2, # Control whether separate the dataset in-function
    early_stop_num=None,
    off_aug=None,
    AE_head_num=2,
    Gaussian_head_num=9,
    pre_model=None,
    save_model=None,
    cap = False,
    
    use_scheduler = False,
    step_size = 10,
    gamma = 0.5,
    random_seed = 123
):
    r"""
        This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP
        given data, model, batch_size, learning_rate, epoch, off_aug and pre_model
        and generate new samples with size specified by the users.

    Parameters
    ----------
    path : string
                              path for reading real data and saving new data
    dataname : string
                    pure data name without .csv. Eg: BRCASubtypeSel_train
    apply_log : boolean
                      logical whether apply log transformation before training
    new_size : int
             the number of generated samples. If CVAE is called, the group sample size will be new_size/2.
    model : string
                              name of the model to be trained
    batch_frac : float
                    batch fraction
    learning_rate : float
              learning rate
    epoch : int
            choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
    early_stop_num : int
            if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == None.
    off_aug : string (AE_head or Gaussian_head or None)
                      choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
    AE_head_num : int
                  how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
    Gaussian_head_num : int
         how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
    pre_model : string
                      transfer learning input model. If pre_model == None, no transfer learning
    save_model : string
                    if the trained model should be saved, specify the path and name of the saved model
    use_scheduler : bool
                    turn on/off scheduler for training
    step_size : int
                    step size for scheduler
    gamma : float
                    gamma for scheduler
    """

    read_path = path + dataname + ".csv"
    # just use an if statement for datasets that are already built in
    if dataname == "BRCASubtypeSel" and not os.path.exists(path):
        with pkg_resources.open_text(
            "syng_bts.Case.BRCASubtype", "BRCASubtypeSel.csv"
        ) as data_file:
            df = pd.read_csv(data_file)
    else:
        df = pd.read_csv(read_path, header=0)
    dat_pd = df
    data_pd = dat_pd.select_dtypes(include=np.number)
    if "groups" in data_pd.columns:
        data_pd = data_pd.drop(columns=["groups"])
    oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)
    colnames = data_pd.columns
    if apply_log:
        oridata = preprocessinglog2(oridata)
    n_samples = oridata.shape[0]
    if "groups" in dat_pd.columns:
        groups = dat_pd["groups"]
    else:
        groups = None

    orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)
    print("1. Read data, path is " + read_path)

    # get model name and kl_weight if modelname is some autoencoder
    if len(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)) > 1:
        kl_weight = int(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[4])
        modelname = re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[1]
    else:
        modelname = model
        kl_weight = 1

    print("2. Determine the model is " + model + " with kl-weight = " + str(kl_weight))

    rawdata = oridata
    rawlabels = orilabels

    # decide batch fraction in file name
    model = "batch" + str(batch_frac).replace(".", "") + "_" + model

    # decide epoch
    num_epochs = epoch
    if early_stop_num is not None:
        early_stop = True
        epoch_info = "early_stop"
        model = "epochES_" + model
    else:
        early_stop = False
        epoch_info = str(epoch)
        model = "epoch" + epoch_info + "_" + model

    # decide offline augmentation
    if off_aug == "AE_head":
        AE_head = True
        Gaussian_head = False
        off_aug_info = off_aug
    elif off_aug == "Gaussian_head":
        Gaussian_head = True
        AE_head = False
        off_aug_info = off_aug
    else:
        AE_head = False
        Gaussian_head = False
        off_aug_info = "No"

    print(
        "3. Determine the training parameters are epoch = "
        + epoch_info
        + " off_aug = "
        + off_aug_info
        + " learing rate = "
        + str(learning_rate)
        + " batch_frac = "
        + str(batch_frac)
    )

    if pre_model is not None:
        model = model + "_transfrom" + re.search(r"from([A-Z]+)_", pre_model).group(1)

    # hyperparameters
    # random_seed = 123

    savepath = path + dataname + "_" + model + "_recons.csv"
    savepathnew = path + dataname + "_" + model + "_generated.csv"
    losspath = path + dataname + "_" + model + "_loss.csv"

    if Gaussian_head:
        rawdata, rawlabels = Gaussian_aug(
            rawdata, rawlabels, multiplier=[Gaussian_head_num]
        )
        savepath = path + dataname + "_Gaussianhead_" + model + "_recons.csv"
        savepathnew = path + dataname + "_Gaussianhead_" + model + "_generated.csv"
        losspath = path + dataname + "_Gaussianhead_" + model + "_loss.csv"
        print("Gaussian head is added.")

    if AE_head:
        savepathextend = path + dataname + "_AEhead_" + model + "_extend.csv"
        savepath = path + dataname + "_AEhead_" + model + "_recons.csv"
        savepathnew = path + dataname + "_AEhead_" + model + "_generated.csv"
        losspath = path + dataname + "_AEhead_" + model + "_loss.csv"
        print("AE reconstruction head is added, reconstruction starting ...")
        feed_data, feed_labels = training_iter(
            iter_times=AE_head_num,  # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
            savepathextend=savepathextend,  # save path of the extended dataset
            rawdata=rawdata,  # pilot data
            rawlabels=rawlabels,  # pilot labels
            random_seed=random_seed,
            modelname="AE",  # choose from AE, VAE
            num_epochs=1000,  # maximum number of epochs if early stop is not triggered, default value for AEhead is 1000
            batch_size=round(
                rawdata.shape[0] * 0.1
            ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
            learning_rate=0.0005,  # learning rate, default value for AEhead is 0.0005
            early_stop=False,  # AEhead by default does not utilize early stopping rule
            early_stop_num=30,  # won't take effect since early_stop == False
            kl_weight=1,  # only take effect if model name is VAE, default value is 2
            loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
            replace=True,  # whether to replace the failure features in each reconstruction
            saveextend=False,  # whether to save the extended dataset, if true, savepathextend must be provided
            plot=False,
        )  # whether or not plot the heatmap of extended data

        rawdata = feed_data
        rawlabels = feed_labels
        print("AEhead added.")

    print("3. Training starts ......")
    # Training
    if "GAN" in modelname:
        log_dict = training_GANs(
            savepathnew=savepathnew,  # path to save newly generated samples
            rawdata=rawdata,  # raw data matrix with samples in row, features in column
            rawlabels=rawlabels,  # labels for each sample, n_samples * 1, will not be used in AE or VAE
            batch_size=round(
                rawdata.shape[0] * batch_frac
            ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
            random_seed=random_seed,
            modelname=modelname,  # choose from "GAN","WGAN","WGANGP"
            num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
            learning_rate=learning_rate,
            new_size=new_size,  # how many new samples you want to generate
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,  # save model for transfer learning, specify the path if want to save model
            save_new=True,  # whether to save the newly generated samples
            plot=False,
        )  # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones

        print("GAN model training finished.")

        log_pd = pd.DataFrame(
            {
                "discriminator": log_dict["train_discriminator_loss_per_batch"],
                "generator": log_dict["train_generator_loss_per_batch"],
            }
        )
        # create directory if not exists
        # temp fix to paths
        components = losspath.split("/")
        directory = "/".join(losspath.split("/")[:2])
        # print("Directory created: " + directory)
        os.makedirs(directory, exist_ok=True)
        for i in range(2, len(components) - 1):
            directory = directory + "/" + components[i]
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)
        log_pd.to_csv(Path(losspath), index=False)

    elif "AE" in modelname:
        log_dict = training_AEs(
            savepath=savepath,  # path to save reconstructed samples
            savepathnew=savepathnew,  # path to save newly generated samples
            rawdata=rawdata,  # raw data tensor with samples in row, features in column
            rawlabels=rawlabels,  # abels for each sample, n_samples * 1, will not be used in AE or VAE
            colnames = colnames,  # colnames saved
            batch_size=round(rawdata.shape[0] * batch_frac),  # batch size
            random_seed=random_seed,
            modelname=modelname,  # choose from "VAE", "AE"
            num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
            learning_rate=learning_rate,
            val_ratio = val_ratio, # validation set ratio
            kl_weight=kl_weight,  # only take effect if model name is VAE, default value is
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,  # save model for transfer learning, specify the path if want to save model
            cap = cap, # whether capping the new samples
            loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
            save_recons=False,  # whether save reconstructed data, if True, savepath must be provided
            new_size=new_size,  # how many new samples you want to generate
            save_new=True,  # whether save new samples, if True, savepathnew must be provided
            plot=False,
            
            use_scheduler = use_scheduler,
            step_size = step_size,
            gamma = gamma
        )  # whether plot reconstructed samples' heatmap

        print("VAEs model training finished.")
        log_pd = pd.DataFrame(
            {
                "kl": log_dict["val_kl_loss_per_batch"],
                "recons": log_dict["val_reconstruction_loss_per_batch"],
                "comb": log_dict['val_combined_loss_per_batch'],
                "param-batch": log_dict['latent_statistics_per_batch']
            }
        )
        # create directory if not exists
        # temp fix to paths
        components = losspath.split("/")
        directory = "/".join(losspath.split("/")[:2])
        print("Directory created: " + directory)
        os.makedirs(directory, exist_ok=True)
        for i in range(2, len(components) - 1):
            directory = directory + "/" + components[i]
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)
        log_pd.to_csv(Path(losspath), index=False)
    elif "maf" in modelname:
        training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=226,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
        )
    elif "realnvp" in modelname:
        training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=226,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
        )

    elif "glow" in modelname:
        training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=226,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
        )

    else:
        print("wait for other models")


# %% Define transfer learing
def TransferExperiment(
    pilot_size,
    fromname,
    toname,
    fromsize,
    model,
    new_size=500,
    apply_log=True,
    epoch=None,
    batch_frac=0.1,
    learning_rate=0.0005,
    off_aug=None,
):
    """
    This function runs transfer learning using VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP.
    The model will be first trained on the pre-training dataset, and then the trained model will be saved, and the fine-tuning dataset will be trained on the save model.
    The fine tuning model training can be pilot experiments or apply experiments depending on the input of the pilot_size.
    Make sure data files for pre_model training and fine tuning model training are in the folder Transfer/.

    Parameters
    ----------
    pilot_size : int
                    if None, the fine tuning model will be apply experiment and new_size will take effect
                    otherwise, the fine tuning model will be trained using pilot experiments
    fromname : string
                name of the pretraining dataset
    toname : string
                name of the fine tuning dataset
    fromsize : int
                number of samples when pre-training the model
    new_size : int
                if apply experiment, this will be the sample size of generated samples
    apply_log : boolean
                logical whether apply log transformation before training
    model : string
                name of the model to be trained
    batch_frac : float
                batch fraction
    learning_rate : float
              learning rate
    epoch : int
            choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
    off_aug : string (AE_head or Gaussian_head or None)
            choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
    """

    path = "../Transfer/"
    save_model = "../Transfer/" + toname + "_from" + fromname + "_" + model + ".pt"
    ApplyExperiment(
        path=path,
        dataname=fromname,
        apply_log=apply_log,
        new_size=[fromsize],
        model=model,
        batch_frac=batch_frac,
        learning_rate=learning_rate,
        epoch=epoch,
        early_stop_num=30,
        off_aug=off_aug,
        AE_head_num=2,
        Gaussian_head_num=9,
        pre_model=None,
        save_model=save_model,
    )

    # training toname using pre-model
    pre_model = "../Transfer/" + toname + "_from" + fromname + "_" + model + ".pt"
    if pilot_size is not None:
        PilotExperiment(
            dataname=toname,
            pilot_size=pilot_size,
            model=model,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            pre_model=pre_model,
            epoch=epoch,
            off_aug=off_aug,
            early_stop_num=30,
            AE_head_num=2,
            Gaussian_head_num=9,
        )
    else:
        ApplyExperiment(
            path=path,
            dataname=toname,
            apply_log=apply_log,
            new_size=[new_size],
            model=model,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            epoch=epoch,
            early_stop_num=30,
            off_aug=off_aug,
            AE_head_num=2,
            Gaussian_head_num=9,
            pre_model=pre_model,
            save_model=None,
        )