import os
import math
import numpy as np
import torch
import torch.nn as nn
import random
import copy
import nibabel as nib
import scipy.ndimage
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter, zoom

from Dataset.datasets import Medical3D
from Implicit_Sampling.hypercube_sampling import hypercube_sampling
from Model.RCAN4D import make_rcan4d
import yaml


def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def rcan_skip(prev_img_representation, prev_psnr_value, real_coords, coords, device):
    loss_rcan = torch.tensor(0.0, device=device)
    psnr_value = prev_psnr_value
    img_representation = prev_img_representation
    if img_representation.dim() == 5:
        img_representation = img_representation.squeeze(0)
    x_idx = real_coords[:, 0]
    y_idx = real_coords[:, 1]
    z_idx = real_coords[:, 2]
    lz = img_representation[x_idx, y_idx, z_idx, :].to(device)
    coords = coords.to(device)
    concatenated_coords = torch.cat((coords, lz), dim=1)
    return concatenated_coords, loss_rcan, psnr_value


def calculate_psnr(pred, target, max_val=None):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    if max_val is None:
        max_val = torch.max(target)
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()


def clip_grad(optimizer, max_val, max_norm):
    for param_group in optimizer.param_groups:
        if max_val > 0:
            torch.nn.utils.clip_grad_value_(param_group['params'], max_val)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm)


def lr_decay(optimizer, step, lr_init, lr_final, max_iter, lr_delay_steps=0, lr_delay_mult=1):
    def log_lerp(t, v0, v1):
        lv0 = np.log(v0)
        lv1 = np.log(v1)
        return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)

    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.0

    new_lr = delay_rate * log_lerp(step / max_iter, lr_init, lr_final)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def load_data(config, mode="train"):
    dataset_params = config["dataset"][mode].copy()
    dataset_params["mode"] = mode
    dataset = Medical3D(**dataset_params)
    normalized_data = dataset.data
    dataloader_params = config["dataloader"][mode]
    trainloader = DataLoader(
        dataset,
        batch_size=dataloader_params["batch_size"],
        shuffle=dataloader_params["shuffle"]
    )
    return trainloader, dataset, normalized_data


def predict_coords_slice(coords, manifold_net, device, config,
                         new_Y, new_Z, new_T, hypercube_sampling=None):

    manifold_net.eval()
    with torch.no_grad():
        chunk_size = config["dataset"]["eval"]["bsize"]
        len_lz = config["len_lz"]
        pred_chunks = []
        coords_device = coords.to(device)

        for i in range(0, coords_device.shape[0], chunk_size):
            chunk = coords_device[i:i + chunk_size]                
            data_main, extra_feats = torch.split(chunk, [12, len_lz], dim=-1)
            cnts = data_main[..., :4]                              
            
            coord_pts = cnts.unsqueeze(1)                         
            latent_pts = extra_feats.unsqueeze(1)                

            net_out = manifold_net(coord_pts, latent_pts)         
            intensity = net_out[..., 0]                            

            pred_chunks.append(intensity.squeeze(-1).cpu())      

        pred = torch.cat(pred_chunks, dim=0)

    pred_slice = pred.view(1, new_Y, new_Z, new_T)
    return pred_slice


def parse_arguments_and_update_config():
    import argparse
    parser = argparse.ArgumentParser(description="Update configuration and run inference")
    parser.add_argument("--config", type=str, default="./Config/config.yaml")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--scale", type=int)
    parser.add_argument("--scaleT", type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.save_path is not None:
        config["save_path"] = args.save_path
    if args.file is not None:
        if "dataset" in config:
            if "train" in config["dataset"]:
                config["dataset"]["train"]["file"] = args.file
            if "eval" in config["dataset"]:
                config["dataset"]["eval"]["file"] = args.file
    if args.scale is not None:
        if "dataset" in config and "eval" in config["dataset"]:
            config["dataset"]["eval"]["scale"] = args.scale
    if args.scaleT is not None:
        if "dataset" in config and "eval" in config["dataset"]:
            config["dataset"]["eval"]["scaleT"] = args.scaleT

    return config