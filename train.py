import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import random
from tqdm import tqdm
from scipy.ndimage import zoom
from itertools import product
import warnings

from Model.RCAN4D import make_rcan4d
from Model.SIREN import SIREN_FiLM
from Implicit_Sampling.hypercube_sampling import hypercube_sampling
from Loss.loss import manifold_reg, manifold_cons
from utils import *

warnings.filterwarnings("ignore", category=FutureWarning)



def construct_L(L_flat, init_scale=0.01):
    *batch_shape, _ = L_flat.shape
    diag_raw = L_flat[..., :4]
    off_diag = L_flat[..., 4:10] * 0.1            
    diag = (F.softplus(diag_raw - 2.0) + 1e-3) * init_scale  
    L = torch.diag_embed(diag)
    rows = torch.tensor([1,2,2,3,3,3], device=L_flat.device)
    cols = torch.tensor([0,0,1,0,1,2], device=L_flat.device)
    L_strict = L_flat.new_zeros(*batch_shape, 4, 4)
    L_strict[..., rows, cols] = off_diag * init_scale
    return L + L_strict


def gmm_decode(intensity, L_flat, delta):

    L = construct_L(L_flat)                             
    delta_u = delta.unsqueeze(-1)                        

    
    y = torch.linalg.solve_triangular(L, delta_u, upper=False)
    mahal = (y * y).sum(dim=(-2, -1))                    

   
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)         
    log_det = 2.0 * torch.log(diag_L + 1e-8).sum(dim=-1) 

   
    log_e = -0.5 * log_det - 0.5 * mahal                
    weight = F.softmax(log_e, dim=1)                   

    intensity_sq = intensity.squeeze(-1)                
    I_hat = (intensity_sq * weight).sum(dim=1)          
    return I_hat, weight



def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    config["is_train"] = True
    log_iter = config.get("log_iter", 100)
    save_iter = config.get("save_iter", 300)
    save_dir = config.get("save_path", "./save")
    os.makedirs(save_dir, exist_ok=True)

    rcan_cfg = config['rcan']
    n_resgroups = rcan_cfg.get('n_resgroups', 3)
    n_resblocks = rcan_cfg.get('n_resblocks', 4)
    n_feats = rcan_cfg.get('n_feats', 32)
    reduction = rcan_cfg.get('reduction', 16)
    phase1_ratio = rcan_cfg.get('phase1_ratio', 0.2)
    patch_size = tuple(rcan_cfg.get('patch_size', [64, 64, 64]))
    f_min, f_max = rcan_cfg.get('aug_f_range', [1.0, 4.5])

    trainloader, dataset, hr_data = load_data(config)
    hr_data = hr_data.detach().cpu().numpy()
    hr_data = zoom(hr_data, zoom=(0.5, 0.5, 0.5, 1.0), order=3)
    hr_min, hr_max = hr_data.min(), hr_data.max()
    hr_data_norm = (hr_data - hr_min) / (hr_max - hr_min + 1e-8)
    X, Y, Z, T = hr_data_norm.shape
    hr_tensor_full = torch.from_numpy(hr_data_norm).unsqueeze(0).to(device)


    rcan = make_rcan4d(
        n_resgroups=n_resgroups,
        n_resblocks=n_resblocks,
        n_feats=n_feats,
        reduction=reduction,
        in_channels=T,
        len_Z=config["len_lz"],
        out_channels=T
    ).to(device)

    optimizer_rcan = optim.Adam(rcan.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    total_iters = config["max_iter"]
    phase1_iters = int(phase1_ratio * total_iters)

    lr_init = config["lr_decay"]["lr_init"]
    lr_final = config["lr_decay"]["lr_final"]
    max_iter_decay = config["lr_decay"]["max_iter"]
    lr_delay_steps = config["lr_decay"]["lr_delay_steps"]
    lr_delay_mult = config["lr_decay"]["lr_delay_mult"]
    clip_max_val = config["clip_grad"]["max_val"]
    clip_max_norm = config["clip_grad"]["max_norm"]

    lambda_reg = float(config.get("lambda_reg", 0.01))
    alpha_prior = float(config.get("alpha_prior", 10.0))

    latent_code = None
    prev_psnr_value = 1.0
    data_iter = None
    running_loss = 0.0
    running_psnr = 0.0

    manifold_net = None
    optimizer_mlp = None

    pbar = tqdm(range(1, total_iters + 1), desc="Training")

    for step in pbar:

        if step <= phase1_iters:
            rcan.train()
            f = random.uniform(f_min, f_max)
            target_t_len = max(2, int(round(T / f)))
            indices = np.unique(np.linspace(0, T - 1, num=target_t_len).astype(int))
            sx = random.randint(0, max(0, X - patch_size[0]))
            sy = random.randint(0, max(0, Y - patch_size[1]))
            sz = random.randint(0, max(0, Z - patch_size[2]))
            ex, ey, ez = sx + patch_size[0], sy + patch_size[1], sz + patch_size[2]

            lr_patch = hr_tensor_full[:, sx:ex, sy:ey, sz:ez, indices]
            target_patch = hr_tensor_full[:, sx:ex, sy:ey, sz:ez, :]

            optimizer_rcan.zero_grad()
            output_p, _ = rcan(lr_patch)
            loss = criterion(output_p, target_patch)
            loss.backward()
            optimizer_rcan.step()

            running_loss += loss.item()
            if step % log_iter == 0:
                #pbar.set_postfix({"Phase": "RCAN",
                                 # "AvgLoss": f"{running_loss/log_iter:.6f}"})
                running_loss = 0.0

        if step == phase1_iters:
            rcan.eval()
            temp_latent = torch.zeros((1, X, Y, Z, config["len_lz"]), device=device)
            with torch.no_grad():
                x_steps = range(0, X, patch_size[0])
                y_steps = range(0, Y, patch_size[1])
                z_steps = range(0, Z, patch_size[2])
                for sx, sy, sz in product(x_steps, y_steps, z_steps):
                    ex = min(sx + patch_size[0], X)
                    ey = min(sy + patch_size[1], Y)
                    ez = min(sz + patch_size[2], Z)
                    _, lz_p = rcan(hr_tensor_full[:, sx:ex, sy:ey, sz:ez, :])
                    temp_latent[:, sx:ex, sy:ey, sz:ez, :] = lz_p

            latent_code_cpu = temp_latent.cpu().clone()

            del temp_latent, lz_p, rcan, optimizer_rcan
            del hr_tensor_full, hr_data_norm
            try: del hr_data
            except: pass
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            latent_code = latent_code_cpu.to(device)

            manifold_net = SIREN_FiLM(**config['model']).to(device)
            optimizer_mlp = optim.Adam(
                manifold_net.parameters(),
                lr=config["optim"]["lr"], betas=config["optim"]["betas"]
            )

            data_iter = iter(trainloader)
            manifold_net.train()
            running_loss = 0.0

        if step > phase1_iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainloader)
                batch = next(data_iter)

            data = batch[0].squeeze(0).to(device)
            coords = batch[1].squeeze(0).to(device)
            real_coords = batch[2].squeeze(0).to(device)

            concatenated_coords, _, _ = rcan_skip(
                latent_code, prev_psnr_value, real_coords, coords, device
            )

            sample_ans = hypercube_sampling(
                batch=concatenated_coords,
                n_samples=config["n_samples"],
                len_lz=config["len_lz"],
                is_train=True
            )
            pts = sample_ans["pts"].to(device)
            delta_pts = sample_ans["delta_pts"].to(device)        

            coord_pts = pts[..., :4].requires_grad_(True)         
            latent_pts = pts[..., 4:]

           
            net_out = manifold_net(coord_pts, latent_pts)          
            intensity = net_out[..., 0:1]                          
            L_flat = net_out[..., 1:11]                           

           
            I_hat, weight = gmm_decode(intensity, L_flat, delta_pts)

           
            loss_cons = manifold_cons(I_hat, data)

           
            loss_prior = manifold_reg(intensity, coord_pts, alpha=alpha_prior)

            total_loss = loss_cons + lambda_reg * loss_prior

            optimizer_mlp.zero_grad()
            total_loss.backward()
            clip_grad(optimizer_mlp, clip_max_val, clip_max_norm)
            optimizer_mlp.step()

            mlp_step = step - phase1_iters
            current_lr = lr_decay(optimizer_mlp, mlp_step,
                                  lr_init, lr_final, max_iter_decay,
                                  lr_delay_steps, lr_delay_mult)

            running_loss += total_loss.item()
            psnr_val = 10 * math.log10(1 / (loss_cons.item() + 1e-8))
            running_psnr += psnr_val

            if step % log_iter == 0:
               # pbar.set_postfix({
                    #"Phase": "MLP",
                   # "Loss": f"{running_loss/log_iter:.5f}",
                    #"Cons": f"{loss_cons.item():.5f}",
                  #  "Prior": f"{loss_prior.item():.5f}",
                  #  "PSNR": f"{running_psnr/log_iter:.2f}",
                   # "LR": f"{current_lr:.6f}"
                #})
                running_loss = 0.0
                running_psnr = 0.0

            if step % save_iter == 0:
                save_path = os.path.join(save_dir, f"model_step_{step}.pth")
                torch.save({
                    "step": step,
                    "manifold_net_state_dict": manifold_net.state_dict(),
                    "latent_code": latent_code.cpu(),
                    "optimizer_mlp": optimizer_mlp.state_dict()
                }, save_path)

    save_all_path = os.path.join(config["save_path"], "final_model.pth")
    os.makedirs(os.path.dirname(save_all_path), exist_ok=True)
    if manifold_net is not None:
        torch.save({
            "manifold_net_state_dict": manifold_net.state_dict(),
            "latent_code": latent_code.cpu()
        }, save_all_path)
        print(f"\nTraining finished. Model saved to {save_all_path}")
    else:
        print("\nTraining ended.")


if __name__ == "__main__":
    config = parse_arguments_and_update_config()
    train(config)