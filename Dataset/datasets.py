import os
import numpy as np
import math
import random
import copy
from PIL import Image
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Base(Dataset):
    def __init__(self, params):
        super(Base, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

    def __getitem__(self, index):
        return self.sample_points_4d(index)

    def __len__(self):
        return self.LEN

    def setup(self):
        if self.mode == 'train':
            self.LEN = self.x
            self.pad = self.radius  
        elif self.mode == 'eval':
            self.vals = list(range(self.x * self.scale))
            self.LEN = self.x * self.scale
            self.pad = self.radius
        else:
            print(f'Not {self.mode} mode!')
            exit()

        self.pad = min(self.pad, 1)

    def normalize_coord(self, coord, dim_size):

        return -math.pi + (coord / (dim_size - 1)) * (2 * math.pi)

    def sample_points_4d(self, index):
        if self.mode == 'train':

            spatial_sample_count = int(self.bsize // self.t)
            grid_x = torch.arange(self.x)
            grid_y = torch.arange(self.y)
            grid_z = torch.arange(self.z)
            xx, yy, zz = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
            spatial_coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)  
            total_spatial_points = spatial_coords.shape[0]
            chosen_idx = torch.randperm(total_spatial_points)[:spatial_sample_count]
            sampled_spatial_idx = spatial_coords[chosen_idx]  

            sampled_data = self.data[
                sampled_spatial_idx[:, 0],
                sampled_spatial_idx[:, 1],
                sampled_spatial_idx[:, 2],
                :
            ]
            data_tensor = sampled_data.reshape(-1)  

            true_x = sampled_spatial_idx[:, 0]
            true_y = sampled_spatial_idx[:, 1]
            true_z = sampled_spatial_idx[:, 2]
            true_x_exp = true_x.unsqueeze(1).repeat(1, self.t).reshape(-1)
            true_y_exp = true_y.unsqueeze(1).repeat(1, self.t).reshape(-1)
            true_z_exp = true_z.unsqueeze(1).repeat(1, self.t).reshape(-1)
            true_t = torch.arange(self.t).unsqueeze(0).repeat(spatial_sample_count, 1).reshape(-1)

            true_x_exp11 = (true_x_exp.float() / (self.downsample)).long()
            true_y_exp11 = (true_y_exp.float() / (self.downsample)).long()
            true_z_exp11 = (true_z_exp.float() / (self.downsample)).long()
            true_t11 = (true_t.float() / (self.downsample)).long()
            
            real_coords_samples = torch.stack([true_x_exp11, true_y_exp11, true_z_exp11, true_t11], dim=1)

            center_x = true_x_exp
            center_y = true_y_exp
            center_z = true_z_exp
            center_t = true_t
            raw_pad = self.pad
            coords_samples_raw = torch.stack([
                center_x,
                center_y,
                center_z,
                center_t,
                center_x - raw_pad, center_x + raw_pad,
                center_y - raw_pad, center_y + raw_pad,
                center_z - raw_pad, center_z + raw_pad,
                center_t - raw_pad, center_t + raw_pad
            ], dim=1)

            coords_samples = coords_samples_raw.clone().float()
            coords_samples[:, [0, 4, 5]] = self.normalize_coord(coords_samples[:, [0, 4, 5]], self.x)
            coords_samples[:, [1, 6, 7]] = self.normalize_coord(coords_samples[:, [1, 6, 7]], self.y)
            coords_samples[:, [2, 8, 9]] = self.normalize_coord(coords_samples[:, [2, 8, 9]], self.z)
            coords_samples[:, [3, 10, 11]] = self.normalize_coord(coords_samples[:, [3, 10, 11]], self.t)


            return data_tensor, coords_samples, real_coords_samples, self.data.shape, self.scale

        else:
 
            new_X = int(self.x * self.scale)
            new_Y = int(self.y * self.scale)
            new_Z = int(self.z * self.scale)
            new_t = int(self.t * self.scaleT)

            raw_x = torch.arange(new_X)
            raw_y = torch.arange(new_Y)
            raw_z = torch.arange(new_Z)
            raw_t = torch.arange(new_t)
            
            fixed_x = raw_x[index]

            y_coords = torch.arange(new_Y)
            z_coords = torch.arange(new_Z)
            t_coords = torch.arange(new_t)
            yy, zz, tt = torch.meshgrid(y_coords, z_coords, t_coords, indexing='ij')
            plane_coords = torch.stack([
                torch.full_like(yy.flatten(), fixed_x),
                yy.flatten(),
                zz.flatten(),
                tt.flatten()
            ], dim=1)

            true_fixed_x = torch.full_like(yy.flatten(), index)
            true_y = yy.flatten()
            true_z = zz.flatten()
            true_t = tt.flatten()
            real_x = (true_fixed_x.float() / (self.scale * self.downsample)).long()
            real_y = (true_y.float() / (self.scale * self.downsample)).long()
            real_z = (true_z.float() / (self.scale * self.downsample)).long()
            real_t = (true_t.float() / (self.scaleT * self.downsample)).long()
            real_coords_samples = torch.stack([real_x, real_y, real_z, real_t], dim=1)

            raw_pad = self.pad
            coords_samples_raw = torch.stack([
                plane_coords[:, 0],
                plane_coords[:, 1],
                plane_coords[:, 2],
                plane_coords[:, 3],
                plane_coords[:, 0] - raw_pad, plane_coords[:, 0] + raw_pad,
                plane_coords[:, 1] - raw_pad, plane_coords[:, 1] + raw_pad,
                plane_coords[:, 2] - raw_pad, plane_coords[:, 2] + raw_pad,
                plane_coords[:, 3] - raw_pad, plane_coords[:, 3] + raw_pad
            ], dim=1)

            coords_samples = coords_samples_raw.clone().float()
            coords_samples[:, [0, 4, 5]] = self.normalize_coord(coords_samples[:, [0, 4, 5]], new_X)
            coords_samples[:, [1, 6, 7]] = self.normalize_coord(coords_samples[:, [1, 6, 7]], new_Y)
            coords_samples[:, [2, 8, 9]] = self.normalize_coord(coords_samples[:, [2, 8, 9]], new_Z)
            coords_samples[:, [3, 10, 11]] = self.normalize_coord(coords_samples[:, [3, 10, 11]], new_t)

            return coords_samples, real_coords_samples, self.data.shape, self.scale


class Medical3D(Base):
    def __init__(self, **params):
        super(Medical3D, self).__init__(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()

    def load_data(self):
        data = self.load_file()
        data = self.nomalize(data)
        self.data = data
        self.x, self.y, self.z, self.t = self.data.shape
        #print("Data shape:", self.x, self.y, self.z, self.t)
        self.setup()

    def align(self, data):
        return data

    def load_file(self):
        niimg = nib.load(self.file)
        numpy_data = niimg.get_fdata()

        tensor_data = torch.tensor(numpy_data, dtype=torch.float32).to(self.device)
        return tensor_data

    def nomalize(self, data):
        return (data - data.min()) / (data.max() - data.min())


if __name__ == '__main__':
    pass
