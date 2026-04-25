import torch
import math


def hypercube_sampling(batch, len_lz, n_samples, is_train):
    data_main, extra_feats = torch.split(batch, [12, len_lz], dim=-1)
    cnts, x_bound, y_bound, z_bound, t_bound = torch.split(
        data_main, [4, 2, 2, 2, 2], dim=-1
    )

    base_steps = round(math.pow(n_samples, 1.0 / 4))
    bin_size = 1.0 / base_steps
    grid = torch.arange(base_steps, device=batch.device).float() * bin_size

    mesh_x, mesh_y, mesh_z, mesh_t = torch.meshgrid(
        grid, grid, grid, grid, indexing='ij'
    )
    t_vals_left = torch.stack([mesh_x, mesh_y, mesh_z, mesh_t], dim=-1)
    t_vals_left = t_vals_left.contiguous().view(-1, 4).unsqueeze(0)   # (1, n, 4)

    B = batch.shape[0]
    n_points = t_vals_left.shape[1]

    x_l = x_bound[:, 0:1].unsqueeze(1)
    x_r = x_bound[:, 1:2].unsqueeze(1)
    y_l = y_bound[:, 0:1].unsqueeze(1)
    y_r = y_bound[:, 1:2].unsqueeze(1)
    z_l = z_bound[:, 0:1].unsqueeze(1)
    z_r = z_bound[:, 1:2].unsqueeze(1)
    t_l = t_bound[:, 0:1].unsqueeze(1)
    t_r = t_bound[:, 1:2].unsqueeze(1)

    if is_train:
        x_rand = torch.rand(B, n_points, 1, device=batch.device)
        y_rand = torch.rand(B, n_points, 1, device=batch.device)
        z_rand = torch.rand(B, n_points, 1, device=batch.device)
        t_rand = torch.rand(B, n_points, 1, device=batch.device)
        x_unit = t_vals_left[:, :, 0:1] + x_rand * bin_size
        y_unit = t_vals_left[:, :, 1:2] + y_rand * bin_size
        z_unit = t_vals_left[:, :, 2:3] + z_rand * bin_size
        t_unit = t_vals_left[:, :, 3:4] + t_rand * bin_size
    else:
        x_unit = t_vals_left[:, :, 0:1] + 0.5 * bin_size
        y_unit = t_vals_left[:, :, 1:2] + 0.5 * bin_size
        z_unit = t_vals_left[:, :, 2:3] + 0.5 * bin_size
        t_unit = t_vals_left[:, :, 3:4] + 0.5 * bin_size

    x_vals = x_l + x_unit * (x_r - x_l)
    y_vals = y_l + y_unit * (y_r - y_l)
    z_vals = z_l + z_unit * (z_r - z_l)
    t_vals_coord = t_l + t_unit * (t_r - t_l)

    pts_coords = torch.cat([x_vals, y_vals, z_vals, t_vals_coord], dim=-1)  # (B, n, 4)

    cnts_expanded = cnts.unsqueeze(1).expand(B, n_points, 4)
    delta_pts = pts_coords - cnts_expanded                                  # (B, n, 4)

    extra_feats_expanded = extra_feats.unsqueeze(1).expand(B, n_points, len_lz)
    pts = torch.cat([pts_coords, extra_feats_expanded], dim=-1)             # (B, n, 4+lz)

    dx = (x_r - x_l).mean() / 2
    dy = (y_r - y_l).mean() / 2
    dz = (z_r - z_l).mean() / 2
    dt = (t_r - t_l).mean() / 2

    return {
        'pts': pts,
        'cnts': cnts,
        'delta_pts': delta_pts,
        'dx': dx, 'dy': dy, 'dz': dz, 'dt': dt
    }


if __name__ == '__main__':
    pass