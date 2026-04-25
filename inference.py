import os
import numpy as np
import torch
import nibabel as nib
import warnings
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter

from Dataset.datasets import Medical3D
from Implicit_Sampling.hypercube_sampling import hypercube_sampling
from Model.SIREN import SIREN_FiLM
from utils import (
    load_data, rcan_skip, predict_coords_slice,
    parse_arguments_and_update_config
)

warnings.filterwarnings("ignore", category=FutureWarning)


def inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["is_train"] = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    save_path = config.get("save_path", "./save")
    model_path = os.path.join(save_path, "final_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model file: {model_path}")

    print(f"Loading model: {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device)

    latent_code = checkpoint["latent_code"].to(device)

    manifold_net = SIREN_FiLM(**config['model']).to(device)
    manifold_net.load_state_dict(checkpoint["manifold_net_state_dict"])
    manifold_net.eval()

    evalloader, dataset, _ = load_data(config, mode="eval")

    new_X = int(dataset.x * dataset.scale)
    new_Y = int(dataset.y * dataset.scale)
    new_Z = int(dataset.z * dataset.scale)
    new_T = int(dataset.t * dataset.scaleT)

    print(f"Starting inference, target size: {new_X}x{new_Y}x{new_Z}x{new_T}")

    predicted_slices = []
    prev_psnr_value = 1.0

    for idx, batch in tqdm(enumerate(evalloader),
                           total=len(evalloader),
                           desc="[Inferencing]"):

        coords, real_coords, data_shape, scale = batch
        coords = coords.squeeze(0).to(device)
        real_coords = real_coords.squeeze(0).to(device)

        concatenated_coords, _, _ = rcan_skip(
            latent_code, prev_psnr_value, real_coords, coords, device
        )

        pred_slice = predict_coords_slice(
            coords=concatenated_coords,
            manifold_net=manifold_net,
            device=device,
            config=config,
            new_Y=new_Y,
            new_Z=new_Z,
            new_T=new_T,
            hypercube_sampling=hypercube_sampling
        )
        predicted_slices.append(pred_slice)

    full_volume = torch.cat(predicted_slices, dim=0)
    print("Full volume shape:", full_volume.shape)

    full_volume_np = full_volume.numpy()

    v_min, v_max = np.percentile(full_volume_np, (0, 100))
    img_rescaled = rescale_intensity(
        full_volume_np, in_range=(v_min, v_max), out_range=(0, 255)
    ).astype(np.uint8)

    img_rescaled = gaussian_filter(img_rescaled, sigma=1)

    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(img_rescaled, affine)

    output_path = os.path.join(config["save_path"], "output.nii.gz")
    nib.save(nifti_img, output_path)
    print(f"Saved 4D NIfTI image to {output_path}")


if __name__ == "__main__":
    config = parse_arguments_and_update_config()
    inference(config)

