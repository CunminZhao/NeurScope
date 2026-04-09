## Neural Spatiotemporal Manifold Extrapolation for 4D Microscopy Super-Resolution

**Motivations**: 4D microscopy is fundamentally constrained by system bandwidth, resulting in spatiotemporal sparse sampling. The crux of 4D microscopy super-resolution lies in deciphering biophysically meaningful dynamic processes rather than settling for mere numerical interpolation. However, existing methods predominantly rely on discrete numerical interpolation, which fails to preserve such physical fidelity during global non-linear biological evolutions. To overcome this dilemma, we propose Neural Spatiotemporal Scope (NeurScope), a physics-driven continuous generative paradigm. Diverging from traditional local interpolation, NeurScope utilizes high-frequency lateral observations as physical anchors and introduces a Neural Gaussian Integrator to adaptively model the spatiotemporal integration degradation and Dirac comb sampling inherent in microscopic imaging. By enforcing global manifold alignment within a continuous space, our framework formulates the ill-posed sparse reconstruction as a physical inverse problem strictly constrained by biological dynamics priors. Extensive experiments on cell migration and embryonic division tasks under sparsity (e.g., 6\(\times\) temporal downsampling) demonstrate that NeurScope not only achieves state-of-the-art reconstruction accuracy but also rigorously maintains biophysical consistency in morphological metrics such as volume, surface area, and sphericity, successfully extrapolating valid biological evolutionary logic from fragmented projections.

**The training and inference code will be released soon.**

## Overview
<img width="800" alt="Ai" src="https://github.com/user-attachments/assets/40308712-faad-4b06-b41d-3e6acecafa88">  

**An overview of our proposed SpatimeINR method**
## Get Started
### Dependencies and Installation
- Python 3.11.0
- Pytorch 2.0.1

1. Create Conda Environment
```
conda create --name SpatimeINR python=3.11.0
conda activate SpatimeINR
conda install anaconda::git
```
2. Clone Repo
```
git clone https://github.com/cunminzhao/SpatimeINR.git
```
3. Install Dependencies
```
cd SpatimeINR
pip install -r ./requirement.txt
```

### Dataset
You can refer to the following links to download the datasets of our collected dataset in 
[figshare](https://figshare.com/articles/dataset/Public_Raw_Data/29064530).

### Training  
**Example**: to train yourdata with *SpatimeINR*, you need to keep these data in
* **Structure of data folder**: 
    ```buildoutcfg
    data/
      |yourdata.nii.gz #data.shape=(x,y,z,t)
    Config/
      |config.yaml
    ```

Then run
```
python train.py --config ./Config/config.yaml --save_path "./save" --file "./data/yourdata.nii.gz"
```

you will get your result in 
* **Structure of save folder**: 
    ```buildoutcfg
    save/
      |current.pth
      |latent_code.pt
    ```


### Inference  
Then run，where ```scale``` is the spatial resolution that you can set arbitrarily, and ```scaleT``` is the temporal resolution that you can set arbitrarily. ```yourdata``` is also necessary for the sampling shape.
```
python inference.py --config ./Config/config.yaml --save_path "./save" --file "./data/yourdata.nii.gz" --scale 1 --scaleT 2
```

you will get your result in 
 ```buildoutcfg
    save/
      |current.pth
      |latent_code.pt
      |output.nii.gz #data.shape=(x*scale,y*scale,z*scale,t*scaleT)
 ``` 

## Compute Report
This method was tested on both the NVIDIA RTX 4090 and A100, and will be released soon.

## Acknowledgement
- We appreciate several previous works for their algorithms related/helpful to this project, including [*LIIF*](https://github.com/yinboc/liif), [*RCAN*](https://github.com/AiviaCommunity/3D-RCAN), [*CuNeRF*](https://github.com/NarcissusEx/CuNeRF), and [*SVIN*](https://github.com/yyguo0536/SVIN).

