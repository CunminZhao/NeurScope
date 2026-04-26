## Neural Spatiotemporal Manifold Extrapolation for 4D Microscopy Super-Resolution

Source code of the paper "xxx"

### Installation
```
git clone https://github.com/cunminzhao/NeurScope.git
cd NeurScope
pip install -r ./requirement.txt
```

### Training NeurScope
```
python train.py --config ./Config/config.yaml --save_path "./save" --file "./data/yourdata.nii.gz"
```

### Inference  
```scale``` is the spatial resolution that you can set arbitrarily, and ```scaleT``` is the temporal resolution that you can set arbitrarily. ```yourdata``` is also necessary for the sampling shape.
```
python inference.py --config ./Config/config.yaml --save_path "./save" --file "./data/yourdata.nii.gz" --scale 1 --scaleT 4
```

## Compute Report
This method was tested on the NVIDIA RTX 4090, A100 and A6000, and will be released soon.

## Acknowledgement
- We appreciate several previous works for their algorithms related/helpful to this project, including [*LIIF*](https://github.com/yinboc/liif), [*RCAN*](https://github.com/AiviaCommunity/3D-RCAN), [*CuNeRF*](https://github.com/NarcissusEx/CuNeRF), and [*SVIN*](https://github.com/yyguo0536/SVIN).

