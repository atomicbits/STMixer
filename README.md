# STMixer
This repository gives the official PyTorch implementation of [STMixer: A One-Stage Sparse Action Detector](https://arxiv.org/abs/2303.15879) (CVPR 2023)

## Installation
- PyTorch == 1.8 or 1.12 (other versions are not tested)
- tqdm
- yacs
- opencv-python
- tensorboardX
- SciPy
- fvcore
- timm
- iopath

## Data Preparation
Please refer to [ACAR-Net repo DATA.md](https://github.com/Siyu-C/ACAR-Net/blob/master/DATA.md)  for AVA dataset preparation.

## Model Zoo
| Backbone          | Config | Pre-train Model | Frames | Sampling Rate | Model |
|-------------------|:------:|:---------------:|:------:|:-------------:|:-----:|
| SlowOnly-R50      |   [cfg](https://github.com/MCG-NJU/STMixer/blob/main/config_files/PySlowonly-R50-K400-4x16.yaml)     |       [K400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_4x16_R50.pkl)      |    4   |       16      |  [Link]() |
| SlowFast-R50      |   [cfg](https://github.com/MCG-NJU/STMixer/blob/main/config_files/PySlowfast-R50-K400-8x8.yaml)      |       [K400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)      |    8   |       8       |  [Link]() |
| SlowFast-R101-NL  |   [cfg](https://github.com/MCG-NJU/STMixer/blob/main/config_files/PySlowfast-R101-NL-K600-8x8.yaml)  |       [K600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_32x2_R101_50_50.pkl)      |    8   |       8       |  [Link]() |
| ViT-B(VideoMAE)   |   TODO  |       K400      |   16   |       4       |  TODO |
| ViT-B(VideoMAEv2) |   TODO  |    K701+K400    |   16   |       4       |  TODO |

## Train
python -m torch.distributed.launch --nproc_per_node=8 train_net.py --config-file "config_files/config_file.yaml" --transfer --no-head --use-tfboard

## Val
python -m torch.distributed.launch --nproc_per_node=8 test_net.py --config-file "config_files/config_file.yaml" MODEL.WEIGHT "/path/to/model"

