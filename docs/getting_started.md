Modify from the official mmdet3d [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md)

# Prerequisites
BEVDet is developed with the following version of modules.
- Linux or macOS (Windows is not currently officially supported)
- Python 3.7
- PyTorch 1.9.0
- CUDA 11.3.1 
- GCC 7.3.0
- MMCV==1.3.13
- MMDetection==2.14.0
- MMSegmentation==0.14.1


# Installation

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n bevdet python=3.7 -y
conda activate bevdet
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
conda install -c pytorch pytorch torchvision -y
```

**c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/).**
```shell
pip install mmcv-full==1.3.13
```

**d. Install [MMDetection](https://github.com/open-mmlab/mmdetection).**

```shell
pip install mmdet==2.14.0
```

**e. Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).**

```shell
pip install mmsegmentation==0.14.1
```


**f. Clone the BEVDet repository.**

```shell
git clone https://github.com/HuangJunJie2017/BEVDet.git
cd BEVDet
```

**g.Install build requirements and then install BEVDet.**

```shell
pip install -v -e .  # or "python setup.py develop"
```


## A from-scratch setup script

Here is a full script for setting up MMdetection3D with conda.

```shell
conda create -n bevdet python=3.7 -y
conda activate bevdet

# install latest PyTorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install mmcv
pip install mmcv-full==1.3.13

# install mmdetection
pip install mmdet==2.14.0

# install mmsegmentation
pip install mmsegmentation==0.14.1

# install BEVDet
git clone https://github.com/HuangJunJie2017/BEVDet.git
cd BEVDet
pip install -v -e .
```

# Data Preparation

**a. Please refer to [nuScenes](docs/datasets/nuscenes_det.md) for initial preparation.**

**b. Prepare dataset specific for BEVDet4D.**
```shell
python tools/data_converter/prepare_nuscenes_for_bevdet4d.py
```