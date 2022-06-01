# BEVDet


 ![Illustrating the performance of the proposed BEVDet on the nuScenes val set](./resources/nds-fps.png)
 
## News
* \[2022/6/1\] We release the code and models of both BEVDet and BEVDet4D!
* \[2022/4/1\] We propose BEVDet4D to lift the scalable BEVDet paradigm from the spatial-only 3D space to the spatial-temporal 4D space. Technical report is released on [ArXiv](https://arxiv.org/abs/2203.17054).
* \[2022/4/1\] We upgrade the BEVDet paradigm with some modifications to improve its performance and inference speed. [ArXiv](https://arxiv.org/abs/2112.11790).
* \[2021/12/23\] BEVDet is now on [ArXiv](https://arxiv.org/abs/2112.11790).

## Main Results
| Method            | mAP      | NDS     | FPS    |   Download |
|--------|----------|---------|--------|-------------|
| [**BEVDet-Tiny**](configs/bevdet/bevdet-sttiny.py)   | 30.8     | 40.4    | 15.6   | [model](https://drive.google.com/file/d/10innSxqN7NgbktrlfPjWjE7gz-xpbJO_/view?usp=sharing)        |
| [**BEVDet4D-Tiny**](configs/bevdet4d/bevdet4d-sttiny.py) | 33.8     | 47.6    | 15.5   | [model](https://drive.google.com/file/d/1nyQfp7Gt-xbXDzcw5ritmFb8lvPM1H6n/view?usp=sharing)        |
## Get Started
* Please follow the guidelines in the original mmdet3d for preparing the repo and dataset.

i.e. Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection3D. We provide guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with customized dataset](docs/2_new_data_model.md) for beginners. There are also tutorials for [learning configuration systems](docs/tutorials/config.md), [adding new dataset](docs/tutorials/customize_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [Waymo dataset](docs/datasets/waymo_det.md).

* **Prepare dataset specific for BEVDet4D.** (Note: Make sure that data preparation in [nuscenes_det.md](docs/datasets/nuscenes_det.md) has been conducted)
```shell
cd BEVDet/
python tools/data_converter/prepare_nuscenes_for_bevdet4d.py
```

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{huang2022bevdet4d,
  title={BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2203.17054},
  year={2022}
}

@article{huang2021bevdet,
  title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}
```
