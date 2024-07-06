## News
- **2023.05.07** Improve the occupancy baseline by enlarging the input size and using long-term temporal fusion.
- **2023.04.28** Support task of [occupancy prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) .
- **2023.04.27** Equip BEVDet with stereo depth estimation.
- **2023.04.10** Use single head for multi-class prediction.
- **2023.01.12** Support TensorRT-INT8.
* **2022.11.24** A new branch of bevdet codebase, dubbed dev2.0, is released. dev2.0 includes the following features:
1. support **BEVPoolv2**, whose inference speed is up to **15.1 times** the previous fastest implementation of Lift-Splat-Shoot view transformer. It is also far less memory consumption.
 ![bevpoolv2](../../resources/bevpoolv2.png)
 ![bevpoolv2](../../resources/bevpoolv2_performance.png)
2. use the origin of ego coordinate system as the center of the receptive field instead of the Lidar's.
3. **support conversion of BEVDet from pytorch to TensorRT.**
4. use the long term temporal fusion as SOLOFusion.
5. train models with ema and without CBGS by default.
6. use key frame for temporal fusion.

* **2022.08.26** A blog in chinese for code explanation. [BEVDet系列源码解读](https://zhuanlan.zhihu.com/p/557613388)
* **2022.08.15** Support FP16 training for BEVDet series that with image view backbone type of ResNet.
* **2022.07.29** Support BEVDepth.
* **2022.07.26** Add configs and pretrained models of bevdet-r50 and bevdet4d-r50.
* **2022.07.13** Support bev-pool proposed in [BEVFusion](https://github.com/mit-han-lab/bevfusion), which will speed up the training process of bevdet-tiny by +25%.
* **2022.07.08** Support visualization remotely! Please refer to [Get Started](https://github.com/HuangJunJie2017/BEVDet#get-started) for usage.
* **2022.06.29** Support acceleration of the Lift-Splat-Shoot view transformer! Please refer to \[[Technical Report](https://arxiv.org/abs/2112.11790)\] for detailed introduction and [Get Started](https://github.com/HuangJunJie2017/BEVDet#get-started) for testing BEVDet with acceleration.
* **2022.06.01** We release the code and models of both BEVDet and BEVDet4D!
* **2022.04.01** We propose BEVDet4D to lift the scalable BEVDet paradigm from the spatial-only 3D space to the spatial-temporal 4D space. Technical report is released on arixv. \[[BEVDet4D](https://arxiv.org/abs/2203.17054)\].
* **2022.04.01** We upgrade the BEVDet paradigm with some modifications to improve its performance and inference speed. **Thchnical report of BEVDet has been updated.** \[[BEVDetv1](https://arxiv.org/abs/2112.11790)\].
* **2021.12.23** BEVDet is now on arxiv. \[[BEVDet](https://arxiv.org/abs/2112.11790)\].
