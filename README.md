# Scalable Unseen Objects 6-DoF Absolute Pose Estimation with Robotic Integration

This is the PyTorch implementation of paper **[SinRef-6D](https://paperreview99.github.io/SinRef-6DoF-Robotic)** published in <b>*IEEE TRO*</b> by <a href="https://cnjliu.github.io/">J. Liu</a>, <a href="http://robotics.hnu.edu.cn/info/1071/1265.htm">W. Sun</a>, <a href="https://github.com/CNJianLiu/SinRef-6D">K. Zeng</a>, <a href="https://github.com/CNJianLiu/SinRef-6D">J. Zheng</a>, <a href="https://github.com/CNJianLiu/SinRef-6D">H. Yang</a>, <a href="https://sites.google.com/view/rahmaniatlu">H. Rahmani</a>, <a href="https://ajmalsaeed.net/">A. Mian</a>, and <a href="https://github.com/CNJianLiu/SinRef-6D">L. Wang</a>. SinRef-6D is a single reference view-based CAD model-free novel object 6D pose estimation method, which is **simple yet effective** and has **strong scalability for practical applications**.

Given a **single RGB-D reference view** of an unseen object in a **default robot manipulation viewpoint**, we aim to predict its 6-DoF absolute pose from **any query view**.

![Fig1](image/teaser.jpg)

## Real-World Demo
SinRef-6D deployment in real-world robotic manipulation scenarios. Notably, the reference view is **not carefully selected**. We **randomly** select a normal view (free of occlusion, with minimal self-occlusion, and resembling a typical robotics manipulation view) using an Intel RealSense L515 RGB-D camera as the reference view.

![Fig2](image/demo.gif)

To the best of our knowledge, we are the first to present a method for novel object 6D absolute pose estimation using only a single reference view in real-world robotic manipulation scenarios. This approach simultaneously eliminates the need for object *CAD models*, *dense reference views*, and *model retraining*, offering enhanced efficiency and scalability while demonstrating **strong generalization to potential real-world robotic applications**.

More robotic demos can be seen at our **[Project Page](https://paperreview99.github.io/SinRef-6DoF-Robotic)**.

## Citation
If you find our work useful, please consider citing:
```latex
@article{2026SinRef-6D,
  author={Liu, Jian and Sun, Wei and Zeng, Kai and Zheng, Jin and Yang, Hui and Rahmani, Hossein and Mian, Ajmal and Wang, Lin},
  title={Scalable Unseen Object 6-DoF Absolute Pose Estimation with Robotic Integration},
  journal={IEEE Transactions on Robotics},
  year={2026}
}
```

## Licence

This project is licensed under the terms of the MIT license.
