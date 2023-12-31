# vary_fast_instance_segmentation_metrics
一个基于PyTorch的快速实现的核实例分割评估指标，包括AJI，PQ，Dice2，mPQ。
这些指标的实现基于[Hover-Net](https://github.com/vqdang/hover_net/blob/master/metrics/stats_utils.py)的实现，并使用[PyTorch](https://pytorch.org/)进行了重写。

## 使用
拷贝`metrics.py`文件，然后按你喜欢的方式调用。

## 性能
我使用我在细胞核实例分割研究时产生的预测结果与其数据集（MoNuSeg的测试集）的ground truth进行了测试，这些图像具有1000X1000的分辨率，表中的时间是所有图像的平均时间。本实现使用了RTX 3060 GPU用于加速。
运行时间测试如下：
||AJI  |PQ   |Dice2|
|:---:|:---:|:---:|:---:|
|原始实现|1.095s|1.061s|1.063s|
|本实现GPU|0.082s|0.087s|0.027s|
|本实现CPU|0.369s|0.350s|0.336s|
由于PyTorch有内存显存复用机制，不便统计，但这里有一个大致的估计：
原始实现在每张图像上使用了大约0.4~1.5GB的内存，而本实现在CPU上的内存占用在100MB以内，而在GPU上的显存占用在1.9MB左右。
这里给出一张图片在AJI计算过程中的内存占用情况：
|原始实现|本实现GPU（显存）|本实现CPU|
|:---:|:---:|:---:|
|1140.051MB|1.916MB|52.414MB|
