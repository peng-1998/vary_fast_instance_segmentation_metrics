# vary_fast_instance_segmentation_metrics
A fast implementation of instance segmentation evaluation metrics based on PyTorch, including AJI, PQ, Dice2, mPQ.
The implementation of these metrics is based on the implementation in [Hover-Net](https://github.com/vqdang/hover_net/blob/master/metrics/stats_utils.py) and has been rewritten using [PyTorch](https://pytorch.org/).

# 中文说明
[中文说明](README_CN.md)

# Usage
Copy the metrics.py file and call it in your preferred way.

# Performance
I tested it using the predicted results generated during my research on cell nucleus instance segmentation and the ground truth from the dataset (MoNuSeg's test set). The images have a resolution of 1000X1000, and the times in the table are the average times for all the images. This implementation uses an RTX 3060 GPU for acceleration.
The runtime tests are as follows:
||AJI  |PQ   |Dice2| all in one|
|:---:|:---:|:---:|:---:|:---:|
|Original Implementation|1.058s|1.027s|1.004s| - |
|This Implementation (GPU)|0.008552s|0.007037s|0.006611s|0.01003s|
|This Implementation (CPU)|0.3013s|0.3024s|0.2991s|0.3135s|

About memory usage:
Due to PyTorch's memory caching mechanism, it is not easy to measure accurately, but here is a rough estimate:
The original implementation used approximately 0.4~1.5GB of memory per image, while this implementation uses less than 100MB of memory on CPU and around 1.9MB of GPU memory.
Here is the memory usage for a single image during the AJI calculation:
|Original Implementation|This Implementation (GPU)（cuda memory） |This Implementation (CPU)|
|:---:|:---:|:---:|
|1140.051MB|1.916MB|52.414MB|
