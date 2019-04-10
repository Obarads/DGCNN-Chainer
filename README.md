# Dynamic Graph CNN-Chainer
## Introduce
This repository is an implementation of DGCNN[1] in Chainer.

## Installation
Please install Chainer (and cupy if you want to use GPU) beforehand.  
Furthermore, version that I tested operation is described on comments.
```
# chainer version 5.3.0
pip install chainer
# cupy-cuda100 version 5.4.0
pip install cupy-cuda100
```
Also, some extension library is used in some of the code,
```
# Chainer Chemistry version 0.5.0
git clone https://github.com/pfnet-research/chainer-chemistry.git
pip install -e chainer-chemistry
# ChainerEX version 0.0.1
git clone https://github.com/corochann/chainerex.git
pip install -e chainerex
```

## Train
You can simply execute train code with GPU.
```
python train.py -g 0
```

## Result
2019/04/10(YYYY/MM/DD), this implementation is incomplete.
| main/loss | main/accuracy | validation/main/loss | validation/main/accuracy | elapsed_time |
|-|-|-|-|-|
| 0.0147 | 0.9948 | 0.7025 | 0.8906 | 116486.35 |

# Reference
1. [Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon. Dynamic Graph CNN for Learning on Point Clouds. 2018.](https://arxiv.org/abs/1801.07829)
1. [WangYueFt. dgcnn. (access:2019/03/31)](https://github.com/WangYueFt/dgcnn)
2. [corochann. chainer-pointnet. (access:2019/03/31)](https://github.com/corochann/chainer-pointnet)
