#!/bin/bash

# Install Chamfer Distance
cd ./extensions/chamfer_dist
python setup.py install --user

# Install emd
cd ../emd
python setup.py install --user

# Install PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Install GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
