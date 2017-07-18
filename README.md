# shapenet-tensorflow

Instructions on getting wieght file (.npy) as shown in test_load.py:

1. Install Caffe (if not installed)
  a. Clone https://github.com/BVLC/caffe
  b. Follow these steps: https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide

2. Clone https://github.com/ShapeNet/RenderForCNN

3. Go to Off-the-shelf Viewpoint Estimator section in README.md which says to run: 
  a. cd caffe_models
  b. sh fetch_model.sh

4. Clone https://github.com/ethereon/caffe-tensorflow.git
  a. Follow instructions to create npy and py files
  b. (Optional) Rename it to shapenet_tensor.npy to run with test_load.py

