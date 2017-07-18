# shapenet-tensorflow

Instructions on getting wieght file (.npy) as shown in test_load.py:
1. Clone https://github.com/ShapeNet/RenderForCNN
2. Go to Off-the-shelf Viewpoint Estimator section in README.md which says to run: 
  cd caffe_models
  sh fetch_model.sh
3. Clone https://github.com/ethereon/caffe-tensorflow.git
  Follow instructions to create npy and py files
  (Optional) Rename it to shapenet_tensor.npy to run with test_load.py


Instructions to Install Caffe with no CPU:
1. Clone https://github.com/BVLC/caffe
2. Follow these steps: https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
