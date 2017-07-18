# shapenet-tensorflow

Instructions on getting wieght file (.npy) as shown in test_load.py (Assumed that Tensorflow is installed):

1. Install Caffe (if not installed) <br />
	a. Clone https://github.com/BVLC/caffe <br />
  b. Follow these steps: https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide

2. Clone https://github.com/ShapeNet/RenderForCNN

3. Go to Off-the-shelf Viewpoint Estimator section in README.md which says to run: 
    
    <pre>
    cd caffe_models
    sh fetch_model.sh
    </pre>

4. Clone https://github.com/ethereon/caffe-tensorflow.git <br />
  a. Follow instructions to create npy and py files found under examples/mnist/ <br />
  b. (Optional) Rename it to shapenet_tensor.npy to run with test_load.py

<br />
<br />

Do the steps below if you need to reuse the convert script (done in step 4 above):

Change files into correct Tensorflow version
  1. Go to https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility
  2. Change branch to your version of Tensorflow
  3. Copy tf_upgrade.py script
  4. Run tf_upgrade.py on .py file (shown here as shapenet_tensor.py) create in step 4
  5. Run tf_upgrade on network.py file in caffe-tensorflow/kaffe/tensorflow/network.py to fit your version of tensorflow.
  
