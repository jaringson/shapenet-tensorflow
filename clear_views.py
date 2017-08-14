#!/usr/bin/python

import os.path as osp
import sys
import os, tempfile, glob, shutil
import random
import time

BASE_DIR = osp.dirname(__file__)
sys.path.append(osp.join(BASE_DIR,'../'))



blank_file = 'blank.blend'
render_code = 'render_model_views.py'

# MK TEMP DIR
temp_dirname = tempfile.mkdtemp()

models = './models_airplanes/cub_cessna'

all_models = os.listdir(models)
#print all_models
batch = 2

for model in all_models:

    angles_file = osp.join(models, model, 'views.txt')
    angles_fout = open(angles_file,'w')
    angles_fout.close()
