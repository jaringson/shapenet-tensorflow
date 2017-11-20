#!/usr/bin/python

import os.path as osp
import os, glob, re
import time
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow
import shutil

models = './models_airplanes/cub_cessna'

all_models = glob.glob(models+'/*')
all_views = []
#print 'Found models: '
#print all_models

out_main_dir = 'training'
for model in all_models:
    print model
    out_dir = os.path.join(out_main_dir, model.split('/')[3]) 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    shutil.copyfile(osp.join(model,'views.txt'),osp.join(out_dir,'views.txt'))
    for i in range(1000):
	file = os.path.join(model,str(i)+'.png')
	img = Image.open(file)
	
	background = Image.open('white-background.png')
	background.paste(img, (background.size[0]/2-img.size[0]/2 + np.random.randint(-200,200),
		background.size[1]/2-img.size[1]/2 + np.random.randint(-200,200)), img)
	background = background.resize((100,100), Image.ANTIALIAS)
	
	outfile = os.path.join(out_dir,str(i)+'.png')
	background.save(outfile)
	
    





