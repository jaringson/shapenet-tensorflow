#!/usr/bin/python

import os.path as osp
import os, glob, re
import time
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow
import shutil


backgrounds = './rock_canyon'
all_backgrounds = glob.glob(backgrounds + '/*')


out_main_dir = './rock_canyon_cut/'

final_size = 150

def cut(rate):
    for i, back in enumerate(all_backgrounds):
	#print back
	background = Image.open(back)
	w, h = background.size
	for j in range(rate):
	    x = np.random.randint(0, w-final_size-1)
	    y = np.random.randint(0, h-final_size-1)
	    out = background.crop((x, y, x+final_size, y+final_size))
	    out.save(out_main_dir+str(i) +'_'+str(j)+'.png')

if __name__ == '__main__':
    start = time.time()
    cut(10)
    end = time.time()
    print(end-start)


