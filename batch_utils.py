#!/usr/bin/python

import os.path as osp
import os, glob, re
import time
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow

#models = './models_airplanes/cub_cessna'
models = './training_150'

all_models = glob.glob(models+'/*')
all_views = []
#print 'Found models: '
#print all_models
for model in all_models:
    f = open(model+'/views.txt')
    lines = f.readlines()
    all_views.append(lines)
    f.close()


def next_batch(batch):
    output = []
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    output.append([])

    for _ in range(batch):
        model_i = np.random.randint(len(all_models))
        image_i = np.random.randint(1000)

	model = all_models[model_i]
	image = model+'/'+str(image_i)+'.png'
	#print image
	img = Image.open(image)
	angles = all_views[model_i][image_i].split(' ')

        az_one_hot = np.zeros(360)
        az_one_hot.put(int(angles[0]),1)
        el_one_hot = np.zeros(180)
        el_one_hot.put(int(angles[1]),1)
        ti_one_hot = np.zeros(360)
        ti_one_hot.put(int(angles[2]),1)


        img = Image.open(image)

        #background = Image.open('white-background.png')
        #background.paste(img, (background.size[0]/2-img.size[0]/2 + np.random.randint(-200,200),
        #             background.size[1]/2-img.size[1]/2 + np.random.randint(-200,200)), img)
        #background = background.resize((100,100), Image.ANTIALIAS)
	
	background = img
        img = np.array(background.convert('L')).flatten()
	img = img / 255.0 
        az_dist = []
	el_dist = []
	ti_dist = [] 
	
	az_dist = np.concatenate(( np.arange(int(angles[0]),0,-1), np.arange(0,360-int(angles[0])) ))
        if int(angles[1]) < 0:
            el_dist = np.concatenate(( np.arange(90+int(angles[1]),0,-1), np.arange(0,90), np.arange(90, 90+int(angles[1]), -1) ))
	else:
            el_dist = np.concatenate(( np.arange(90-int(angles[1]),90), np.arange(90,0,-1), np.arange(0,90-int(angles[1])) ))
	    
	ti_dist = np.concatenate(( np.arange(int(angles[2]),0,-1), np.arange(0,360-int(angles[2])) ))

	output[0].append(img.tolist())
	output[1].append([int(angles[0])*np.pi/180.0])
	output[2].append([int(angles[1])*np.pi/180.0])
	output[3].append([int(angles[2])*np.pi/180.0])
	output[4].append(az_dist)
        output[5].append(el_dist)
        output[6].append(ti_dist)

    return output


if __name__ == '__main__':
    num = 2
    start = time.time()
    b = next_batch(1)
    end = time.time()
    print(end-start)
    #print b[1], int(b[1][0][0]*180/np.pi), len(b[4][0]), b[4]
    #print b[4][0][int(b[1][0][0]*180/np.pi)]
    #print b[2], int(b[2][0][0]*180/np.pi)+90, len(b[5][0]), b[5]
    #print b[5][0][int(b[2][0][0]*180/np.pi)+90]
    print b[3], int(b[3][0][0]*180/np.pi), len(b[6][0]), b[6]
    print b[6][0][int(b[3][0][0]*180/np.pi)]
    #print b[2], b[5] 
    #print b[3], b[6]
    #for i in range(num):
    #    im = np.array(b[0][i])
    #    im = im.reshape([100,100])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

