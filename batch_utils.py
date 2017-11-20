#!/usr/bin/python

import os.path as osp
import os, glob, re
import time
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow

#models = './models_airplanes/cub_cessna'
models = './training'

all_models = glob.glob(models+'/*')
all_views = []
#print 'Found models: '
#print all_models
for model in all_models:
    f = open(model+'/views.txt')
    all_views.append(f.readlines())
    f.close()


def next_batch(batch):
    output = []
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
	 
        
	output[0].append(img.tolist())
	output[1].append(int(angles[0]))
	output[2].append(int(angles[1]))
	output[3].append(int(angles[2]))
	#output[1].append(az_one_hot)
        #output[2].append(el_one_hot)
        #output[3].append(ti_one_hot)

    return output


if __name__ == '__main__':
    num = 2
    #start = time.time()
    b = next_batch(50)
    #end = time.time()
    #print(end-start)
    # print b[1], b2], b[3]
    #for i in range(num):
    #    im = np.array(b[0][i])
    #    im = im.reshape([100,100])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

