#!/usr/bin/python

import os.path as osp
import os, glob, re
import random
import time
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow

models = './models_airplanes/cub_cessna'

all_models = os.listdir(models)


def next_batch(batch):
    output = []
    output.append([])
    output.append([])
    output.append([])
    output.append([])

    for _ in range(batch):
        model = random.choice(all_models)
        image = random.choice(glob.glob(models+'/'+ model + '/*.png' ))

        file_name = re.findall('\d+.png', image)
        line_num = re.findall('\d+', file_name[0])
        # print image

        labels = 'views.txt'


        fp = open(osp.join(models,model,labels), 'r')
        lines = fp.readlines()

        # print lines
        # print int(line_num[0])

        all_angles = lines[int(line_num[0])]
        all_angles = all_angles.split(' ')
        # print all_angles

        az_one_hot = np.zeros(360)
        az_one_hot.put(int(all_angles[0]),1)
        el_one_hot = np.zeros(360)
        el_one_hot.put(int(all_angles[1])+89,1)
        ti_one_hot = np.zeros(360)
        ti_one_hot.put(int(all_angles[2]),1)

        fp.close()

        img = Image.open(image)

        background = Image.open('white-background.png')
        background.paste(img, (background.size[0]/2-img.size[0]/2 + random.randint(-200,200),
                            background.size[1]/2-img.size[1]/2 + random.randint(-200,200)), img)
        background = background.resize((100,100), Image.ANTIALIAS)

        img = np.array(background.convert('L')).flatten()

        output[0].append(img.tolist())
        output[1].append(az_one_hot)
        output[2].append(el_one_hot)
        output[3].append(ti_one_hot)

    return output


if __name__ == '__main__':
    num = 2
    b = next_batch(2)
    # print b[1], b[2], b[3]
    for i in range(num):
        im = np.array(b[0][i])
        im = im.reshape([100,100])
        imsave(str(i)+'.png', im)
        #imshow(im)
