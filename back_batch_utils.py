#!/usr/bin/python

import os.path as osp
import os, glob, re, sys
import time
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow


final_size = 150

models = './models_airplanes/cub_cessna'
#models = './training_150'

backgrounds = './rock_canyon_cut'

all_models = glob.glob(models+'/*')
all_views = []
all_backgrounds = glob.glob(backgrounds + '/*')
#print 'Found models: '
#print all_models
for model in all_models:
    f = open(model+'/views.txt')
    lines = f.readlines()
    all_views.append(lines)
    f.close()

num_models = 1000

def next_batch(batch, testing=False):
    output = []
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    output.append([])

    for i in range(batch):
	stay_err = True
	while_i = 0
	while stay_err:
	    while_i += 1
	    try:
		model_i = np.random.randint(len(all_models)-1)
		image_i = np.random.randint(num_models-50)
		if testing:
		    model_i = np.random.randint(len(all_models)) 
		    image_i = np.random.randint(num_models-50, 1000)
		    if model_i == len(all_models) - 1:
			image_i = np.random.randint(1000)

		model = all_models[model_i]
		image = model+'/'+str(image_i)+'.png'
		#print image
		#img = Image.open(image)

		angles = all_views[model_i][image_i].split(' ')
		
		azim = np.radians(int(angles[0]))
		ele = np.radians(int(angles[1]))
		tilt = np.radians(int(angles[2]))	

		extra = np.radians(np.random.randint(-90,90))	

		Rt = np.array([[1,0,0],
			  [0,np.cos(-tilt),-np.sin(-tilt)],
			  [0,np.sin(-tilt),np.cos(-tilt)]])

		Re = np.array([[np.cos(-ele),0,np.sin(-ele)],
			  [0,1,0],
			  [-np.sin(-ele),0,np.cos(-ele)]])

		Ra = np.array([[np.cos(azim),-np.sin(azim),0],
			  [np.sin(azim),np.cos(azim),0],
			  [0,0,1]])

		# rotates the plane to face you instead of away from you
		Rrev = np.array([[-1,0,0], 
			  [0,-1,0],
			  [0,0,1]])

		Rextra = np.array([[1,0,0],
			  [0,np.cos(-extra),-np.sin(-extra)],
			  [0,np.sin(-extra),np.cos(-extra)]])
		
		R = Rextra.dot(Rrev).dot(Rt).dot(Re).dot(Ra)

		pitch = -np.arcsin(R[2,0])
		roll = np.arctan2(R[2,1],R[2,2])
		yaw = np.arctan2(R[1,0],R[0,0])

		yaw_d = int(np.degrees(yaw))
		pitch_d = int(np.degrees(pitch))
		roll_d = int(np.degrees(roll))
		extra_d = int(np.degrees(extra))
		
		img = Image.open(image)
		img = img.rotate(extra_d)
		img = img.resize((final_size-50,final_size-50))
		
		#background = Image.open(np.random.choice(all_backgrounds))
		rand_back = all_backgrounds[np.random.randint(len(all_backgrounds))]  
		background = Image.open(rand_back)
		#w, h = background.size
		#x = np.random.randint(0, w-final_size-1)
		#y = np.random.randint(0, h-final_size-1)

		#background = background.crop((x, y, x+final_size, y+final_size))
		background.paste(img, (background.size[0]/2-img.size[0]/2 + np.random.randint(-20,20), 
				       background.size[1]/2-img.size[1]/2 + np.random.randint(-20,20)),img)

		#background.save('./test_background/'+str(i)+'.png')
		#img = np.array(background.convert('L')).flatten()
		img = np.array(background).flatten() / 255.0 
		#print img.shape
			
		yaw_dist = []
		pitch_dist = []
		roll_dist = [] 

		
		if yaw_d < 0:
		    yaw_dist = np.concatenate(( np.arange(180+yaw_d,0,-1), np.arange(0,180), np.arange(180, 180+yaw_d, -1) ))	
		else:
		    yaw_dist = np.concatenate(( np.arange(180-yaw_d,180), np.arange(180,0,-1), np.arange(0,180-yaw_d) ))
		
		if pitch_d  < 0:
		    pitch_dist = np.concatenate(( np.arange(90+pitch_d,0,-1), np.arange(0,90), np.arange(90, 90-pitch_d) ))
		else:
		    pitch_dist = np.concatenate(( np.arange(90+pitch_d,90,-1), np.arange(90,0,-1), np.arange(0,90-pitch_d) ))
		    
		if roll_d < 0:
		    roll_dist = np.concatenate(( np.arange(180+roll_d,0,-1), np.arange(0,180), np.arange(180, 180+roll_d, -1) ))	
		else:
		    roll_dist = np.concatenate(( np.arange(180-roll_d,180), np.arange(180,0,-1), np.arange(0,180-roll_d) ))

		output[0].append(img.tolist())
		output[1].append([yaw])
		output[2].append([pitch])
		output[3].append([roll])
		output[4].append(yaw_dist)
		output[5].append(pitch_dist)
		output[6].append(roll_dist)
		stay_err = False
		#temp = stay
	    except:
		e = sys.exc_info()
		print while_i, e
		stay_err = True
		if while_i == 10:
		    print 'Tried 10 times'
		    return

    return output


if __name__ == '__main__':
    num = 2
    start = time.time()
    b = next_batch(1)
    end = time.time()
    print(end-start)
    #print b[1], int(b[1][0][0]*180/np.pi), len(b[4][0]), b[4]
    #print b[4][0][int(b[1][0][0]*180/np.pi)+180]
    print b[2], int(b[2][0][0]*180/np.pi), len(b[5][0]), b[5]
    print b[5][0][int(b[2][0][0]*180/np.pi)+90]
    #print b[3], int(b[3][0][0]*180/np.pi), len(b[6][0]), b[6]
    #print b[6][0][int(b[3][0][0]*180/np.pi)+180]
    #print b[2], b[5] 
    #print b[3], b[6]
    #for i in range(num):
    #    im = np.array(b[0][i])
    #    im = im.reshape([100,100])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

