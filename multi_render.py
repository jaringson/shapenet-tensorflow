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



models = './models_airplanes/cub_cessna'

all_models = os.listdir(models)
#print all_models
batch = 100

time1 = time.time()

for model in all_models:

    angles_file = osp.join(models, model, 'views.txt')
    angles_fout = open(angles_file,'a')


    for i in range(batch):
            model_file = osp.join(BASE_DIR, models, model, 'model.obj')

            azimuth = str(random.randint(0,359))
            elevation = str(random.randint(-89,90))
            tilt = str(random.randint(0,359))
            distance = str(random.randint(2,4)+random.random())

            # MK TEMP DIR
            temp_dirname = tempfile.mkdtemp()

            view_file = osp.join(temp_dirname, 'views_'+str(i)+'.txt')
            view_fout = open(view_file,'w')
            view_fout.write(' '.join([azimuth, elevation, tilt, distance]))
            angles_fout.write(' '.join([azimuth, elevation, tilt, '\n']))
            view_fout.close()

            try:


                render_cmd = '%s %s --background --python %s -- %s %s %s %s %s' % ('~/blender-2.78/blender',
                                blank_file, render_code, model_file, 'xxx', 'xxx', view_file, temp_dirname)
                print render_cmd
                os.system(render_cmd)

                output_img = osp.join(BASE_DIR, models, model, str(i)+ '.png')
                imgs = glob.glob(temp_dirname+'/*.png')
                shutil.move(imgs[0], output_img)
                # CLEAN UP
                shutil.rmtree(temp_dirname)

            except:
                print('render failed. render_cmd: %s' % (render_cmd))

    angles_fout.close()




time2 = time.time()
print 'Time took %0.3f seconds' % ((time2-time1))
