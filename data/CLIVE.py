#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created CLIVE.py by rjw at 19-1-11 in WHU.
"""

import numpy as np
import scipy.io as sio
import os

base_path="/media/rjw/Ran-software/dataset/iqa_dataset/CLIVE/"
image_file = "Data/AllImages_release.mat"
mos_file = "Data/AllMOS_release.mat"
mos_std_file="Data/AllStdDev_release.mat"

images = sio.loadmat(base_path+image_file)['AllImages_release'][7:]
images = images.swapaxes(1,0).reshape(-1,)

mos = sio.loadmat(base_path+mos_file)['AllMOS_release'][0][7:]
mos_std = sio.loadmat(base_path+mos_std_file)['AllStdDev_release'][0][7:]

train_file = open(os.path.join(base_path, 'clive_train.txt'), 'w')
test_file = open(os.path.join(base_path, 'clive_test.txt'), 'w')

total_num = len(images)
shuff=np.random.permutation(total_num)

for i in range(total_num):
    index=shuff[i]
    if i<total_num*0.8:
        # print(type('Images/'))
        # print(type(images[index]))
        train_file.write("%s %f %f\n" % ('Images/'+images[index][0],mos[index],mos_std[index]))
    else:
        test_file.write("%s %f %f\n" % ('Images/'+images[index][0],mos[index],mos_std[index]))

train_file.close()
test_file.close()



