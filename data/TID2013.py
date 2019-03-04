#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created TID2013.py by rjw at 19-1-8 in WHU.
"""

from __future__ import absolute_import,division,print_function
import os
import numpy as np

# Define DB information
BASE_PATH = '/media/rjw/Ran-software/dataset/iqa_dataset/tid2013/'
# [ref_idx,dist_types_,ref_imgs_,dist_imgs_,mos_data_,dmos_std]
LIST_FILE_NAME = 'TID2013.txt'


def split_train_test(list_file_name):

    # Get reference / distorted image file lists:
    # d_img_list and score_list
    d_img_list, r_img_list, r_idx_list, score_list, dmos_std = [], [], [], [],[]
    with open(list_file_name, 'r') as listFile:
        for line in listFile:
            # r[ref_idx,dist_types_,ref_imgs_,dist_imgs_,dmos_data_,dmos_std]
            ref_idx, dis_idx, ref, dis, score,std = line.split()
            d_img_list.append(dis)
            r_img_list.append(ref)
            r_idx_list.append(int(ref_idx))
            score_list.append(float(score))
            dmos_std.append(float(std))

    shuff=np.random.permutation(range(25))
    train_file=open(os.path.join(BASE_PATH,'tid2013_train.txt'),'w')
    test_file=open(os.path.join(BASE_PATH,'tid2013_test.txt'),'w')
    Num_tr = 20
    Num_te = 25
    for i in range(Num_tr):
        for j in range(len(r_idx_list)):
            if r_idx_list[j]==shuff[i]:
                file_name=d_img_list[j]
                labels=score_list[j]
                std_=dmos_std[j]
                train_file.write('%s %f %f\n' % (file_name,labels,std_))
    for i in range(Num_tr,Num_te):
        for j in range(len(r_idx_list)):
            if r_idx_list[j]==shuff[i]:
                file_name = d_img_list[j]
                labels = score_list[j]
                std_=dmos_std[j]
                test_file.write('%s %f %f\n' % (file_name, labels,std_))
    train_file.close()
    test_file.close()

if __name__=="__main__":

    list_file_name=os.path.join(BASE_PATH,LIST_FILE_NAME)
    split_train_test(list_file_name)