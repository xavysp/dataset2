

import numpy as np
import os
import sys, time
import json
import platform
import shutil
import cv2 as cv
import re

from utili.data_augmentation import augment_data, gamma_data
from utili.utls import *

IS_LINUX = True if platform.system()=='Linux' else False

if __name__ == '__main__':

    print("Below you will find the basic operation to run: \n")
    print("Op 0: Dataset list maker .txt in json")
    print("Op 1. data augmentation")


    # op = input("Please select the operation (Op): ")
    op = -1 #input("Choice one option above, then [Enter] ")
    if op==0:
        base_dir = '/root/workspace/datasets' if IS_LINUX else "C:/Users/xavysp/dataset"
        dataset_name = 'BIPED'
        img_base_dir = 'edges/imgs/train/rgbr/aug'
        gt_base_dir = 'edges/edge_maps/train/rgbr/aug'
        save_file = os.path.join(base_dir, dataset_name)
        files_idcs = []
        simple_list = False
        if simple_list:
            # img_dirs = os.path.join(save_file, img_base_dir)
            # gt_base_dir = os.path.join(save_file, gt_base_dir)
            for full_path in os.listdir(os.path.join(save_file, img_base_dir)):
                file_name = os.path.splitext(full_path)[0]
                files_idcs.append(
                    (os.path.join(img_base_dir + '/' + file_name + '.jpg'),
                     os.path.join(gt_base_dir + '/' + file_name + '.png'),))
        # save files
        else:
            for dir_name in os.listdir(os.path.join(save_file, img_base_dir)):
                # img_dirs = img_base_dir + '/' + dir_name
                img_dirs = img_base_dir + '/' + dir_name
                for full_path in os.listdir(os.path.join(save_file, img_dirs)):
                    file_name = os.path.splitext(full_path)[0]
                    files_idcs.append(
                        (os.path.join(img_dirs + '/' + file_name + '.jpg'),
                         os.path.join(gt_base_dir + '/' + dir_name + '/' + file_name + '.png'),))
        # save files
        print(os.path.join(img_dirs+'/'+file_name+'.jpg'))
        print(os.path.join(gt_base_dir+'/'+dir_name+'/'+file_name+'.png'))
        save_path = os.path.join(save_file, 'train_pair.lst')
        with open(save_path, 'w') as txtfile:
            json.dump(files_idcs, txtfile)

        print("Saved in> ",save_path)

        # Check the files

        with open(save_path) as f:
            recov_data = json.load(f)
        idx = np.random.choice(200, 1)
        tmp_files = recov_data[15]
        img = cv.imread(os.path.join(save_file, tmp_files[0]))
        gt = cv.imread(os.path.join(save_file, tmp_files[1]))
        print(f"Image size {img.shape()}, file name {tmp_files[0]}")
        print(f"GT size {gt.shape()}, file name {tmp_files[1]}")
        # cv_imshow(img, 'rgb image')
        # cv_imshow(gt, 'gt image')


    elif op==1:
        base_dir= '/root/workspace/datasets' if IS_LINUX else '../../dataset'
        dataset = 'BIPED'
        augment_both=True# to augment the input and target
        augment_data(base_dir=base_dir,augment_both=augment_both,dataName=dataset, use_all_augs=True)

    else:
        print("(Please try other options")
        base_dir = 'data/real'
        # save_dir = 'data/res'
        # os.makedirs(save_dir,exist_ok=True)
        list_file =os.listdir(base_dir)
        # i = np.random.randint(len(list_file))
        # img = cv.imread(os.path.join(base_dir,'RGB_107G80.jpg'))# 277, 205
        # img2 = cv.imread(os.path.join(base_dir,list_file[205]))# 277, 205
        # img3 = cv.imread(os.path.join(base_dir,list_file[30]))# 277, 205


        # i2Bm, i2Gm, i2Rm = meanImg(img2)
        # i3Bm, i3Gm, i3Rm = meanImg(img3)

        # print(f"Second image  mean> B {i2Bm},G {i2Gm}, and R {i2Rm}")
        # print(f"third image  mean> B {i3Bm},G {i3Gm}, and R {i3Rm}")
        # oriCon = np.concatenate((img[200,200,:],img2[200,200,:]), axis=0)
        # cv_imshow(img)
        # cv_imshow(img2)
        # cv_imshow(img3)
        # tranImg= meanImg_transform(img)
        #  # tranImg = cv.transform( img, [[[ 0.114, 0.587, 0.299]],[[-0.322, 0.274, 0.596]],
        # #                               [[ 0.311,-0.523, 0.212]]] )
        # tranImg.show()
        # cv_imshow(img, 'rgb image')
        gamma_data(data_dir='data',augment_both=False)



print("Ready")
