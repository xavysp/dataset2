

import numpy as np
import os
import cv2 as cv
import shutil
import imutils

from utili.utls import image_normalization, gamma_correction, make_dirs, cv_imshow

def cv_imshow(img,title='image'):
    print(img.shape)
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def scale_img(img,scl=1.):

    scaled_img = cv.resize(img, dsize=(0,0),fx=scl,fy=scl)
    return scaled_img


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def experimental_rotation(img, angle=90):
    # rot_img = imutils.rotate(img,degree) # to rotate but not ensure taken all previous image data
    # keep image shape
    rot_img = imutils.rotate_bound(img,angle) # to rotate but ensure taken all previous image data
    #
    return rot_img

def rotated_img_extractor(x=None, gt=None,img_width=None, img_height=None,i=None, two_data=False):

    if two_data:
        if img_width==img_height:
            # for images whose sizes are the same

            if i % 90 == 0:
                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                # print("just for check 90: ", i)

            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                    rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                    rot_gt = rot_gt[100:720 - 100, 100:720 - 100]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (75, 75), (720 - 75, 720 - 75), (0, 0, 255), (2))
                    rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                    rot_gt = rot_gt[75:720 - 75, 75:720 - 75]
                    # print("just for check 19: ", i, rot_x.shape)

                else:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(95,95),(720-95,720-95),(0,0,255),(2) )
                    rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                    rot_gt = rot_gt[95:720 - 95, 95:720 - 95]
                    # print("just for check 19: ", i, rot_x.shape)

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(85,85),(720-85,720-85),(0,0,255),(2) )
                    rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                    rot_gt = rot_gt[85:720 - 85, 85:720 - 85]
                    # print("just for check 23: ", i, rot_x.shape)
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    rot_gt = rot_gt[105:720 - 105, 105:720 - 105]
                    # print("just for check 23:", i, rot_x.shape)

            return rot_x, rot_gt
        else:
            # # for images whose sizes are ***not*** the same *********************************
            img_size = img_width if img_width < img_height else img_height
            if i % 90 == 0:
                if i==180:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+250, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[10:img_size-90, 10:img_size+110, :]
                    rot_gt = rot_gt[10:img_size-90, 10:img_size+110]
                    # cv.rectangle(a, (10, 10), (img_size+110, img_size-90), (0, 0, 255), (2))
                    # print("just for check 90: ", i)

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 450, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height + 450, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[100:img_size + 200, 300:img_size + 200, :]
                    rot_gt = rot_gt[100:img_size + 200, 300:img_size + 200]
                    # cv.rectangle(a, (300, 100), (img_size+200, img_size+200), (0, 0, 255), (2))

            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+i+5, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+i+5, img_width))
                    # a = np.copy(rot_x)
                    # #                 x    y             x           y
                    # cv.rectangle(a, (275, 275), (img_size+55, img_size+55), (0, 0, 255), (2))
                    #                   y                   x
                    rot_x = rot_x[275:img_size+55, 275:img_size+55, :]
                    rot_gt = rot_gt[275:img_size+55, 275:img_size+55]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+i, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+i, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (355, 100), (img_size+205, img_size-50), (0, 0, 255), (2))
                    rot_x = rot_x[100:img_size-50,355:img_size+205, :]
                    rot_gt = rot_gt[100:img_size-50,355:img_size+205]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i==19:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+200, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+200, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                    rot_x = rot_x[150:img_size-70, 150:img_size+30, :]
                    rot_gt = rot_gt[150:img_size-70, 150:img_size+30]
                    # print("just for check 19: ", i, rot_x.shape)

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (400, 115), (img_size+180, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size-105, 400:img_size+180, :]
                    rot_gt = rot_gt[115:img_size-105, 400:img_size+180]
                    # print("just for check 19: ", i, rot_x.shape)

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+i+200, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+i, img_width))
                    # a = rot_x
                    # cv.rectangle(a, (95, 50), (img_size+75, img_size-170), (0, 0, 255), (2))
                    rot_x = rot_x[50:img_size-170, 95:img_size+75, :]
                    rot_gt = rot_gt[50:img_size-170, 95:img_size+75]
                    # print("just for check 23: ", i, rot_x.shape)
                elif i==207:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (200, 185), (img_size + 160, img_size - 95), (0, 0, 255), (2))
                    rot_x = rot_x[185:img_size - 95, 200:img_size + 160, :]
                    rot_gt = rot_gt[185:img_size - 95, 200:img_size + 160]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (390, 115), (img_size+170, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size-105, 390:img_size+170, :]
                    rot_gt = rot_gt[115:img_size-105, 390:img_size+170]

            return rot_x,rot_gt
    else:
        # For  NIR imagel but just NIR (ONE data)
        if img_height==img_width:

            if i % 90 == 0:
                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                # print("just for check 90: ", i)

            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                    rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (75, 75), (720 - 75, 720 - 75), (0, 0, 255), (2))
                    rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                    # print("just for check 19: ", i, rot_x.shape)

                else:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(95,95),(720-95,720-95),(0,0,255),(2) )
                    rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                    # print("just for check 19: ", i, rot_x.shape)

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(85,85),(720-85,720-85),(0,0,255),(2) )
                    rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                    # print("just for check 23: ", i, rot_x.shape)
                elif i==207:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    # print("just for check 23:", i, rot_x.shape)
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    # print("just for check 23:", i, rot_x.shape)
            else:
                print("Error line 221 in dataset_manager")
                return

        else:
            # when the image size are not the same
            img_size = img_width if img_width < img_height else img_height
            if i % 90 == 0:
                if i == 180:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[10:img_size - 90, 10:img_size + 110, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 450, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[100:img_size + 200, 300:img_size + 200, :]

            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i + 5, img_width))
                    # a = np.copy(rot_x)
                    # #                 x    y             x           y
                    # cv.rectangle(a, (275, 275), (img_size+55, img_size+55), (0, 0, 255), (2))
                    #                   y                   x
                    rot_x = rot_x[275:img_size + 55, 275:img_size + 55, :]
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (355, 100), (img_size+205, img_size-50), (0, 0, 255), (2))
                    rot_x = rot_x[100:img_size - 50, 355:img_size + 205, :]
                elif i == 19:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 200, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                    rot_x = rot_x[150:img_size - 70, 150:img_size + 30, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (400, 115), (img_size+180, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size - 105, 400:img_size + 180, :]

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i + 200, img_width))
                    # a = rot_x
                    # cv.rectangle(a, (95, 50), (img_size+75, img_size-170), (0, 0, 255), (2))
                    rot_x = rot_x[50:img_size - 170, 95:img_size + 75, :]
                elif i == 207:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (200, 185), (img_size + 160, img_size - 95), (0, 0, 255), (2))
                    rot_x = rot_x[185:img_size - 95, 200:img_size + 160, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (390, 115), (img_size+170, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size - 105, 390:img_size + 170, :]
        return rot_x, None

def split_data(data_dir,augment_both=True):
    # split data and copy real image to aug dir
    img_dir = data_dir[0]
    gt_dir = data_dir[1]
    img_aug_dir= os.path.join(img_dir,'aug')
    _=make_dirs(img_aug_dir)
    if augment_both and gt_dir is not None:
        gt_aug_dir = os.path.join(gt_dir,'aug')
        _ = make_dirs(gt_aug_dir)
    elif not augment_both and gt_dir is not None:
        raise NotImplementedError('In single augmentation')

    x_list = os.listdir(os.path.join(img_dir, 'real'))
    x_list.sort()
    n = len(x_list)
    if augment_both:
        gt_list = os.listdir(os.path.join(gt_dir, 'real'))
        gt_list.sort()
        n = len(gt_list) if len(x_list) == len(gt_list) else 0

    # real folder copy to aug dir
    shutil.copytree(os.path.join(img_dir, 'real'),img_aug_dir+'/real')
    if augment_both:
        shutil.copytree(os.path.join(gt_dir, 'real'), gt_aug_dir+'/real')

    # splitting up
    tmp_img = cv.imread(os.path.join(
        os.path.join(img_dir, 'real'), x_list[0]))
    img_width = tmp_img.shape[1]
    img_height = tmp_img.shape[0]

    x_p1_dir = os.path.join(img_aug_dir, 'p1')
    x_p2_dir = os.path.join(img_aug_dir, 'p2')
    _= make_dirs(x_p1_dir)
    _= make_dirs(x_p2_dir)

    if augment_both:
        gt_p1_dir = os.path.join(gt_aug_dir, 'p1')
        gt_p2_dir = os.path.join(gt_aug_dir, 'p2')
        _ = make_dirs(gt_p1_dir)
        _ = make_dirs(gt_p2_dir)

    for i in range(n):
        x_tmp = cv.imread(os.path.join(
            os.path.join(img_dir, 'real'), x_list[i]))
        x_tmp1 = x_tmp[:, 0:img_height, :]
        x_tmp2 = x_tmp[:, img_width - img_height:img_width, :]
        cv.imwrite(os.path.join(x_p1_dir,x_list[i]), x_tmp1)
        cv.imwrite(os.path.join(x_p2_dir,x_list[i]), x_tmp2)

        if augment_both:
            gt_tmp = cv.imread(os.path.join(
            os.path.join(gt_dir, 'real'), gt_list[i]))
            gt_tmp1 = gt_tmp[:, 0:img_height]
            gt_tmp2 = gt_tmp[:, img_width - img_height:img_width]
            cv.imwrite(os.path.join(gt_p1_dir, gt_list[i]), gt_tmp1)
            cv.imwrite(os.path.join(gt_p2_dir, gt_list[i]), gt_tmp2)
            print('saved image: ', x_list[i], gt_list[i])
        else:
            print('saved image: ', x_list[i])

    print('...splitting up augmentation done!')

    if augment_both:
        print('data saved in: ', os.listdir(gt_aug_dir), 'and in',os.listdir(img_aug_dir))
        data_dirs = [img_aug_dir, gt_aug_dir]
        return data_dirs
    else:
        print('data saved in: ', os.listdir(img_aug_dir))
        data_dirs=[img_aug_dir,None]
        return data_dirs

def rotate_data(data_dir, augment_both=True):

    X_dir = data_dir[0]
    GT_dir = data_dir[1]
    x_folders = os.listdir(X_dir)
    x_folders.sort()
    if augment_both:
        gt_folders = os.listdir(GT_dir)
        gt_folders.sort()
        if not x_folders ==gt_folders:
            raise NotImplementedError('gt and x folders not match')

    #         [19, 46,   57, 90, 114, 138, 161,180, 207, 230,  247  270, 285,  322, 342 ]
    degrees = [19, 23*2,19*3,90,19*6,23*6,23*7,180,23*9,23*10,19*13,270,19*15,23*14,19*18]
    print('Folders for working: ',x_folders)
    for folder_name in x_folders:

        x_aug_list = os.listdir(os.path.join(X_dir, folder_name))
        x_aug_list.sort()
        n = len(x_aug_list)
        if augment_both:
            gt_aug_list = os.listdir(os.path.join(GT_dir, folder_name))
            gt_aug_list.sort()
            n = len(gt_aug_list) if len(x_aug_list) == len(gt_aug_list) else None

        tmp_img = cv.imread(os.path.join(X_dir,
                                         os.path.join(folder_name, x_aug_list[1])))
        img_width = tmp_img.shape[1]
        img_height = tmp_img.shape[0]

        for i in (degrees):
            if folder_name == 'p1':
                current_X_dir = X_dir + '/p1_rot_' + str(i)
            elif folder_name == 'p2':
                current_X_dir = X_dir + '/p2_rot_' + str(i)
            elif folder_name == 'real':
                current_X_dir = X_dir + '/real_rot_' + str(i)
            else:
                print('error')
                return
            if augment_both:
                if folder_name == 'p1':
                    current_GT_dir = GT_dir + '/p1_rot_' + str(i)
                elif folder_name == 'p2':
                    current_GT_dir = GT_dir + '/p2_rot_' + str(i)
                elif folder_name == 'real':
                    current_GT_dir = GT_dir + '/real_rot_' + str(i)
                else:
                    print('error')
                    return
                _ = make_dirs(current_GT_dir)

            _=make_dirs(current_X_dir)
            for j in range(n):

                tmp_x = cv.imread(os.path.join(X_dir,
                                               os.path.join(folder_name, x_aug_list[j])))
                tmp_gt = cv.imread(os.path.join(GT_dir,
                                                os.path.join(folder_name, gt_aug_list[j]))) if augment_both else None
                rot_x, rot_gt = rotated_img_extractor(tmp_x, tmp_gt, img_width, img_height, i, two_data=augment_both)

                cv.imwrite(os.path.join(current_X_dir, x_aug_list[j]), rot_x)
                tmp_imgs = rot_x
                if augment_both and rot_gt is not None:
                    cv.imwrite(os.path.join(current_GT_dir, gt_aug_list[j]), rot_gt)
                    tmp_imgs = np.concatenate((rot_x, rot_gt), axis=1)
                cv.imshow('Rotate Data', tmp_imgs)
                cv.waitKey(200)

            print("rotation with {} degrees fullfiled folder: {} ".format(i, folder_name))

    cv.destroyAllWindows()

    print("... rotation done in ", folder_name)

def flip_data(data_dir, augment_both=True):

    X_dir= data_dir[0]
    GT_dir = data_dir[1]
    type_aug = '_flip'
    dir_list = os.listdir(X_dir)
    dir_list.sort()
    if augment_both:
        gt_folders = os.listdir(GT_dir)
        gt_folders.sort()
        if not dir_list ==gt_folders:
            raise NotImplementedError('gt and x folders not match')

    for i in (dir_list):
        X_list = os.listdir(os.path.join(X_dir, i))
        X_list.sort()
        save_dir_x = X_dir + '/' + str(i) + type_aug
        _=make_dirs(save_dir_x)
        n = len(X_list)
        if augment_both:
            GT_list = os.listdir(os.path.join(GT_dir, i))
            GT_list.sort()
            save_dir_gt = GT_dir + '/' + str(i) + type_aug
            _= make_dirs(save_dir_gt)
            n = len(GT_list) if len(X_list) == len(GT_list) else 0
            print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
        else:
            print("Working on the dir: ", os.path.join(X_dir, i))

        for j in range(n):
            x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
            flip_x = np.fliplr(x_tmp)
            cv.imwrite(os.path.join(save_dir_x, X_list[j]), flip_x)
            tmp_imgs =flip_x
            if augment_both:
                gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
                flip_gt = np.fliplr(gt_tmp)
                cv.imwrite(os.path.join(save_dir_gt, GT_list[j]), flip_gt)
                tmp_imgs = np.concatenate((flip_x, flip_gt), axis=1)

            cv.imshow('Flipping data',tmp_imgs)
            cv.waitKey(200)

        print("End flipping file in {}".format(os.path.join(X_dir, i)))

    cv.destroyAllWindows()

    print("... Flipping  data augmentation finished")

def gamma_data(data_dir,augment_both=True, in_gt=False):

    X_dir = data_dir[0]
    GT_dir=data_dir[1]

    gamma30 = '_ga30'
    gamma60 = '_ga60'
    gamma80 = '_ga80'
    dir_list = os.listdir(X_dir)
    dir_list.sort()
    if augment_both:
        gt_folders = os.listdir(GT_dir)
        gt_folders.sort()
        if not dir_list ==gt_folders:
            raise NotImplementedError('gt and x folders not match')
    for i in (dir_list):
        X_list = os.listdir(os.path.join(X_dir, i))
        X_list.sort()
        save_dir_x30 = X_dir + '/' + str(i) + gamma30
        save_dir_x60 = X_dir + '/' + str(i) + gamma60
        save_dir_x80 = X_dir + '/' + str(i) + gamma80
        _ = make_dirs(save_dir_x30)
        _ = make_dirs(save_dir_x60)
        _ = make_dirs(save_dir_x80)
        n =len(X_list)
        if augment_both:
            GT_list = os.listdir(os.path.join(GT_dir, i))
            GT_list.sort()
            save_dir_gt30 = GT_dir + '/' + str(i) + gamma30
            save_dir_gt60 = GT_dir + '/' + str(i) + gamma60
            save_dir_gt80 = GT_dir + '/' + str(i) + gamma80
            _=make_dirs(save_dir_gt30)
            _=make_dirs(save_dir_gt60)
            _=make_dirs(save_dir_gt80)
            n = len(GT_list) if len(X_list) == len(GT_list) else None
            print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
        else:
            print("Working on the dir: ", os.path.join(X_dir, i))
        for j in range(n):
            x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
            if not in_gt:
                x_tmp = image_normalization(x_tmp,0,1)
                x_tmp = gamma_correction(x_tmp, 0.4040, False)
                gam30_x = gamma_correction(x_tmp, 0.3030, True)
                gam60_x = gamma_correction(x_tmp, 0.6060, True)
                gam80_x = gamma_correction(x_tmp, 0.8080, True)
                gam30_x = np.uint8(image_normalization(gam30_x))
                gam60_x = np.uint8(image_normalization(gam60_x))
                gam80_x = np.uint8(image_normalization(gam80_x))
            else:
                gam30_x=x_tmp
                gam60_x = x_tmp
                gam80_x = x_tmp
            if augment_both:
                gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
            cv.imwrite(os.path.join(save_dir_x30, X_list[j]), gam30_x)
            cv.imwrite(os.path.join(save_dir_x60, X_list[j]), gam60_x)
            cv.imwrite(os.path.join(save_dir_x80, X_list[j]), gam80_x)

            tmp_imgs = np.concatenate((gam30_x, gam60_x, gam80_x), axis=1)
            if augment_both:
                cv.imwrite(os.path.join(save_dir_gt30, GT_list[j]), gt_tmp)
                cv.imwrite(os.path.join(save_dir_gt60, GT_list[j]), gt_tmp)
                cv.imwrite(os.path.join(save_dir_gt80, GT_list[j]), gt_tmp)
                tmp_imgs1 = np.concatenate((gam30_x, gt_tmp), axis=1)
                tmp_imgs2 = np.concatenate((gam60_x, gt_tmp), axis=1)
                tmp_imgs3 = np.concatenate((gam80_x, gt_tmp), axis=1)
                tmp_imgs = np.concatenate((tmp_imgs1, tmp_imgs2, tmp_imgs3), axis=0)
            cv.imshow('gramma correction',tmp_imgs)
            cv.waitKey(200)

        print("End gamma correction, file in {}".format(os.path.join(X_dir, i)))

    cv.destroyAllWindows()

    print("... gamma correction  data augmentation finished")

def scale_data(data_dir,augment_both=True):

    X_dir = data_dir[0]
    GT_dir=data_dir[1]

    scl1 = 0.5
    scl2 = 1.5
    scl1t = '_s05'
    scl2t = '_s15'
    dir_list = os.listdir(X_dir)
    dir_list.sort()
    if augment_both:
        gt_list = os.listdir(GT_dir)
        gt_list.sort()
        if not dir_list ==gt_list:
            raise NotImplementedError('gt and x folders not match')
    for i in (dir_list):
        X_list = os.listdir(os.path.join(X_dir, i))
        X_list.sort()
        save_dir_s1 = X_dir + '/' + str(i) + scl1t
        save_dir_s2 = X_dir + '/' + str(i) + scl2t
        _ = make_dirs(save_dir_s1)
        _ = make_dirs(save_dir_s2)
        n =len(X_list)
        if augment_both:
            GT_list = os.listdir(os.path.join(GT_dir, i))
            GT_list.sort()
            save_dir_gts1 = GT_dir + '/' + str(i) + scl1t
            save_dir_gts2 = GT_dir + '/' + str(i) + scl2t
            _=make_dirs(save_dir_gts1)
            _=make_dirs(save_dir_gts2)
            n = len(GT_list) if len(X_list) == len(GT_list) else None
            print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
        else:
            print("Working on the dir: ", os.path.join(X_dir, i))
        for j in range(n):
            x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
            x_tmp1 = scale_img(x_tmp,scl1)
            x_tmp2 = scale_img(x_tmp,scl2)

            if augment_both:
                gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
                gt_tmp1 = scale_img(gt_tmp, scl1)
                gt_tmp2 = scale_img(gt_tmp, scl2)
            cv.imwrite(os.path.join(save_dir_s1, X_list[j]), x_tmp1)
            cv.imwrite(os.path.join(save_dir_s2, X_list[j]), x_tmp2)
            if augment_both:
                cv.imwrite(os.path.join(save_dir_gts1, GT_list[j]), gt_tmp1)
                cv.imwrite(os.path.join(save_dir_gts2, GT_list[j]), gt_tmp2)

            tmp_imgs = np.concatenate((x_tmp1, gt_tmp1), axis=1)
            cv.imshow('scaling image 0.5',tmp_imgs)
            cv.waitKey(300)

        print("Scaling finished, file in {}".format(os.path.join(X_dir, i)))

    cv.destroyAllWindows()

    print("... Scaling augmentation has finished")

#  main tool for dataset augmentation
def augment_data(base_dir,augment_both, dataName,use_all_augs=True):

    print('=========== Data augmentation just for 720x1280 image size ==============')
    augment_gt = True # just for augmenting ne data type (rgb or gt)
    # data_dir = os.path.join(base_dir, 'MULTICUE', 'train')
    data_dir = os.path.join(base_dir, dataName, 'edges')
    # ********** single RGB or NIR not rested just GT  ********
    splitting_up = use_all_augs #use_all_type True to augment by splitting up
    rotation = use_all_augs
    flipping = use_all_augs
    correction_gamma = use_all_augs
    image_scaling = False#use_all_augs

    img_dir = os.path.join(data_dir,'imgs','train','rgbr') # path for image augmentation
    # img_dir = os.path.join(data_dir,'imgs') # path for image augmentation
    # gt_dir = os.path.join(data_dir, 'gt') # set this even if the augment is just GT
    gt_dir = os.path.join(data_dir, 'edge_maps', 'train', 'rgbr') # set this even if the augment is just GT

    if not augment_both and augment_gt:
        img_dir = gt_dir
        gt_dir = None
        print('Augmenting  just GTs')
    elif not augment_both and  not augment_gt:
        gt_dir=None
        print('Augmenting  just for RGB imag')
    else:
        print('Augmenting RGB image and the GT')
        # return

    dataset_dirs = [img_dir, gt_dir]
    # *********** starting data augmentation *********
    if splitting_up:
        print("Image augmentation by splitting up have started!")
        dataset_dirs = split_data(data_dir=dataset_dirs,augment_both=augment_both)
        splitting_up =False

    if not splitting_up:
        img_aug_dir = os.path.join(img_dir,'aug')
        gt_aug_dir = os.path.join(gt_dir,'aug') if augment_both else None
        dataset_dirs = [img_aug_dir,gt_aug_dir]

    if rotation:
        print("Image augmentation by rotation have started!")
        rotate_data(data_dir=dataset_dirs,augment_both=augment_both)

    if flipping:
        print("Image augmentation by flipping have started!")
        flip_data(data_dir=dataset_dirs,augment_both=augment_both)

    if correction_gamma:
        print("Image augmentation by gamma correction have started!")
        gamma_data(data_dir=dataset_dirs, augment_both=augment_both, in_gt=augment_gt)

    if image_scaling:
        print("Data augmentation by image scaling has started!")
        scale_data(data_dir=dataset_dirs, augment_both=augment_both)