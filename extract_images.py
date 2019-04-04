import os
import cv2
import csv
import numpy as np


# check if image directory is created, create it if not
def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# function to find the boundaries of non-face image patches
def non_boundary(im_min, im_max, crop, dim):
    if im_min > crop:
        non_max = crop
        non_min = 0
    elif dim - im_max > crop:
        non_max = dim
        non_min = dim - crop
    else:
        non_min = 0
        non_max = 0
    return non_min, non_max


check_dir(os.path.join(os.getcwd(), 'Extracted_Faces'))
check_dir(os.path.join(os.getcwd(), 'Extracted_Non'))
annotation_file = os.path.join(os.getcwd(), 'list_bbox_celeba.csv')
im_crop_size = (60, 60)

with open(annotation_file, 'r') as csvfile:
    annotation_list = list(csv.reader(csvfile, delimiter=','))
    idx = 0
    for annotations in annotation_list[1:]:
        # annotations = line.split(',')
        img_file = os.path.join(os.getcwd(), os.path.join('Faces', annotations[0]))
        x_min = int(float(annotations[1]))
        y_min = int(float(annotations[2]))
        x_max = int(float(annotations[3])) + x_min
        y_max = int(float(annotations[4].rstrip('\n'))) + y_min
        img = cv2.imread(img_file)
        h = img.shape[0]
        w = img.shape[1]
        # extract portion of image within bounding box annotation (with face)
        img_face = img #[y_min:y_max, x_min:x_max]
        # check for empty image
        if img_face.size == 0:
            continue
        # resize image
        img_face_scale = cv2.resize(img_face, im_crop_size, interpolation=cv2.INTER_AREA)
        img_out_file = 'face' + str(idx) + '.jpg'
        img_path_face = os.path.join('Extracted_Faces', img_out_file)
        cv2.imwrite(img_path_face, img_face_scale)

        # extract non-face images
        # find boundaries for non-face images (background outside of bounding box)
        # [xn_min, xn_max] = non_boundary(x_min,x_max,im_crop_size[0], w)
        # if xn_min == 0 and xn_max == 0:
        #     [xn_min, xn_max] = non_boundary(x_min, x_max, im_crop_size[0]//2, w)
        # [yn_min, yn_max] = non_boundary(y_min, y_max, im_crop_size[1]//2, h)
        # if yn_min == 0 and yn_max == 0:
        #     [yn_min, yn_max] = non_boundary(y_min, y_max, im_crop_size[1]//2, h)
        img_non = img[0:29, 0:29]
        img_non = cv2.resize(img_non, im_crop_size, interpolation=cv2.INTER_AREA)
        img_out_file = 'non' + str(idx) + '.jpg'
        img_path_non = os.path.join('Extracted_Non', img_out_file)
        cv2.imwrite(img_path_non, img_non)
        idx += 1
        if idx > 12001:
            break


