#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import csv
import numpy as np
import codecs

def crop_image(xmin, ymin, height, width, label, childpath,
               dest, index, destlabelpath):
    images = cv2.imread(childpath)
    # cv2.imshow('raw image', images)
    crop_img = images[ymin:ymin + height, xmin:xmin + width]
    # cv2.imshow('crop_img', crop_img)
    cv2.imwrite(
        '/home/rechardchen123/Documents/github/'
        'crop_image_and_generate_label/crop_images/%s-%d.jpg' % (dest, index), crop_img)
    file = open(destlabelpath+'label.txt', 'a')
    file.write(dest + '-' + str(index) + ' ' + label + '\n')
    file.close()

imagepath = '/home/rechardchen123/Documents/github/crop_image_and_generate_label/train_images_test/'
imagelabelpath = '/home/rechardchen123/Documents/github/crop_image_and_generate_label/image_label/'
destpath = '/home/rechardchen123/Documents/github/crop_image_and_generate_label/crop_images/'
destlabelpath = '/home/rechardchen123/Documents/github/crop_image_and_generate_label/crop_labels/'
pathDir = os.listdir(imagepath)
for file in pathDir:
    child_path = os.path.join(imagepath, file)
    dest = os.path.splitext(file)[0]
    txt_image = file.replace(os.path.basename(file).split('.')[1], 'txt')
    txt_path = os.path.join(imagelabelpath, txt_image)
    with open(txt_path, 'r') as f:
        reader = csv.reader(f)
        for index, line in enumerate(reader):
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            oriented_bbox = [int(line[i]) for i in range(8)]
            oriented_bbox = np.asarray(oriented_bbox)
            xs = oriented_bbox.reshape(4, 2)[:, 0]
            ys = oriented_bbox.reshape(4, 2)[:, 1]
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            height = ymax - ymin
            width = xmax - xmin
            label = line[-1]
            crop_image(xmin, ymin, height, width, label, child_path, dest, index, destlabelpath)
