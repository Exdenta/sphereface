#!/usr/bin/env python3
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['GLOG_minloglevel'] = '2' # suppress Caffe verbose prints

import sys
import cv2
import math
import time
import caffe
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import spatial
from datetime import datetime

def get_embeddings(net, image):
    image = (image - 127.5) / 128
    image = np.asarray(image)
    image = np.transpose(image, (2,0,1))
    image = image[None, :]
    net.blobs['data'].data[...] = image
    out = net.forward()
    return out['fc5'][0]


def getAccuracy(pairs, threshold):
    TP = sum(map(lambda x: x['score']>threshold and x['flag']==1, pairs))
    TN = sum(map(lambda x: x['score']<threshold and x['flag']!=1, pairs))
    return (TP + TN) / len(pairs)


if __name__ == "__main__":
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    project_path = current_dir.parent.parent
    dataset_path = os.path.join(project_path, 'test/data')
    list_ = os.path.join(dataset_path, 'pairs.txt')
    folder_ = os.path.join(dataset_path, 'lfw-112X96')

    # init model
    caffe.set_mode_cpu()
    model   = os.path.join(project_path, 'train/code/sphereface_deploy.prototxt')
    weights = os.path.join(project_path, 'train/result/sphereface_model.caffemodel')
    net = caffe.Net(model, weights, caffe.TEST)

    # get features
    pairs = []
    with open(list_, 'r') as f:
        lines = f.readlines()
    
    print("Forming features:")
    for i, line in enumerate(tqdm(lines)):
        if i < 1:
            continue
        pair = {}
        l = line.split()
        if len(l) == 3:
            pair['fileL'] = os.path.join(folder_, l[0], l[0] + "_" + l[1].rjust(4, '0') + ".jpg")
            pair['fileR'] = os.path.join(folder_, l[0], l[0] + "_" + l[2].rjust(4, '0') + ".jpg")
            pair['flag']  = 1
        elif len(l) == 4:
            pair['fileL'] = os.path.join(folder_, l[0], l[0] + "_" + l[1].rjust(4, '0') + ".jpg")
            pair['fileR'] = os.path.join(folder_, l[2], l[2] + "_" + l[3].rjust(4, '0') + ".jpg")
            pair['flag']  = -1
        pair['fold']  = math.ceil(i / 600)
        # featureL
        image = cv2.imread(pair['fileL'])
        embsL = get_embeddings(net, image)
        pair['featureL'] = embsL.copy()
        # featureR
        image   = cv2.imread(pair['fileR'])
        embsR   = get_embeddings(net, image)
        pair['featureR'] = embsR.copy()
        # cosine distance
        pair['score'] = 1 - spatial.distance.cosine(pair['featureL'], pair['featureR']) 
        pairs.append(pair)

    # save for future testing
    np.save("test/code/pairs_{}.npy".format(datetime.now().strftime("%Y%m%d%H%M%S")), pairs)

    # get accuracy
    print("Accuracy:")
    ACCs = np.zeros(10)
    Thrs = np.zeros(10)
    for j in np.arange(0, 10):
        validation_pairs = list(filter(lambda x: x['fold']!=j+1, pairs))
        test_pairs = list(filter(lambda x: x['fold']==j+1, pairs))

        thrNum = 1000
        accuracys  = np.zeros(2*thrNum)
        thresholds = np.arange(-thrNum, thrNum) / thrNum
        for i in np.arange(0, 2*thrNum):
            accuracys[i] = getAccuracy(validation_pairs, thresholds[i])
        
        threshold = np.mean(thresholds[np.where(accuracys == max(accuracys))])
        ACCs[j] = getAccuracy(test_pairs, threshold)
        Thrs[j] = threshold
        print("accuracy: ", ACCs[j] * 100, ", threshold: ", threshold)

    print('----------------')
    print('AVE: ', np.mean(ACCs)*100)
    print('Threshold: ', np.mean(Thrs))
