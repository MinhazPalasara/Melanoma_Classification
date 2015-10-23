__author__ = 'minhaz palasara'

import re
import os
import numpy as np
import random as rand
from sklearn import preprocessing
import yaml
import cv2

# label -1 and 1
def createLableMap(label_file):
    labelFile = open(label_file, 'r')
    labelMap = {}
    for line in labelFile:
        line=re.split(r'\t+',line.lstrip().rstrip())
        if line[2] == 'Negative':
            labelMap[line[0]] = -1
        else:
            labelMap[line[0]] = 1
    labelFile.close()

    return labelMap

def load_features(feature_path, label_map):

    files = os.listdir(feature_path)

    rand.shuffle(files)

    f = open(feature_path+"/"+files[0], 'r')

    feature_vector_dims = sum([1 for line in f])

    features = np.zeros((len(files), feature_vector_dims))
    labels = np.zeros((len(files), 1))
    # labels = []
    f.close()
    i = 0

    for file in files:
          # print str(i)
          out_file_name = '.'.join(file.split('.')[0:-1])
          f = open(feature_path+"/"+file, 'r')

          j = 0
          for line in f:
            # print str(j)
            line = line.lstrip().rstrip()
            line = map(float, [line])
            features[i, j] = line[0]
            j += 1

          labels[i, 0] = label_map.get(out_file_name)
          f.close()
          i += 1

    # features = np.transpose(features)
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
    # min_max_scaler.fit_transform(features)
    # features = np.transpose(features)

    return features, labels


# Loading feature text files in a 2D array
def load_combined_features(feature_path1, feature_path2, label_map):

    files = os.listdir(feature_path1)
    rand.shuffle(files)

    f = open(feature_path1+"/"+files[0], 'r')
    feature_vector_dims1 = sum([1 for line in f])
    f.close()

    f = open(feature_path2+"/"+files[0], 'r')
    feature_vector_dims2 = sum([1 for line in f])
    f.close()

    features = np.zeros((len(files), feature_vector_dims1+feature_vector_dims2))
    features1 = np.zeros((len(files), feature_vector_dims1))
    features2 = np.zeros((len(files), feature_vector_dims2))
    labels = np.zeros((len(files), 1))

    i = 0

    for file in files:
        file_name = '.'.join(file.split('.')[0:-1])

        # First set of features
        f = open(feature_path1+"/"+file, 'r')

        j = 0
        for line in f:
            line = line.lstrip().rstrip()
            line = map(float, [line])
            features1[i, j] = line[0]
            j += 1
        f.close()

        # Second set of features
        j = 0
        f = open(feature_path2+"/"+file, 'r')
        for line in f:
           line = line.lstrip().rstrip()
           line = map(float, [line])
           features2[i, j] = line[0]
           j += 1

        f.close()
        labels[i, 0] = label_map.get(file_name)

        i += 1

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)

    # min_max_scaler.fit_transform(features1)
    # min_max_scaler.fit_transform(features2)

    # Returning the combined features
    features[:, 0:feature_vector_dims1] = features1
    features[:, feature_vector_dims1:] = features2

    return features, labels


def load_yml_features(feature_path, label_map):
    files = os.listdir(feature_path)
    rand.shuffle(files)

    f = np.asarray(cv2.cv.Load(feature_path+"/"+files[0]))
    features = np.zeros((len(files), f.shape[1]))
    labels = np.zeros((len(files), 1))

    i = 0

    for file in files:
        out_file_name = '.'.join(file.split('.')[0:-1])
        f = np.asarray(cv2.cv.Load(feature_path+"/"+file))
        features[i, :] = f[:]
        labels[i, 0] = label_map.get(out_file_name)

        i += 1

    # features = np.transpose(features)
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
    # min_max_scaler.fit_transform(features)
    # features = np.transpose(features)

    return features, labels






