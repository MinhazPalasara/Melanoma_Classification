__author__ = 'minhaz palasara'

import re
import os
import numpy as np
import random as rand

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

    # Normalizing each row
    row_sums = features.sum(axis=1)
    features = features / row_sums[:, np.newaxis]

    return features, labels








