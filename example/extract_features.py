__author__ = 'minhaz palasara'

import sys
from melanoma.feature_extraction import feature_extractor

sys.setrecursionlimit(50000)

# Batch size should be same as in the training
batch_size = 100
data_dir = '/home/roboticslab/suborders/suborder-6/'
is3D = False

# feature indexes ( start from 0)
feature_ids = [10, 13, 15]

# list of directories to store the results
result_dir = ['/home/roboticslab/dense1/', '/home/roboticslab/dense2/', '/home/roboticslab/dense3/']

# Trained Network file
model_file_path="best_model_74.pkl"

feature_extractor(batch_size, data_dir, feature_ids, result_dir, model_file_path, is3D)
