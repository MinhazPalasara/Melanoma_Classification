__author__ = 'Minhaz Palasara'

import sys
import pickle
from melanoma.model_training_handler import MelanomaModel
import os
from keras.datasets import shapes_3d

sys.setrecursionlimit(50000)

#Training parameters
nb_train_batches = 50
nb_test_batches = 100
nb_classes = 2
nb_epoch = 2000
batch_size = 10
data_dir = '/home/minhazpalasara/suborders/donwnsampled_suborder-5-&-6-&-8/suborder-6/'
training_perc = 0.8

is3D = True


# # Training Models
# handler=MelanomaModel(nb_train_batches, nb_test_batches, batch_size, is3D)
# handler.create_model()
#
# # For resuming the training
# # handler.load_model('trained_model_11.pkl')
#
# handler.load_melanoma_dataset(data_dir, training_perc)
# handler.train_model(nb_epoch, 1, 10, 10)


