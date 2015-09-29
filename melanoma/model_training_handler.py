__author__ = 'Minhaz Palasara'

import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten3D, Flatten2D
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from melanoma.melanoma_dataset  import MelanomaDataset3D, MelanomaDataset2D
import random
import os
import time
import cPickle
import numpy as np

# @Author: Jake Varley, Minhaz Palasara
class MelanomaModel:

    def __init__(self, nb_train_batches, batch_size, is3D):

        self.model = None
        self.train_data_set=None
        self.test_data_set=None
        self.valid_data_set=None
        self.is3D = is3D;

        # compute the number of mini-batches for training, validation and testing
        self.nb_train_batches=nb_train_batches
        self.batch_size=batch_size

    def create_model(self):
        self.model = Sequential()
        self.model.add(Convolution3D(16, stack_size=1, nb_row=11, nb_col=11, nb_depth=6, border_mode='valid'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling3D(poolsize=(3, 3, 1)))
        self.model.add(Convolution3D(32, stack_size=16, nb_row=5, nb_col=5, nb_depth=1, border_mode='valid' ))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling3D(poolsize=(3, 3, 1)))
        self.model.add(Convolution3D(64, stack_size=32, nb_row=3, nb_col=3, nb_depth=1, border_mode='valid' ))
        self.model.add(MaxPooling3D(poolsize=(3, 3, 1)))
        self.model.add(Flatten3D())
        self.model.add(Dense(4096, 1024, init='normal'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, 512, init='normal'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512, 2, init='normal'))

        # let's train the self.model using SGD + momentum(how original).
        sgd = RMSprop(rho=0.9, epsilon=1e-3, lr=0.001)
        self.model.compile(loss='mean_squared_error', optimizer=sgd)

    def load_model(self, model_file_path):
        model_file = open(model_file_path)
        self.model = cPickle.load(model_file)

    def load_melanoma_dataset(self, data_dir, training_perc):

        # Preparing melanoma dataset
        # data directory folder path
        file_names = ([data_dir + filename for filename in os.listdir(data_dir) if ".h5" in filename])
        random.shuffle(file_names)

        train_file_names = file_names[0:int(training_perc*len(file_names))]
        test_file_names = file_names[int(training_perc*len(file_names)):]

        if self.is3D:
            self.train_data_set = MelanomaDataset3D(data_dir, examples=train_file_names)
            self.test_data_set = MelanomaDataset3D(data_dir, examples=test_file_names)
        else:
            self.train_data_set = MelanomaDataset2D(data_dir, examples=train_file_names)
            self.test_data_set = MelanomaDataset2D(data_dir, examples=test_file_names)

    # storing and printing average error over all the mini-batches in an apoch
    def train_model(self, nb_epoch, model_starting_id, model_snapshot_freq, stat_snapshot_freq):
        losses = []
        errors = []

        last_error = float("inf")

        for e in range(nb_epoch):

            print " Performing Epoch no : " + str(e)

            train_iterator = self.train_data_set.iterator(batch_size=self.batch_size,
                                                          num_batches=self.nb_train_batches,
                                                          mode='even_shuffled_sequential')

            for b in range(self.nb_train_batches):
                X_batch, Y_batch = train_iterator.next()
                loss = self.model.train(X_batch, Y_batch)
                print "loss: "+ str(loss)
                losses.append(loss)

            test_iterator = self.test_data_set.iterator(batch_size=self.batch_size,
                                                        mode='sequential')


            errors1 = []
            while test_iterator.has_next():
                X_batch, Y_batch, bacth_files = test_iterator.next()
                error = self.model.test(X_batch, Y_batch)
                errors1.append(error)

            mean_error = np.mean(errors1)
            errors.append(mean_error)
            print "error:   "+ str(mean_error)

            if mean_error < last_error:
                last_error = mean_error
                pickle.dump(self.model, open("best_model_"+str(e)+".pkl","wc"))


            if(e % stat_snapshot_freq == 0):
                pickle.dump(losses, open("loss.pkl", "wc"))
                pickle.dump(errors, open("error.pkl", "wc"))

            if(e % model_snapshot_freq == 0):
                pickle.dump(self.model, open("trained_model.pkl","wc"))
                model_starting_id += 1