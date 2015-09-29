
import h5py
import random
import numpy as np
import os


# Author: Jake Varley, Minhaz Palasara
# Class for melanoma dataset generation.
class MelanomaDataset3D():

    def __init__(self, data_dir, examples=None):
        if examples:
            self.examples = examples
        else:
            self.examples = [data_dir +"/"+filename for filename in os.listdir(data_dir) if ".h5" in filename]

        datafile = h5py.File(examples[0], 'r')
        data = datafile['data'][0]
        data_dimension = data.shape

        #Sample Dimension(layers, Width, Height, Depth)
        self.sample_dimensions = [data_dimension[0], data_dimension[1], data_dimension[2]]

    # return number of samples
    def get_num_examples(self):
        return len(self.examples)

    def iterator(self, mode=None, batch_size=None, num_batches=None):
            if mode=="sequential":
                return MelanomaSequentialIterator3D(self, batch_size=batch_size)

            elif mode=="even_shuffled_sequential":
                return MelanomaRandomIterator3D(self,
                                              batch_size=batch_size,
                                              num_batches=num_batches)
            else:
                raise "No such mode present"


# Author: Jake Varley
# Iterator used for random extraction of the samples. It is mainly used while training the Net
class MelanomaRandomIterator3D():

    def __init__(self, dataset,
                 batch_size,
                 num_batches):

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.dataset = dataset
        dataset_size = dataset.get_num_examples()

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        #for 3D
        batch_x = np.zeros(([self.batch_size]+self.dataset.sample_dimensions+[1]))

        batch_y = np.zeros((self.batch_size, 2))

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            example_filepath = self.dataset.examples[index]

            dset = h5py.File(example_filepath, 'r')

            batch_x[i, :, :, :, 0] = dset['data'][0]

            if dset['label'][0] == -1:
                batch_y[i, 0] = 1
                batch_y[i, 1] = 0
            else:
                batch_y[i, 0] = 0
                batch_y[i, 1] = 1

            dset.close()

        # Organizing the data based on the theano documentation
        #(http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d)
        if theano.config.device == 'gpu':
            batch_x = batch_x.transpose(0, 4, 1, 2, 3) # as required by conv2d3d.Conv3d
        else: #'cpu'
            batch_x = batch_x.transpose(0, 2, 3, 1, 4) # as required by conv3D

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()


# Author: Minhaz Palasara
# Iterator used for sequential extraction of the samples in batches.
# When the number of remaining samples is less than the batch size, it appends samples with zeros to complete the batch
# Return: XBatch, YBatch and the Batch_File_Names(will be blank for the appended samples)
class MelanomaSequentialIterator3D():

    def __init__(self, dataset,
                 batch_size):

        self.batch_size = batch_size
        self.num_batches = len(dataset.examples)/ batch_size
        self.dataset = dataset
        self.files_returned = 0

        # Number of batches and partial batch size
        self.partial_batch_size = dataset.get_num_examples() % batch_size
        self.is_partial_batch = False

        if self.partial_batch_size  > 0:
           self.is_partial_batch = True

    def __iter__(self):
        return self

    def next(self):

        if not self.has_next():
               raise "No more files to return"

        batch_indices = np.array(range(self.batch_size))

        # Creating the object to store the information. Pre-allocation is also beneficial for partial batch.
        batch_x = np.zeros(( [self.batch_size] + self.dataset.sample_dimensions+[1]))
        batch_y = np.zeros((self.batch_size, 2))
        batch_files = [""] * self.batch_size

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            example_filepath = self.dataset.examples[(self.files_returned+ index)%self.num_examples()]
            batch_files[i] = example_filepath

            dset = h5py.File(example_filepath, 'r')

            batch_x[i, :, :, :, 0] = dset['data'][0]
            if dset['label'][0] == -1:
                batch_y[i, 0] = 1
                batch_y[i, 1] = 0
            else:
                batch_y[i, 0] = 0
                batch_y[i, 1] = 1

            dset.close()

        #(http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d)
        #0 ; batch_size
        #1 ; stack size, number of channels(z) in 3D data
        #2 ; image row size
        #3 ; image column size
        #4 ; 4th dimension, set to  1 for one channel in 3D data
        if theano.config.device == 'gpu':
            batch_x = batch_x.transpose(0, 4, 1, 2, 3) # as required by conv2d3d.Conv3d
        else: #'cpu'
            batch_x = batch_x.transpose(0, 2, 3, 1, 4) # as required by conv3D

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        # increment number of processed files
        self.files_returned = self.files_returned + self.batch_size
        return batch_x, batch_y, batch_files

    def has_next(self):
        if self.files_returned < self.num_examples() :
            return True
        else:
            return False

    def is_partial(self):
        # Has some file left?, partial batch flag is true? and all the batches are done?
        if self.has_next() and self.files_returned == self.num_batches * self.batch_size and self.is_partial_batch:
            return True
        else:
            return False

    def partial_batch_size(self):
        return self.partial_batch_size

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()


# Author: Jake Varley, Minhaz Palasara
# Class for melanoma dataset creation. Data dimension extraction is automated
class MelanomaDataset2D():

    def __init__(self, data_dir, examples=None):
        if examples:
            self.examples = examples
        else:
            self.examples = [data_dir +"/"+filename for filename in os.listdir(data_dir) if ".h5" in filename]

        datafile = h5py.File(examples[0], 'r')
        data = datafile['data']
        data_dimension = data.shape

        # Sample Dimension(layers, Width, Height, Depth)
        self.sample_dimensions = [data_dimension[1], data_dimension[2], data_dimension[3]]

    # return number of samples
    def get_num_examples(self):
        return len(self.examples)

    def iterator(self, mode=None, batch_size=None, num_batches=None):
            if mode=="sequential":
                return MelanomaSequentialIterator2D(self, batch_size=batch_size)

            elif mode=="even_shuffled_sequential":
                return MelanomaRandomIterator2D(self,
                                              batch_size=batch_size,
                                              num_batches=num_batches)
            else:
                raise "No such mode present"


# Author: Jake Varley
# Iterator used for random extraction of the samples. It is mainly used while training the Net
class MelanomaRandomIterator2D():

    def __init__(self, dataset,
                 batch_size,
                 num_batches):

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.dataset = dataset
        dataset_size = dataset.get_num_examples()

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        #for 2D
        batch_x = np.zeros(( [self.batch_size] + self.dataset.sample_dimensions))
        batch_y = np.zeros((self.batch_size, 2))

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            example_filepath = self.dataset.examples[index]

            dset = h5py.File(example_filepath, 'r')

            batch_x[i, :, :, :] = dset['data'][0]

            if dset['label'][0] == -1: #or write dset['label'][0, 0] == -1
                batch_y[i, 0] = 1
                batch_y[i, 1] = 0
            else:
                batch_y[i, 0] = 0
                batch_y[i, 1] = 1

            dset.close()

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()


# Author: Minhaz Palasara
# Iterator used for sequential extraction of the samples in batches.
# When the number of remaining samples is less than the batch size, it appends samples with zeros to complete the batch
# Return: XBatch, YBatch and the Batch_File_Names(will be blank for the appended samples)
class MelanomaSequentialIterator2D():

    def __init__(self, dataset,
                 batch_size):

        self.batch_size = batch_size
        self.num_batches = len(dataset.examples)/ batch_size
        self.dataset = dataset
        self.files_returned = 0

        # Number of batches and partial batch size
        self.partial_batch_size = dataset.get_num_examples() % batch_size
        self.is_partial_batch = False

        if self.partial_batch_size  > 0:
           self.is_partial_batch = True

    def __iter__(self):
        return self

    def next(self):

        if not self.has_next():
               raise "No more files to return"

        batch_indices = np.array(range(self.batch_size))

        # Creating the object to store the information. Pre-allocation is also beneficial for partial batch.
        batch_x = np.zeros(( [self.batch_size] + self.dataset.sample_dimensions))
        batch_y = np.zeros((self.batch_size, 2))
        batch_files = [""] * self.batch_size

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            example_filepath = self.dataset.examples[(self.files_returned + index) % self.num_examples()]
            batch_files[i] = example_filepath

            dset = h5py.File(example_filepath, 'r')

            batch_x[i, :, :, :] = dset['data'][0]
            if dset['label'][0] == -1:
                batch_y[i, 0] = 1
                batch_y[i, 1] = 0
            else:
                batch_y[i, 0] = 0
                batch_y[i, 1] = 1

            dset.close()

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        # increment number of processed files
        self.files_returned = self.files_returned + self.batch_size
        return batch_x, batch_y, batch_files

    def has_next(self):
        if self.files_returned < self.num_examples() :
            return True
        else:
            return False

    def is_partial(self):
        # Has some file left?, partial batch flag is true? and all the batches are done?
        if self.has_next() and self.files_returned == self.num_batches * self.batch_size and self.is_partial_batch:
            return True
        else:
            return False

    def partial_batch_size(self):
        return self.partial_batch_size

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()