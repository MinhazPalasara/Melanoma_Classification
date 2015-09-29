__author__ = 'Jake Varley'

import numpy as np
import math
import theano


def load_data(test_split=0.2, dataset_size=5000, patch_size=32):
        if patch_size < 10:
            raise NotImplementedError

        num_labels = 2

        # TODO: allow users to specify how they want the classes to be distributed.
        # Currently using same probability for each class
        geometry_types = np.random.randint(0, num_labels, dataset_size)

        # Getting the training set
        y_train = geometry_types[0:abs((1-test_split)*dataset_size)]
        x_train = __generate_solid_figures(geometry_types=y_train, patch_size=patch_size)

        # Getting the testing set
        y_test = geometry_types[abs((1-test_split)*dataset_size):]
        x_test = __generate_solid_figures(geometry_types=y_test, patch_size=patch_size)

        return (x_train, y_train),(x_test, y_test)


def __generate_solid_figures(geometry_types, patch_size):

        shapes_no = geometry_types.shape[0]

        # Assuming data is centered
        (x0, y0, z0) = ((patch_size-1)/2,)*3

        # Allocate 3D data array, data is in cube(all dimensions are same)
        # Order of data_holder, (batch, Z, X, Y, 1)
        solid_figures = np.zeros((len(geometry_types), patch_size,
                                  patch_size, patch_size, 1), dtype=np.bool)

        for i in xrange(shapes_no):
            # radius is a random number in [3, self.patch_size/2)
            radius = (patch_size/2 - 3) * np.random.rand() + 3

            # bounding box values for optimization
            x_min = int(max(math.ceil(x0-radius), 0))
            y_min = int(max(math.ceil(y0-radius), 0))
            z_min = int(max(math.ceil(z0-radius), 0))
            x_max = int(min(math.floor(x0+radius), patch_size-1))
            y_max = int(min(math.floor(y0+radius), patch_size-1))
            z_max = int(min(math.floor(z0+radius), patch_size-1))

            if geometry_types[i] == 0: # Sphere
                # We only iterate through the bounding box of the sphere to check whether voxels are inside the sphere
                radius_squared = radius**2
                for z in xrange(z_min, z_max+1):
                    for x in xrange(x_min, x_max+1):
                        for y in xrange(y_min, y_max+1):
                            if (x-x0)**2 + (y-y0)**2 + (z-z0)**2 <= radius_squared:
                                # inside the sphere
                                solid_figures[i, z, x, y, 0] = 1
            elif geometry_types[i] == 1: # Cube
                solid_figures[i, z_min:z_max+1, x_min:x_max+1, y_min:y_max+1, 0] = 1
            else:
                raise NotImplementedError

        #(http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d)
        #0 ; batch_size
        #1 ; stack size, number of channels(z) in 3D data
        #2 ; image row size
        #3 ; image column size
        #4 ; 4th dimension, set to  1 for one channel in 3D data
        if theano.config.device == 'gpu':
            solid_figures = solid_figures.transpose(0, 1, 4, 2, 3) # as required by conv2d3d.Conv3d
        else: #'cpu'
            solid_figures = solid_figures.transpose(0, 2, 3, 1, 4) # as required by conv3D

        return solid_figures
