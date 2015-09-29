# -*- coding: utf-8 -*-
from __future__ import absolute_import

import theano
import os
import theano.tensor as T
from theano.tensor.signal import downsample
from .. import activations, initializations
from ..utils.theano_utils import shared_zeros
from ..layers.core import Layer

from theano.tensor.nnet.conv3d2d import *

# class Convolution1D(Layer): TODO

# class MaxPooling1D(Layer): TODO

class Convolution2D(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col, 
        init='glorot_uniform', activation='linear', weights=None,
        image_shape=None, border_mode='valid', subsample=(1,1)):
        super(Convolution2D,self).__init__()

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.image_shape = image_shape
        self.nb_filter = nb_filter
        self.stack_size = stack_size
        self.nb_row = nb_row
        self.nb_col = nb_col

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((nb_filter,))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)

        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W,
            border_mode=self.border_mode, subsample=self.subsample, image_shape=self.image_shape)
        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "nb_filter":self.nb_filter,
            "stack_size":self.stack_size,
            "nb_row":self.nb_row,
            "nb_col":self.nb_col,
            "init":self.init.__name__,
            "activation":self.activation.__name__,
            "image_shape":self.image_shape,
            "border_mode":self.border_mode,
            "subsample":self.subsample}

class Convolution3D(Layer):

    def __init__(self, nb_filter, stack_size, nb_row, nb_col, nb_depth,
                 init='uniform', activation='linear', weights=None,
                 image_shape=None, border_mode='valid'):

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.image_shape = image_shape
        dtensor5 = T.TensorType('float32', (0,)*5)
        self.input = dtensor5()

        if theano.config.device == 'gpu':
            self.W_shape = (nb_filter, nb_depth, stack_size, nb_row, nb_col)
        else: #cpu
            self.W_shape = (nb_filter, nb_row, nb_col, nb_depth, stack_size)

        self.W = self.init(self.W_shape)
        self.b = shared_zeros((nb_filter,))
        self.params = [self.W, self.b]


        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):

        X = self.get_input(train)

        if theano.config.device == 'gpu':
            conv_out = theano.tensor.nnet.conv3d2d.conv3d(signals=X,
                                                         filters=self.W,
                                                         signals_shape=self.image_shape,
                                                         filters_shape=self.W_shape,
                                                         border_mode=self.border_mode)
            output = self.activation(conv_out + self.b.dimshuffle('x', 'x',  0, 'x', 'x'))
        else:
            conv_out = theano.tensor.nnet.conv3D(V=X, W=self.W, b=self.b, d=(1,1,1))
            output = self.activation(conv_out + self.b.dimshuffle('x', 'x', 'x', 'x', 0))

        return output

class MaxPooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        super(MaxPooling2D,self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, self.poolsize, ignore_border=self.ignore_border)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize,
            "ignore_border":self.ignore_border}

class MaxPooling3D(Layer):

    def __init__(self, poolsize=(2, 2, 2), ignore_border=True):

        self.pool_size = poolsize
        self.ignore_border = ignore_border

        dtensor5 = T.TensorType('float32', (0,)*5)
        self.input = dtensor5()

        self.params = []

    def get_output(self, train):
        X = self.get_input(train)


        if theano.config.device == 'gpu':
            # max_pool_2d X and Z
            output = downsample.max_pool_2d(input=X.dimshuffle(0, 4, 2, 3, 1),
                                            ds=(self.pool_size[1], self.pool_size[2]),
                                            ignore_border=self.ignore_border)

            # max_pool_2d X and Y (with X constant)
            output = downsample.max_pool_2d(input=output.dimshuffle(0, 4, 2, 3, 1),
                                            ds=(1, self.pool_size[0]),
                                            ignore_border=self.ignore_border)
        else: #cpu  order:(batch, row, column, time, inchannel) from cpu convolution
             # max_pool_2d X and Z
            output = downsample.max_pool_2d(input=X.dimshuffle(0, 4, 1, 2, 3),
                                            ds=(self.pool_size[1], self.pool_size[2]),
                                            ignore_border=self.ignore_border)

            # max_pool_2d X and Y (with X constant)
            output = downsample.max_pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                            ds=(1, self.pool_size[0]),
                                            ignore_border=self.ignore_border)
            output = output.dimshuffle(0, 4, 3, 2, 1)

        return output

# class ZeroPadding2D(Layer): TODO
        
