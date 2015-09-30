__author__ = 'Minhaz Palasara'

import os
import h5py
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte

# storing a .h5 file an images(.h5 file can have single or multiple layers)
class ExtractImage:

    def __init__(self):
        self.image_extension = '.jpg'

    def extractImage(self, data_path, output_path):

        files = os.listdir(data_path)

        for file in files:
            print file
            data_file = h5py.File(data_path+"/"+file,'r')
            data = data_file['data']

            for i in range(0, data.shape[1]):
                layer = data[0, i, :]
                io.imsave(output_path+"/"+'.'.join(file.split('.')[0:-1])+"-"+str(i)+self.image_extension,
                          img_as_ubyte(layer/255)) # printing in gray scale

            data_file.close()


# Extracting the suborders from complete dataset.
# I have processed complete sets and stored it, Either you could create new suborders from complete data and store it, OR
# The other option is to not use this class but use required data while unpacking an h5 file in the Keras
class ExtractSuborder:

    def extract_suborder(self, data_path, output_path, suborder_list):

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        files = os.listdir(data_path)
        print len(files)

        for file in files:
            data_file = h5py.File(data_path+"/"+file,'r')
            data = data_file['data']
            layer = data[0, :, :, :]

            output_data = np.zeros((1, len(suborder_list), layer.shape[1], layer.shape[2]))

            i = 0
            for suborder in suborder_list:
              output_data[0, i, :, :] = layer[suborder, :, :]
              i += 1

            output_data_file = h5py.File(output_path+"/"+file, 'w')
            output_data_file.create_dataset(name='data', data=output_data, compression="gzip")
            output_data_file.create_dataset(name='label', data=data_file['label'], compression="gzip")

            output_data_file.close()
            data_file.close()
