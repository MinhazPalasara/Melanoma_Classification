__author__ = 'Minhaz Palasara'

import h5py
import os
import numpy as np

# Extract the metadata related to the data files,
# As of now, only rows and cols are used for the purpose
class DataMetadata:
    def __init__(self):
        self.rows = 0
        self.row_file =''
        self.col_file =''
        self.cols = 0

    def loadSize(self, data_path):

        files = os.listdir(data_path)

        for file in files:
            data_file = h5py.File(data_path+"/"+file,'r')
            data = data_file['data']
            layer = data[0,:,:,:]

            if self.rows < layer.shape[1]:
                self.rows = layer.shape[1]
                self.row_file = file

            if self.cols < layer.shape[2]:
                self.cols = layer.shape[2]
                self.col_file = file

            data_file.close()

    def getMetadata(self):

        print "Max Rows: "+str(self.rows)+" From file: "+self.row_file
        print "Max Cols: "+str(self.cols)+" From file: "+self.col_file


# Pre-processing of data to make all the samples of same size.
# Currently, only appending is implemented
class ProcessData:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols


    def appendData(self, data_path, output_path):
     """
     :param data_path: data file paths, .h5py
     :param output_path:
     :return: processed data in the output_path
     """
     if not os.path.exists(output_path):
        os.makedirs(output_path)

        files = os.listdir(data_path)
        print len(files)

        for file in files:
            data_file = h5py.File(data_path+"/"+file,'r')
            data = data_file['data']
            layer = data[0, :, :, :]

            #Fixed size data
            output_data = np.zeros((1, layer.shape[0], self.rows, self.cols))
            output_data[0, :, 0:layer.shape[1], 0:layer.shape[2]] = layer

            output_data_file = h5py.File(output_path+"/"+file, 'w')
            output_data_file.create_dataset(name='data', data=output_data, compression="gzip")
            output_data_file.create_dataset(name='label', data=data_file['label'], compression="gzip")

            output_data_file.close()
            data_file.close()

    def upsampleData(self, data_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        files = os.listdir(data_path)
        print len(files)

        for file in files:
            data_file = h5py.File(data_path+"/"+file,'r')
            data = data_file['data']
            layer = data[0, :, :, :]

            output_data = np.zeros((1, layer.shape[0], self.rows, self.cols))
            output_data[0, :, 0:layer.shape[1], 0:layer.shape[2]] = layer

            output_data_file = h5py.File(output_path+"/"+file, 'w')
            output_data_file.create_dataset(name='data', data=output_data, compression="gzip")
            output_data_file.create_dataset(name='label', data=data_file['label'], compression="gzip")

            output_data_file.close()
            data_file.close()


# Data Augmentation for deep learning
class GenerateData:

    def generateRotatedData(self, data_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        files = os.listdir(data_path)
        print len(files)

        for file in files:
            data_file = h5py.File(data_path+"/"+file,'r')
            data = data_file['data']

            output_file_name = '.'.join(file.split('.')[0:-1])

            data1 = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
            data2 = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))

            for i in range(0, data.shape[1]):
                data1[0, i, :] = np.rot90(data[0, i, :])
                data2[0, i, :] = np.rot90(data[0, i, :], 2)

            output_data_file = h5py.File(output_path+"/"+output_file_name+"_1.h5", 'w')
            output_data_file.create_dataset(name='data', data=data1, compression="gzip")
            output_data_file.create_dataset(name='label', data=data_file['label'], compression="gzip")
            output_data_file.close()

            output_data_file = h5py.File(output_path+"/"+output_file_name+"_2.h5", 'w')
            output_data_file.create_dataset(name='data', data=data2, compression="gzip")
            output_data_file.create_dataset(name='label', data=data_file['label'], compression="gzip")
            output_data_file.close()

            data_file.close()







