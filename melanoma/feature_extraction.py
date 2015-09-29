__author__ = 'minhaz_palasara'

from melanoma.melanoma_dataset import MelanomaDataset2D, MelanomaDataset3D
import os
import cPickle
import theano
import numpy as np


# @Author: Jake Varley, Minhaz Palasara
# Feature extraction from the fully connected layers,
# Features are stored in a text file per sample
def feature_extractor(batch_size, data_dir, feature_ids, result_dirs, model_file_path, is3D):

    # Names of files from the data directory
    files = ([filename for filename in os.listdir(data_dir) if ".h5" in filename])
    file_names = [data_dir+"/"+s for s in files]

    # Initializing the dataset
    if is3D:
        data_set = MelanomaDataset3D(data_dir, examples=file_names)
    else:
        data_set = MelanomaDataset2D(data_dir, examples=file_names)

    # Load the model - deserialization using Pickle
    model_file = open(model_file_path)
    model = cPickle.load(model_file)

    # output features from all the layers
    outputs = []

    for i in feature_ids:
        l0 = model.layers[i]
        outputs.append(l0.output(False))

    # compiling a function for feature extraction
    model.extract_features = theano.function(
            [model.layers[0].input],
            outputs,
            on_unused_input='ignore', allow_input_downcast=True)

    # create the data iterator
    iterator = data_set.iterator(batch_size=batch_size, mode='sequential')

    # loop through the iterator and extract the features
    j = 0

    print "Start extracting  the features"

    while iterator.has_next():
        X, Y, batch_files = iterator.next()
        features = model.extract_features(X)

        for k in range(0, len(batch_files)):
            # if indeed a file
            if len(batch_files[k]) > 0:
                for j in range(0, len(feature_ids)):
                    feature_list = features[j][k, :]
                    np.savetxt(result_dirs[j] + '.'.join(batch_files[k].split('/')[-1].split('.')[0:-1])+'.txt',
                               (feature_list,), delimiter='\n', fmt='%g')

    print "Finished fetching features!!"