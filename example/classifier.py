__author__ = 'Minhaz Palasara'

from utils.util import  createLableMap
from utils.util import load_features
from melanoma.binary_classifiers import SVMBinaryClassifier
import numpy as np


def genRange(start, end, step):

    range = [start]
    current = start

    while current < end:
        current = current + step
        range.append(current)

    return range

label_file='LesionResults-Raw.txt'
feature_path = '/home/roboticslab/ALTERNATE-STTINGS/2/dense4/'

map=createLableMap(label_file)
features, labels = load_features(feature_path, map)

weights = genRange(0.01, 0.1, 0.01)

folds = 5

for weight in weights:

    sensitivities = []
    specificities = []

    total_rounds  = 1

    # derive result over multiple run for stable stats
    print "###################### Weight: "+str(weight)+" ##############################################"
    for i in range(0, total_rounds):
        print str(i)+"..",
        classifier = SVMBinaryClassifier(kernel='rbf', gamma=0.0, C=100000, class_weigth={-1: weight, 1: 1})
        sensitivity, specificity = classifier.cross_validate(features, labels, folds)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    print " "
    print " Sensitivity Mean: "+str(np.mean(sensitivities))+" Deviation: "+str(np.std(sensitivities))
    print " Specificity Mean: "+str(np.mean(specificities))+" Deviation: "+str(np.std(specificities))