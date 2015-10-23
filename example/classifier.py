__author__ = 'Minhaz_Palasara'

import numpy as np
from utils.util import  createLableMap
from utils.util import load_features, load_combined_features, load_yml_features
from melanoma.svm.binary_classifiers import SVMBinaryClassifier



# Generating parameter range based on the parameter
def genRange(start, end, step):

    range = [start]
    current = start

    while current < end:
        current = current + step
        range.append(current)

    return range

label_file='LesionResults-Raw.txt'
feature_path1 = '/media/minhazpalasara/Melanoma1/BEST_SETTINGS/melanin-150/7/dense4/'
feature_path2 = '/media/minhazpalasara/Melanoma1/BEST_SETTINGS/reduced-features-long-run-suborder6/5/dense3'

print "Loading Features.............",
map=createLableMap(label_file)
features, labels = load_features(feature_path1, map)
print "Done"

weights = genRange(0.1, 1.0, 0.01)
weights = genRange(0.01, 0.1, 0.01)
gamma_range = np.logspace(-10, 1, 20)
print gamma_range
total_rounds = 10
folds = 5

print "Classifying Features: @ "+feature_path1

for weight in weights:
    print "###################### Weight: "+str(weight)+" ##############################################"

    best_sensitivity = 0
    best_specificity = 0
    sensitivity_deviation = 0.0
    specificity_deviation = 0.0
    best_gamma = 0
    for gamma in gamma_range:

        sensitivities = []
        specificities = []

        # derive results over multiple runs                                                                                                                                                 for stable stats
        for i in range(0, total_rounds):
            classifier = SVMBinaryClassifier(kernel='rbf', gamma=gamma, C=1000, class_weigth={-1: weight, 1: 1})
            sensitivity, specificity = classifier.cross_validate(features, labels, folds)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        if np.mean(sensitivities) + np.mean(specificities) > (best_sensitivity + best_specificity):
            best_sensitivity = np.mean(sensitivities)
            best_specificity = np.mean(specificities)
            sensitivity_deviation = np.std(sensitivities)
            specificity_deviation = np.std(specificities)
            best_gamma =gamma

    print "Best Results are at: "+ str(best_gamma)
    print " Sensitivity Mean: "+str(best_sensitivity)+" Deviation: "+str(sensitivity_deviation)
    print " Specificity Mean: "+str(best_specificity)+" Deviation: "+str(specificity_deviation)