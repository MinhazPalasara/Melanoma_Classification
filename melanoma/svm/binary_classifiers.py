__author__ = 'minhazpalasara'


import os
import h5py
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from random import shuffle

# very light wrapper around our SVC in sklearn,
# in our problem we need class weights as input
class SVMBinaryClassifier:

    def __init__(self, kernel, gamma, C, class_weigth):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.class_weight = class_weigth

    def cross_validate(self, features, labels, folds):

        model = SVC(kernel=self.kernel, gamma=self.gamma, C=self.C, class_weight=self.class_weight)

        k_fold = cross_validation.KFold(n=features.shape[0], n_folds=folds, shuffle=True)

        true_positives = 0
        true_negatives = 0

        total_positives = 0
        total_negatives = 0

        for train, test in k_fold:
            # converting two dimensional array(matrix) to simply an array as it has only one dimension
            train_label = labels[train]
            train_label = np.reshape(train_label, (train_label.shape[0]))

            classifier = model.fit(features[train], train_label)
            predictions = classifier.predict(features[test])

            for prediction, actual in zip(predictions, labels[test]):

                if actual == -1:
                    total_negatives += 1
                else:
                    total_positives += 1

                if prediction == actual:
                    if prediction == -1:
                        true_negatives += 1
                    else:
                        true_positives += 1

        sensitivity = (true_positives * 100.0)/total_positives
        specificity = (true_negatives * 100.0)/total_negatives

        return sensitivity, specificity
