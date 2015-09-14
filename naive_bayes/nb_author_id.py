#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
sys.path.append("../tools/")

from time import time
from email_preprocess import preprocess




### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
print "preprocessing train and test data sets.."
t0 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print "pre-processing time: ", round(time()-t0, 3), "s"



#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB as GNB
clf = GNB()

t0 = time()
print "training..."
clf.fit(features_train, labels_train)
print "training time: ", round(time()-t0, 3), "s"

t0 = time()
print "test score (accuracy): ", clf.score(features_test, labels_test)
print "scoring time: ", round(time()-t0, 6), "s"


#########################################################


