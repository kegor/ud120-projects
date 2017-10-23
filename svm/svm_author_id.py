#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=10000)

t0 = time()
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
# 216.074 s
# 0.061 s (for 1%)

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"
# 262.014 s
# 4.619 s (for 1%)

print accuracy_score(pred, labels_test) 
# 0.981226533166 (for 50%, also initially was used 50% of data for results in comments above)
# 0.884527872582 (for 1%)
# 0.616040955631 (for rbf)
# 0.892491467577 (for 1% & C=10000)
# 0.990898748578 (for 100% & C=10000)

