# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:41:43 2018

@author: SAURABH
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
from plot_decision_region import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


forest = RandomForestClassifier(criterion="entropy",n_jobs=2,random_state=1)
forest.fit(X_train,y_train)

plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105,150))

plt.xlabel("petel length [cm]")
plt.ylabel("petel width [cm]")
plt.legend(loc="upper left")
plt.show()

y_pred = forest.predict(X_test)

print("accurecy : %.3f"% accuracy_score(y_test,y_pred))
