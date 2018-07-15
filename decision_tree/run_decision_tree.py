# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:47:08 2018

@author: SAURABH
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from plot_decision_region import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
tree.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

y_pred = tree.predict(X_test)
print("accuurecy :%.3f"%  accuracy_score(y_test, y_pred))