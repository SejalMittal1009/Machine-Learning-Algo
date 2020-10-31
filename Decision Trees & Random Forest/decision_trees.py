# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 08:15:34 2019

@author: 17BIS0091
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate,train_test_split
#from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

from utilities import visualize_classifier
from sklearn import tree
import graphviz

input_file='data_decision_trees.txt'
data=np.loadtxt(input_file,delimiter=',')
X,y=data[:,:-1],data[:,-1]

#Seperate input data into two classes based on labels
class_0=np.array(X[y==0])
class_1=np.array(X[y==1])

#Visualize input data
plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],s=None,facecolor='black',edgecolors='blue',linewidth=1,marker='x')
plt.scatter(class_1[:,0],class_1[:,1],s=None,facecolor='white',edgecolors='red',linewidth=1,marker='o')
plt.title('Input_data')

#Split Data into training and testing datasets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=5)

#Decision Tree Classifier
params={'random_state':0,'max_depth':4}
classifier=DecisionTreeClassifier(**params)
classifier.fit(X_train,y_train)
visualize_classifier(classifier,X_train,y_train)

y_test_pred=classifier.predict(X_test)
visualize_classifier(classifier,X_test,y_test)

#Evaluate classifier performance
class_names=['Class-0','Class-1']
print("\n"+"#"*40)
print("\n Classifer Performance on training dataset\n")
print(classification_report(y_test,y_test_pred,target_names=class_names))
print("#"*40+"\n")
plt.show()

from sklearn.tree import export_graphviz
#Export as dot file
export_graphviz(classifier,out_file='tree1.dot',class_names=['0','1'],rounded=True,proportion=False,precision=2,filled=True)

      