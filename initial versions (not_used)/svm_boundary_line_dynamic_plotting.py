# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:50:57 2019
SVM BOUNDARY LINE DYNAMIC PLOTTING
@author: omerzulal
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")
plt.style.use("seaborn-pastel")
#%% load, normalies and organise the IRIS data
from sklearn.datasets import load_iris
import numpy as np
from sklearn import preprocessing
iris = load_iris()
#use only petal length and petal width data, only versicolor and virginica targets
target_cond = (iris.target == 1) | (iris.target == 2)
df_features = pd.DataFrame(preprocessing.scale(iris.data[target_cond,2:]),
                           columns = iris.feature_names[2:])
df_targets = pd.DataFrame(iris.target[target_cond],columns=["Targets"])
df = pd.concat([df_features,df_targets],axis=1)

#shuffle the dataset
df = df.reindex(np.random.permutation(df.index))
df = df.reset_index(drop=True)

###plot classes with different colours
#plt.scatter(df["petal length (cm)"],df["petal width (cm)"],c=df["Targets"])

#%% train SVM dynamically by adding new data in the loop
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib.animation as an

def svm_boundary_line(ax,w, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = -intercept/w[1] + (-w[0]/w[1]) * x_vals
    plt.plot(x_vals, y_vals, '--')

#generate an SVM model
SVM_model = LinearSVC()

init_rows = 10
X = df.iloc[:init_rows,:2]
y = df["Targets"][:init_rows]
    
SVM_model.fit(X,y)

scores = [SVM_model.score(X,y)]
lines = []

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(init_rows+1,len(df_targets)):
    #clear the axes to avoid crowdy plotting
    ax.cla()
    
    #extend the data with new entries in each loop and train SVM
    X = df.iloc[:i,:2]
    y = df["Targets"][:i]
    SVM_model.fit(X,y)
    
    #plot the data and model
    ax.scatter(df["petal length (cm)"][:i],df["petal width (cm)"][:i],c=df["Targets"][:i])
    scores.append(SVM_model.score(X,y))
    
    #draw the SVM boundary line
    svm_boundary_line(ax,SVM_model.coef_[0],SVM_model.intercept_)
    
    ax.set_title("SVM Training Accuracy: %.2f"%(SVM_model.score(X,y)),fontweight="bold")
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    
    plt.pause(0.1)