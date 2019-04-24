# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:50:57 2019
SVM BOUNDARY LINE DYNAMIC PLOTTING
Install celluloid package first
!pip install celluloid
@author: omerzulal
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")
plt.style.use("seaborn-pastel")
#%% load, normalise and organise the IRIS data
from sklearn.datasets import load_iris
import numpy as np
from sklearn import preprocessing

#load iris data from sklearn datasets
iris = load_iris()

#use only petal length and petal width data, 
flowers = [1,2]

#only versicolor and virginica are the targets
target_cond = (iris.target == flowers[0]) | (iris.target == flowers[1])

#construct a dataframe with the conditions
df_features = pd.DataFrame(preprocessing.scale(iris.data[target_cond,2:]),
                           columns = iris.feature_names[2:])
df_targets = pd.DataFrame(iris.target[target_cond],columns=["Targets"])
df = pd.concat([df_features,df_targets],axis=1)

#shuffle the dataset
df = df.reindex(np.random.RandomState(seed=2).permutation(df.index))
df = df.reset_index(drop=True)

#%% train SVM dynamically by adding new data in the loop
from sklearn.svm import SVC
from celluloid import Camera
from svm_boundary_line_drawer import svm_boundary_line

#generate an SVM model
SVM_model = SVC(kernel="linear")

##generate an SVM model (alternative)
#from sklearn.svm import LinearSVC
#SVM_model = LinearSVC()

#train the SVM model with the first 2 instances
#otherwise SVM will throw error when tried to train SVM feeding samples one-by-one
#(the permutation seed above is determined for this)
init_rows = 2
X = df.iloc[:init_rows,:2]
y = df["Targets"][:init_rows]
SVM_model.fit(X,y)

classes = df.Targets.value_counts().index

scores = [SVM_model.score(X,y)]

# Dynamic plotting part
fig = plt.figure()
camera = Camera(fig)
ax = fig.add_subplot(111)

#train SVM dynamically in the for loop by adding data in the training set
for i in range(init_rows+1,len(df_targets)):
    
    #extend the data with new entries in each loop and train SVM
    X = df.iloc[:i,:2]
    y = df["Targets"][:i]
    SVM_model.fit(X,y)
    
    #get model parameters
    w0 = SVM_model.intercept_[0]
    w1 = SVM_model.coef_[0][0]
    w2 = SVM_model.coef_[0][1]
    sv_inxs = pd.Series(SVM_model.support_) #support vector indices
    
    l = df["petal length (cm)"][:i]
    w = df["petal width (cm)"][:i]
    t = df["Targets"][:i]
    
    f1 = ax.scatter(l[t==classes[0]],w[t==classes[0]],c="cornflowerblue",marker="o")
    f2 = ax.scatter(l[t==classes[1]],w[t==classes[1]],c="sandybrown",marker="^")
#    alternative visualisation of data points
#    f1 = ax.scatter(l[t==classes[0]],w[t==classes[0]],c="cornflowerblue",marker="o",s=100,edgecolors="steelblue")
#    f2 = ax.scatter(l[t==classes[1]],w[t==classes[1]],c="sandybrown",marker="^",s=100,edgecolors="sienna")
 
    #find indexes of support vectors for each class
    a0 = pd.Series(t.index[t==classes[0]])
    inx0 = a0[a0.isin(sv_inxs)].values
    a1 = pd.Series(t.index[t==classes[1]])
    inx1 = a1[a1.isin(sv_inxs)].values
    
    #plot the support vectors
    ax.scatter(l[inx0],w[inx0],c="cornflowerblue",marker="o",edgecolors="black")
    ax.scatter(l[inx1],w[inx1],c="sandybrown",marker="^",edgecolors="black")
    
    #calculate the model accuracy
    scores.append(SVM_model.score(X,y))
    
    #draw the SVM boundary line
    func_text_pos_y = svm_boundary_line(ax,w1,w2,w0)
    
    #title wont work with celluloid package, text is an alternative to workaround
    ax.text(0.25, 1.03, "SVM Training Accuracy: %.2f"%(SVM_model.score(X,y)), 
            fontweight="bold", transform=ax.transAxes)
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    
    #constant text position for the boundary function
#    ax.text(0.6, 1.03, r"$y = \frac{%.2f}{%.2f}+\frac{%.2f}{%.2f}x$"%(-w0,w2,-w1,w2), 
#            fontweight="bold", transform=ax.transAxes)
    
    #dynamic text position for the boundary function
    if func_text_pos_y < 2.95:
        ax.text(-2, func_text_pos_y, r"$y = \frac{%.2f}{%.2f}+\frac{%.2f}{%.2f}x$"%(-w0,w2,-w1,w2), 
                fontweight="bold")
        
    ax.legend([f1,f2],iris.target_names[flowers],fontsize=9)
   
    #take a snapshot of the figure
    camera.snap()

#create animation
anim = camera.animate()
#save the animation as a gif file
anim.save("SVM_Boundary_Video.gif",writer="pillow")