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

#features to use
features = ["sepal width (cm)", "petal length (cm)"]

#only versicolor and virginica are the targets
flowers = [1,2]
target_cond = (iris.target == flowers[0]) | (iris.target == flowers[1])

#construct a dataframe with the conditions
df_features = pd.DataFrame(preprocessing.scale(iris.data[target_cond,:]),
                           columns = iris.feature_names)
df_features = df_features[features]
df_targets = pd.DataFrame(iris.target[target_cond],columns=["Targets"])
df = pd.concat([df_features,df_targets],axis=1)

#shuffle the dataset
df = df.reindex(np.random.RandomState(seed=2).permutation(df.index))
df = df.reset_index(drop=True)

#%% train SVM dynamically by adding new data in the loop
from sklearn.svm import SVC
from celluloid import Camera

#play with SVM parameters
kernel = "linear"
C = 1
gamma = "auto"

#generate an SVM model
SVM_model1 = SVC(kernel="linear",C=C)
SVM_model2 = SVC(kernel="rbf",C=C,gamma=gamma)

#train the SVM model with the first 2 instances
#otherwise SVM will throw error when tried to train SVM feeding samples one-by-one
#(the permutation seed above is determined for this)
init_rows = 2
X = df.iloc[:init_rows,:2]
y = df["Targets"][:init_rows]
SVM_model1.fit(X,y)
SVM_model2.fit(X,y)

classes = df.Targets.value_counts().index
scores1 = [SVM_model1.score(X,y)]
scores2 = [SVM_model2.score(X,y)]

# Dynamic plotting part
fig = plt.figure(figsize=(12,5))
camera = Camera(fig)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#train SVM dynamically in the for loop by adding data in the training set
for i in range(init_rows+1,len(df_targets)):
    
#    ax1.cla()
#    ax2.cla()
    
    #extend the data with new entries in each loop and train SVM
    X = df.iloc[:i,:2]
    y = df["Targets"][:i]
    SVM_model1.fit(X,y)
    SVM_model2.fit(X,y)
    
    f1 = df[features[0]][:i]
    f2 = df[features[1]][:i]
    t = df["Targets"][:i]
    
    f11 = ax1.scatter(f1[t==classes[0]],f2[t==classes[0]],c="cornflowerblue",marker="o")
    f12 = ax1.scatter(f1[t==classes[1]],f2[t==classes[1]],c="sandybrown",marker="^")

    f21 = ax2.scatter(f1[t==classes[0]],f2[t==classes[0]],c="cornflowerblue",marker="o")
    f22 = ax2.scatter(f1[t==classes[1]],f2[t==classes[1]],c="sandybrown",marker="^")
    
    #calculate the model accuracy
    scores1.append(SVM_model1.score(X,y))
    scores2.append(SVM_model2.score(X,y))
    
    #draw the SVM boundary line
    #prepare data for decision boundary plotting
    x_min = X.iloc[:,0].min()
    x_max = X.iloc[:,0].max()
    y_min = X.iloc[:,1].min()
    y_max = X.iloc[:,1].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z1 = SVM_model1.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z1 = Z1.reshape(XX.shape)
    Z2 = SVM_model2.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z2 = Z2.reshape(XX.shape)

    #plot the decision boundary
    ax1.contour(XX, YY, Z1, colors=['darkgrey','dimgrey','darkgrey'],
                linestyles=[':', '--', ':'], levels=[-.5, 0, .5])
    ax2.contour(XX, YY, Z2, colors=['darkgrey','dimgrey','darkgrey'],
                linestyles=[':', '--', ':'], levels=[-.5, 0, .5])
    
    #title wont work with celluloid package, text is an alternative to workaround
    ax1.text(0.25, 1.03, "SVM (Linear) Training Accuracy: %.2f"%(SVM_model1.score(X,y)), 
            fontweight="bold", transform=ax1.transAxes)
    ax1.set_xlim([-3,3])
    ax1.set_ylim([-3,3])
    
    ax2.text(0.25, 1.03, "SVM (RBF) Training Accuracy: %.2f"%(SVM_model2.score(X,y)), 
            fontweight="bold", transform=ax2.transAxes)
    ax2.set_xlim([-3,3])
    ax2.set_ylim([-3,3])
    
    #x-y labels
    ax1.set_xlabel(features[0])
    ax1.set_ylabel(features[1])
    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])
       
#    #dynamic text position for the boundary function
#    if func_text_pos_y < 2.95:
#        ax.text(-2, func_text_pos_y, r"$y = \frac{%.2f}{%.2f}+\frac{%.2f}{%.2f}x$"%(-w0,w2,-w1,w2), 
#                fontweight="bold")
        
    ax1.legend([f11,f12],iris.target_names[flowers],fontsize=9)
    ax2.legend([f21,f22],iris.target_names[flowers],fontsize=9)
#    plt.pause(0.1)
   
    #take a snapshot of the figure
    camera.snap()

#create animation
anim = camera.animate()
#save the animation as a gif file
anim.save("SVM_Boundary_RBF_vs_Linear.gif",writer="pillow")