# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:53:43 2019
SVM boundary line drawing function
@author: omerzulal
"""
import numpy as np

def svm_boundary_line(ax,w1,w2,w0):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = -w0/w2 + (-w1/w2) * x_vals
    
    margin = 1 / np.sqrt(np.sum(np.array([w1,w2]) ** 2))
    margin_bot = y_vals - np.sqrt(1 + (-w1/w2) ** 2) * margin
    margin_top = y_vals + np.sqrt(1 + (-w1/w2) ** 2) * margin
#    margin_bot = y_vals - 1/w2
#    margin_top = y_vals + 1/w2
    
    #plot the hyperplane
    ax.plot(x_vals, y_vals, '--',c="dimgrey")
    
    #plot the margins
    ax.plot(x_vals, margin_top,':',c="darkgrey")
    ax.plot(x_vals, margin_bot,':',c="darkgrey")
    
    return -w0/w2 + (-w1/w2) * -2



