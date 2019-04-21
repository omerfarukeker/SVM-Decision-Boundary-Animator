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
    ax.plot(x_vals, y_vals, ':k')
    return -w0/w2 + (-w1/w2) * -2