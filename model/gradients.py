import numpy as np
from model.utils import onehot_array
def MSE_grad(y,y_pred):
	'''
	Derivative of MSE loss w.r.t y_pred (not w)
	'''
	return (2/len(y))*(y_pred-y)
def MAE_grad(y, y_pred):
    #TODO
    pass
def logloss_sigmoid_grad(y,y_pred):
	'''
	Derivative of sigmoid + log loss combination is equivalent to derivative of MSE 
	'''
	return MSE_grad(y,y_pred)/2
def mse_logistic_grad(y, y_pred):
    #TODO
    pass
