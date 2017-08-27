import numpy as np
from numpy import genfromtxt, matmul
import numpy.matlib

Featurematrix = genfromtxt('in_x.csv', delimiter=',')

y_ = genfromtxt('in_y.csv', delimiter=',')[np.newaxis]
y = y_.T
#w = matmul(numpy.linalg.inv(matmul(X_,X)) , matmul(X_, y))
#print (w)


def LeastSquare(X, Y):
	X_ = X.T
	return (matmul(numpy.linalg.inv(matmul(X_,X)) , matmul(X_, Y)))

def RidgeRegression(X, Y, L):
	X_ = X.T
	return (matmul(numpy.linalg.inv(matmul(X_,X) + L*np.eye(X_.shape[0])), matmul(X_,Y)))

print (LeastSquare(Featurematrix,y))

print (RidgeRegression(Featurematrix, y, .1))