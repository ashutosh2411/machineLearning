# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt, matmul
from numpy import *
import numpy.matlib
#import numpy.arrange
import matplotlib.pyplot as plt

x_ = genfromtxt('Regressiondata/x.txt', delimiter=',')[np.newaxis]
x = x_.T
y_ = genfromtxt('Regressiondata/y.txt', delimiter=',')[np.newaxis]
y = y_.T

X = np.hstack((x**0, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10))
X_ = X.T

XTrain = X
yTrain = y

x = genfromtxt('Regressiondata/xts.txt', delimiter=',')[np.newaxis]
x = x_.T
y = genfromtxt('Regressiondata/yts.txt', delimiter=',')[np.newaxis]
y = y_.T

X = np.hstack((x**0, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10))
X_ = X.T

XTest = X
yTest = y_.T

	

for x in xrange(-10,10):
	w = matmul(numpy.linalg.inv(matmul(X_,X) + 2**x*np.eye(X_.shape[0])), matmul(X_,y))
	eTrain = (np.linalg.norm(x = matmul(XTrain,w)-yTrain, ord = 2))**.5
	eTest  = (np.linalg.norm(x = matmul(XTest,w)-yTest, ord = 2))**.5
	plt.scatter(x,eTrain,color='blue',label='training error')
	plt.scatter(x,eTest,color='red',label='testing error')
#	w = matmul(numpy.linalg.inv(matmul(X_,X) - 2**x*np.eye(X_.shape[0])), matmul(X_,y))
#	eTrain = (np.linalg.norm(x = matmul(XTrain,w)-yTrain, ord = 2))**.5
#	eTest  = (np.linalg.norm(x = matmul(XTest,w)-yTest, ord = 2))**.5
#	plt.scatter(-2**x,eTrain,color='blue',label='training error')
#	plt.scatter(-2**x,eTest,color='red',label='testing error')
plt.show()