"""
	Finding optimal value od lamda using K Fold cross validation technique
"""

# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt, matmul
from numpy import *
import numpy.matlib
from numpy.linalg import norm, inv
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

def RidgeRegression(X, Y, L):
	X_ = X.T
	return (matmul(inv(matmul(X_,X) + L*np.eye(X_.shape[0])), matmul(X_,Y)))

def CrossValidation(X, Y, fold):
	#X.shape[0] = X.shape[0]
	set_length = X.shape[0]/fold
	TrainError = []
	TestError = []
	ValidationError = []
	L = []

	for x in range(-10,11):
		lamda = 2**x
		L = hstack((L, x))
		i = 0
		sqError = 0
		sqTrainError = 0
		sqTestError = 0
		while i*set_length + set_length <= X.shape[0]:
			CvX = np.vstack((X[:i*set_length], X[(i+1)*set_length:]))
			CvY = np.vstack((Y[:i*set_length],Y[(i+1)*set_length:]))

			w = RidgeRegression(CvX, CvY, lamda)

			error1 = norm(x=(matmul(CvX, w) - CvY), ord =2)
			sqTrainError = sqTrainError + error1

			error2 = norm(x=(matmul(X, w) - Y), ord =2)
			sqTestError = sqTestError + error1

			error = norm(matmul(X[i*set_length:(i+1)*set_length], w) - y[i*set_length:(i+1)*set_length])
			sqError = sqError + error

			i = i + set_length

		sqError = sqError / set_length**2
		sqTrainError = sqTrainError / ((X.shape[0]-set_length)*set_length)
		sqTestError = sqTestError / (set_length * Y.shape[0])

		ValidationError.append(sqError)
		TrainError.append(sqTrainError)
		TestError.append(sqTestError)
		if x == -10:
			prevError = sqError
			lamdaMin = lamda
		elif sqError<prevError:
			prevError = sqError
			lamdaMin = lamda

	print ("optimal lamda: "+str(lamdaMin))

	plt.plot(L, TrainError, 'r')
	plt.plot(L, ValidationError, 'g')
	plt.plot(L, TestError, 'b')
	plt.show()

CrossValidation(X, y, 5)	

#for x in xrange(-10,10):
#	w = matmul(numpy.linalg.inv(matmul(X_,X) + 2**x*np.eye(X_.shape[0])), matmul(X_,y))
#	eTrain = (np.linalg.norm(x = matmul(XTrain,w)-yTrain, ord = 2))**.5
#	eTest  = (np.linalg.norm(x = matmul(XTest,w)-yTest, ord = 2))**.5
#	plt.scatter(x,eTrain,color='blue',label='training error')
#	plt.scatter(x,eTest,color='red',label='testing error')
#	
##	w = matmul(numpy.linalg.inv(matmul(X_,X) - 2**x*np.eye(X_.shape[0])), matmul(X_,y))
##	eTrain = (np.linalg.norm(x = matmul(XTrain,w)-yTrain, ord = 2))**.5
##	eTest  = (np.linalg.norm(x = matmul(XTest,w)-yTest, ord = 2))**.5
##	plt.scatter(-2**x,eTrain,color='blue',label='training error')
#	plt.scatter(-2**x,eTest,color='red',label='testing error')
#plt.show()