# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt, matmul
from numpy import *
import numpy.matlib
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt


x_ = genfromtxt('Regressiondata/x.txt', delimiter=',')[np.newaxis]
x = x_.T
y_ = genfromtxt('Regressiondata/y.txt', delimiter=',')[np.newaxis]
y = y_.T

X = np.hstack((x**0, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10))
X_ = X.T

l = [10**x for x in xrange(-10,10)]
ridgecv = RidgeCV(alphas=l, fit_intercept = True , normalize=True, store_cv_values = True)
ridgecv.fit(X, y)
#print (ridgecv.alpha_)

#X_ = X.T
#s = 0
w = (matmul(numpy.linalg.inv(matmul(X_,X) + ridgecv.alpha_*np.eye(X_.shape[0])), matmul(X_,y)))
#E = .5*matmul((y-matmul(X,w)).T, y-matmul(X,w)) + ridgecv.alpha_*1*.5*matmul(w.T,w)
#print ('Train Error'+str(E))

XTrain = X
yTrain = y

xTest_ = genfromtxt('Regressiondata/xts.txt', delimiter=',')[np.newaxis]
yTest_ = genfromtxt('Regressiondata/yts.txt', delimiter=',')[np.newaxis]
xTest = xTest_.T
yTest = yTest_.T
X = np.hstack((xTest**0, xTest, xTest**2, xTest**3, xTest**4, xTest**5, xTest**6, xTest**7, xTest**8, xTest**9, xTest**10))
#print (X.shape)

y_hat = matmul(X,w + ridgecv.alpha_)
s = 0
E = .5*matmul((yTest-matmul(X,w)).T, yTest-matmul(X,w)) + ridgecv.alpha_*1*.5*matmul(w.T,w)
print ('Test  Error'+str(E))
XTest = X
E_train = []
for x in xrange(-1,20):
	E_train = .5*matmul((yTrain-matmul(XTrain,w)).T, yTrain-matmul(XTrain,w)) + 3**x*1*.5*matmul(w.T,w)
	E_test  = .5*matmul((yTest-matmul(XTest,w)).T, yTest-matmul(XTest,w)) + 3**x*1*.5*matmul(w.T,w) 
	print(E_train,E_test)
	plt.scatter(x,E_train,color='blue',label='training error')
	plt.scatter(x,E_test,color='red',label='training error')
plt.show()