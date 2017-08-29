import numpy as np
from numpy import genfromtxt, matmul
from numpy.linalg import norm, inv
import numpy.matlib
from sklearn.preprocessing import normalize

#Featurematrix = genfromtxt('in_x.csv', delimiter=',')
x_ = genfromtxt('Regressiondata/x.txt', delimiter=',')[np.newaxis]
x = x_.T
y_ = genfromtxt('Regressiondata/y.txt', delimiter=',')[np.newaxis]
y = y_.T
y = normalize(y, axis = 0, norm = 'l2')
Featurematrix = np.hstack((x**0, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10))
Featurematrix = normalize(Featurematrix, axis = 0, norm = 'l2') 
#X_ = X.T

#y_ = genfromtxt('in_y.csv', delimiter=',')[np.newaxis]
#y = y_.T

def LeastSquare(X, Y):
	X_ = X.T
	return (matmul(inv( matmul(X_,X)) , matmul(X_, Y)))
def RidgeRegression(X, Y, L):
	X_ = X.T
	return (matmul(inv(matmul(X_,X) + L*np.eye(X_.shape[0])), matmul(X_,Y)))
def GradientDescent(X,Y,L):
	X_ = X.T
	count = 0
	w = np.ones((X.shape[1], 1))
	t = .5*w
	esp = 10**-20
	while(all(i >= esp for i in np.fabs(w-t))):
		t = w
		count = count + 1
		w = w - .005*(matmul(matmul(X_,X),w)-matmul(X_,Y)+L*w)
	#print (w-t)
	print ('Gradient descent achieved after '+str(count)+' iterations')
	t = w
	return t
def StochasticGrad(X,Y,L):
	k = 0
	j = 0
	diff = 10
	w = np.ones((X.shape[1], 1))
	error = norm(x=(matmul(X,w)-y), ord = 2)
	while diff > 10**-10:
		for i in range(0,X.shape[0]):
			p = X[i][np.newaxis]
			#print y[i].shape
			#w1 = .0000005*(matmul(p.T,matmul(p,w) - y[i]) - L*w)
			w = w-.005*(matmul((matmul(p,w) - y[i]).T, p) - L*w)
			errorNew = norm(x=(matmul(X,w)-y), ord = 2)
			diff = errorNew-error
			#print diff
			error = errorNew
			#print error
		k = k+1
	print k
	return w

print (LeastSquare(Featurematrix,y))
print (RidgeRegression(Featurematrix, y, 1))
print (GradientDescent(Featurematrix, y, 1))
print (StochasticGrad(Featurematrix, y, 1))