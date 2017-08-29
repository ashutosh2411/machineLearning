import numpy as np
from numpy import matmul
from numpy.linalg import inv
from numpy import genfromtxt, hstack, vstack
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt



# function for least square solution
def LeastSquares(Featurematrix, y):
	#least square solution
	w1 = matmul(Featurematrix.transpose(), Featurematrix)
	w2 = matmul(Featurematrix.transpose(),y)
	w = matmul(inv(w1), w2)
	
	return(w)

def RidgeRegression(Featurematrix, y, lamda):
	k = 0
	m = Featurematrix.shape[0]
	n = Featurematrix.shape[1]
	w = []
	dif = 10
	for i in range(0, n ):
		w.append(0)
	w = np.array(w)
	
	w = w [np.newaxis]
	w =w.T
	
	error = np.subtract(matmul(Featurematrix, w) , y)
	error = np.dot(error.transpose(), error) 
	error = float(error)
	
			
	while(dif > 0.00000000001):
		for j in range(0,m):
			p = Featurematrix[j] [np.newaxis]
			w1 = matmul(p, w)
			
			w1 = w1 - y[j]
			
			w1 = w1*p.T
			w2 = lamda*w
			w1 = np.subtract(w1 , w2)
			w1 = 0.05*w1
			w  = np.subtract(w , w1)
			error_new = np.subtract(matmul(Featurematrix, w) , y)
			error_new = np.dot(error_new.transpose(), error_new)
			error_new = float(error_new)			
			dif = error_new - error
			error = error_new 
		k = k+1

	return(w)		
	
def Relation():

	my_data = genfromtxt('x.txt', delimiter=',') [np.newaxis]
	my_data = my_data.T 
	X =my_data**0
	for i in range(1, 11):
		X = np.hstack((X,my_data**i))


	my_data2 = genfromtxt('y.txt', delimiter=',') [np.newaxis]
	my_data2 = my_data2.T


	X = normalize(X, axis = 0, norm = 'l2')
	my_data2 = normalize(my_data2, axis = 0, norm = 'l2')


	w = LeastSquares(X, my_data2)
	print("the least square solution:")
	print(w)


	w = RidgeRegression(X, my_data2, 1)
	print("the ridge regression solution:")
	print(w)	

	#optimalHP(X,my_data2,5)	
	

	
def optimalHP(Featurematrix, y, fold):
	m = Featurematrix.shape[0]
	set = m/fold
	error_train = []
	error_test = []
	error_validation = []
	lamda_log = []

	my_data = genfromtxt('xts.txt', delimiter=',') [np.newaxis]
	my_data = my_data.T 
	X =my_data**0
	for i in range(1, 11):
		X = np.hstack((X,my_data**i))


	my_data2 = genfromtxt('yts.txt', delimiter=',') [np.newaxis]
	my_data2 = my_data2.T


	for x in range(-10,11):
		lamda = pow(2,x)
		lamda_log.append(np.log(lamda))
		i = 0
		sq_error = 0
		sq_error_train = 0
		sq_error_test = 0
		while (i*set + set <= m):
			NewFeaturematrix = np.vstack((Featurematrix[:i*set], Featurematrix[(i+1)*set:]))
			NewY = np.vstack((y[:i*set],y[(i+1)*set:]))
			i = i+1
			w = RidgeRegression(NewFeaturematrix, NewY, lamda)
			
			error1 = np.subtract(matmul(NewFeaturematrix, w) , NewY)
			error1 = np.dot(error1.transpose(), error1) 
			sq_error_train =np.add(sq_error_train , error1)

			error2 = np.subtract(matmul(X, w) , my_data2)
			error2 = np.dot(error2.transpose(), error2) 
			sq_error_test =np.add(sq_error_test , error2)

			error = np.subtract(matmul(Featurematrix[i*set: (i+1)*set], w) , y[i*set: (i+1)*set])
			error = np.dot(error.transpose(), error) 
			sq_error =np.add(sq_error , error)

		sq_av_error = sq_error/i
		sq_error_train = sq_error_train/i
		sq_error_test = sq_error_test/i

		#for plotting graph
		error_validation.append(float(sq_av_error))
		error_train.append(float(sq_error_train))
		error_test.append(float(sq_error_test))
		if (x==-10):
			prev_error = sq_av_error
			lamda_min = lamda
		elif (sq_av_error < prev_error):
			prev_error = sq_av_error
			lamda_min = lamda
	
	error_train = np.array(error_train)
	error_validation = np.array(error_validation)
	lamda_log = np.array(lamda_log)

	print("optimal hyper parameter")
	print(lamda_min)

	plt.plot(lamda_log, error_train)
	plt.show()
	plt.plot(lamda_log, error_validation)
	plt.show()	
	plt.plot(lamda_log, error_test)
	plt.show()	

Relation()
