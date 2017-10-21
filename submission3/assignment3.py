import random
import pandas

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
from numpy import *

def accuracy (Y, l):
	acc = 0
	for i in range(0, len(l)):
		if (Y[i] == l [i]) :
			acc = acc + 1
	return float(acc) / len(l)

def shell (n, r_in, r_out, l):
	x = []
	y = []
	z = []
	l_ = []
	for i in range(0, n):
		theta = random.uniform(0, 2*math.pi)
		r = random.uniform(r_in, r_out)
		z_ = random.uniform(-r, r)
		x = x + [sqrt(r**2 - z_**2)*cos(theta)]
		y = y + [sqrt(r**2 - z_**2)*sin(theta)]
		z = z + [z_]
		l_ = l_ + [l]
	return vstack((x, y, z, l_))

def plot3(array, label):
	fig = pyplot.figure()
	ax = Axes3D(fig)
	t = [[],[],[]]
	for i in range(array.shape[0]):
		if label[i] == 0:
			t[0] = t[0] + [i]
		elif label[i] == 1:
			t[1] = t[1] + [i]
		elif label[i] == 2:
			t[2] = t[2] + [i]
	a = [0,0,0]
	for i in t[0]:
		a = vstack((a,array[i]))
	a = a[1:]
	ax.scatter(a[:,0],a[:,1],a[:,2],color = 'red')	
	a = [0,0,0]
	for i in t[1]:
		a = vstack((a,array[i]))
	a = a[1:]
	ax.scatter(a[:,0],a[:,1],a[:,2],color = 'green')	
	a = [0,0,0]
	for i in t[2]:
		a = vstack((a,array[i]))
	a = a[1:]
	ax.scatter(a[:,0],a[:,1],a[:,2],color = 'blue')	
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	pyplot.show()

def split(array, size):
	sh = int(array.shape[0]*size)
	return (array[:sh], array[sh:])

def LR(X_train, Y_train, X_test, Y_test):
	lr = LogisticRegression()
	lr.fit(X_train, Y_train)
	predictions = lr.predict (X_test)
	print 'LR : ' + str(accuracy(Y_test, predictions))

def LDA(X_train, Y_train, X_test, Y_test):
	lda = LinearDiscriminantAnalysis()
	lda.fit(X_train, Y_train)
	predictions = lda.predict (X_test)
	print 'LDA: ' +str(accuracy(Y_test, predictions))

def RF(X_train, Y_train, X_test, Y_test):
	rf = RandomForestClassifier(max_depth=10, random_state=0)
	rf.fit(X_train, Y_train)
	predictions = rf.predict (X_test)
	print 'RF : '+str(accuracy(Y_test, predictions))

def KNN(X_train, Y_train, X_test, Y_test):
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict (X_test)
	print 'KNN: '+str(accuracy(Y_test, predictions))

def DT(X_train, Y_train, X_test, Y_test):
	rf = DecisionTreeClassifier()
	rf.fit(X_train, Y_train)
	predictions = rf.predict (X_test)
	print 'RF : ' +str(accuracy(Y_test, predictions))

def NB(X_train, Y_train, X_test, Y_test):
	nb = GaussianNB()
	nb.fit(X_train, Y_train)
	predictions = nb.predict (X_test)
	print 'NB : '+str(accuracy(Y_test, predictions))	

def SVM(X_train, Y_train, X_test, Y_test):
	svm = SVC()
	svm.fit(X_train, Y_train)
	predictions = svm.predict (X_test)
	print 'SVM: '+str(accuracy(Y_test, predictions))	
	#print '--------------------'

def classifier():
	print "Breast Cancer Results"
	print '---------------------'
	dataset = pandas.read_csv('boobycancer.data' , usecols = range(1,11), header = None)
	dataset_ = dataset.sample(frac=1)
	#print dataset_
	array = dataset_.values
	X = array[:,:-1]
	Y = array[:,-1]

	(X_train, X_test) = split(X, .6)
	(Y_train, Y_test) = split(Y, .6)
	
	LR(X_train, Y_train, X_test, Y_test)
	LDA(X_train, Y_train, X_test, Y_test)
	KNN(X_train, Y_train, X_test, Y_test)
	RF(X_train, Y_train, X_test, Y_test)
	NB(X_train, Y_train, X_test, Y_test)
	SVM(X_train, Y_train, X_test, Y_test)
	print '---------------------'

def cluster ():
	print "Sphere"
	print "--------------------"
	M1 = shell(500, 9.5, 10.5, 0)
	M2 = shell(500, 19.5, 20.5, 1)
	M3 = shell(500, 29.5, 30.5, 2)

	X_ = hstack((M1,M2,M3))
	X = X_[:-1].T
	Y = X_[-1]
	#plot3(X, Y)

	km = KMeans(n_clusters = 3)
	l = km.fit_predict(X)
	#plot3 (X,l)
	print 'KMeans  :'+str(accuracy (Y, l))

	sc = SpectralClustering(n_clusters=3, affinity="nearest_neighbors").fit(X)
	l = sc.labels_
	#plot3 (X,l)
	print 'Spectral:'+str(accuracy (Y, l))

	gm = GaussianMixture(n_components=3, covariance_type='full').fit(X)
	l = gm.predict(X)
	#plot3 (X,l)
	print 'GMM     :'+str(accuracy (Y, l))

	db = DBSCAN(eps='7')
	l = db.fit_predict(X)
	#plot3 (X,l)
	print 'DBSCAN  :'+str(accuracy (Y, l))
	print "--------------------"

	kpca = KernelPCA(n_components='1',kernel = 'rbf')
	kpca.fit(X)
	print kpca.lambdas_
#	print kpca.singular_values_

	pca = PCA(n_components=1)
	pca.fit(X)
	print len(pca.explained_variance_ratio_)
	print pca.singular_values_


classifier()
cluster()