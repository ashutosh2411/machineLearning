import random
import pandas
from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
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

N_ = 500						# Number of samples on a sphere

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

def split(array, size):
	sh = int(array.shape[0]*size)
	return (array[:sh], array[sh:])

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
	
	print 'LR : '+str(accuracy(Y_test, LogisticRegression().fit(X_train, Y_train).predict (X_test)))
	print 'LDA: '+str(accuracy(Y_test, LinearDiscriminantAnalysis().fit(X_train, Y_train).predict (X_test)))
	print 'RF : '+str(accuracy(Y_test, RandomForestClassifier(max_depth=100).fit(X_train, Y_train).predict (X_test)))
	print 'KNN: '+str(accuracy(Y_test, KNeighborsClassifier().fit(X_train, Y_train).predict (X_test)))
	print 'DT : '+str(accuracy(Y_test, DecisionTreeClassifier().fit(X_train, Y_train).predict (X_test)))
	print 'NB : '+str(accuracy(Y_test, GaussianNB().fit(X_train, Y_train).predict (X_test)))	
	print 'SVM: '+str(accuracy(Y_test, SVC().fit(X_train, Y_train).predict (X_test)))	
	print '---------------------'

def plot(dim,array, label, title,t):
	if dim == 3:
		fig = pyplot.figure(t+title)
		ax = Axes3D(fig)
		col1 = label == 0
		col2 = label == 1
		col3 = label == 2
		ax.scatter(array[col1,0],array[col1,1],array[col1,2])
		ax.scatter(array[col2,0],array[col2,1],array[col2,2])
		ax.scatter(array[col3,0],array[col3,1],array[col3,2])
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')
		ax.set_title(title)
		fig.savefig(t+title+'.png')
		pyplot.show()
	elif dim == 2:
		fig = pyplot.figure(t+title)
		col1 = label == 0
		col2 = label == 1
		col3 = label == 2
		pyplot.scatter(array[col1,0],array[col1,1])
		pyplot.scatter(array[col2,0],array[col2,1])
		pyplot.scatter(array[col3,0],array[col3,1])
		fig.savefig(t+title+'.png')
		pyplot.show()	
	elif dim == 1:
		fig = pyplot.figure(t+title)
		col1 = label == 0
		col2 = label == 1
		col3 = label == 2
		pyplot.scatter(array,label)
		fig.savefig(t+title+'.png')
		pyplot.show()

def cluster3d (X,t):
	print "clustering with 3 features"
	plot(3,X,KMeans(n_clusters = 3).fit_predict(X), 'KMeans',t)
	plot(3,X,SpectralClustering(n_clusters=3, affinity="nearest_neighbors").fit(X).labels_, 'SpectralClustering',t)
	plot(3,X,GaussianMixture(n_components=3, covariance_type='full').fit(X).predict(X),'GMM',t)
	plot(3,X,DBSCAN(eps='7').fit_predict(X),'DBSCAN',t)
	print "---------------------"

def cluster1d (X,e,t):
	print "clustering with 1 features"
	plot(1,X,KMeans(n_clusters = 3).fit_predict(X),'KMeans',t)
	plot(1,X,SpectralClustering(n_clusters = 3,affinity='nearest_neighbors').fit(X).labels_,'SpectralClustering',t)
	plot(1,X,GaussianMixture(n_components = 3).fit(X).predict(X),'GaussianMixture',t)
	plot(1,X,DBSCAN(eps=e).fit_predict(X),'DBSCAN',t)
	print "---------------------"

def cluster2d (X,e,t):
	print "clustering with 2 features"
	plot(2,X,KMeans(n_clusters = 3).fit_predict(X),'KMeans',t)
	plot(2,X,SpectralClustering(n_clusters = 3,affinity='nearest_neighbors').fit(X).labels_,'SpectralClustering',t)
	plot(2,X,GaussianMixture(n_components = 3).fit(X).predict(X),'GaussianMixture',t)
	plot(2,X,DBSCAN(eps=e).fit_predict(X),'DBSCAN',t)
	print "---------------------"

def kernel(x):
	return (matmul(x,x.T)+1)**2/1000

def sphere():
	print "Sphere"
	M1 = shell(N_, 9.5, 10.5, 0)
	M2 = shell(N_, 19.5, 20.5, 1)
	M3 = shell(N_, 29.5, 30.5, 2)

	X_ = hstack((M1,M2,M3))
	X = X_[:-1].T
	Y = X_[-1]
	plot(3,X, Y, 'Data','full')
	cluster3d(X,'full')
	kpca = PCA(n_components=1).fit_transform((kernel(X)))
	plot(1,kpca,Y,'Data','KPCA_2D_')
	cluster1d(kpca,50,'kPCA_1D_')
	pca = PCA(n_components=1).fit_transform(X)
	plot(1,pca,Y,'Data','PCA')
	cluster1d(pca,.5,'PCA_1D_')
	kpca = PCA(n_components=2).fit_transform((kernel(X)))
	plot(2,kpca,Y,'Data','KPCA_2D_')
	cluster2d(kpca,500,'KPCA_2D_')
	pca = PCA(n_components=2).fit_transform(X)
	plot(2,pca,Y,'Data','PCA_2D_')
	cluster2d(pca,3,'PCA_2D_')

classifier()
sphere()