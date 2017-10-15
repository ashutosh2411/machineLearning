import time
import numpy as np
import numpy as np
from svmutil import *
import matplotlib.pyplot as plt
from pandas import DataFrame 
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score 
from sklearn.metrics import confusion_matrix 

def getKernelSVMSolution(Xtr, Ytr, C, G, Xts, Yts):
	model   = svm_train	 (svm_problem(Ytr, Xtr), svm_parameter('-t 2 -g '+str(G)+' -c '+str(C)+' -b 1 -q'))
	L, A, V = svm_predict(Yts,Xts,model,'-b 1')
	dist = [(L[i]*V[i][int((abs(int(L[i]))-int(L[i]))/2)]) for i in range(len(L))]
	return L, dist

def one_vs_one(Xtr, Ytr, C, Xts, Yts):
	Prediction=[]
	Label=[]
	for i in range(10):
		for j in range(10):
			if(i>=j):
				continue
			else:
				X=[]
				Y=[]
				for r in range(len(Ytr)):
					if(i==Ytr[r]):
						X.append(Xtr[r])
						Y.append(1)
					elif(j==Ytr[r]):
						X.append(Xtr[r])
						Y.append(-1)	
				M=np.median(cdist(DataFrame(X).fillna(0).values, DataFrame(X).fillna(0).values)[np.triu_indices(DataFrame(X).fillna(0).values.shape[0],1)])**2		# median distance
				Prediction.append(getKernelSVMSolution(X,Y,C,3/M,Xts,Yts)[0])
	Prediction=np.matrix(Prediction)
	for x in range(Prediction.shape[1]):
		M=np.zeros((10,10))
		M[np.triu_indices(10,1)]=Prediction[:,x].T 
		mapping={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
		for r in range(M.shape[0]):
			for c in range(M.shape[1]):
				if(M[r,c]>0):
					mapping[r]=mapping[r]+1
				elif(M[r,c]<0):
					mapping[c]=mapping[c]+1 
		Label.append([x for x in mapping.keys() if mapping[x]==max(mapping.values())][0])				# max key of the mapping
	return Label

def one_vs_rest(Xtr,Ytr,C,Xts,Yts):
	Prediction=[]
	Label=[]
	for i in range(10):
		X=[]
		Y=[]
		for r in range(len(Ytr)):
			if(i==Ytr[r]):
				X.append(Xtr[r])
				Y.append(1)
			else:
				X.append(Xtr[r])
				Y.append(-1)
		M=np.median(cdist(DataFrame(X).fillna(0).values, DataFrame(X).fillna(0).values)[np.triu_indices(DataFrame(X).fillna(0).values.shape[0],1)])**2		# median distance
		Prediction.append(getKernelSVMSolution(X,Y,C,3/M,Xts,Yts)[1])
	Prediction=np.matrix(Prediction)
	for x in range(Prediction.shape[1]):
		mapping = {i:Prediction[i,x] for i in range(10)}
		Label.append([x for x in mapping.keys() if mapping[x]==max(mapping.values())][0])
	return Label

def misMatchImage(X,Y,P,t):
	C=0
	L=[]
	for i in range(len(Y)):
		if(Y[i]!=P[i]):
			C=C+1
			L.append(i)
	w=int(pow(C,0.5))+1
	c=0
	f,a=plt.subplots(w,w)
	for i in range(w):
		for j in range(w):
			if(C<=c):
				a[i,j].axis('off')
				continue
			a[i,j].imshow(X[L[c],:].reshape(16,16),cmap='gray')
			a[i,j].set_xlabel(str(int(Y[L[c]]))+'->'+str(int(P[L[c]])),labelpad=2)
			plt.setp(a[i,j].get_xticklabels(),visible=False)
			plt.setp(a[i,j].get_yticklabels(),visible=False)
			c=c+1
	plt.suptitle(t+str(C), fontsize=14, fontweight='bold')
	plt.show()	

def mat2dict(X):
	return [{x+1:y[0,x] for x in range(y.shape[1]) if y[0,x]!=0} for y in X]

starTime=time.time()
Xtr=mat2dict(np.matrix(np.loadtxt('../../USPSTrain.csv', delimiter=',')))
Ytr=np.loadtxt('../../USPSTrainLabel.csv', delimiter = ',')
Xts=mat2dict(np.matrix(np.loadtxt('../../USPSTest.csv', delimiter=',')))
Yts=np.loadtxt('../../USPSTestLabel.csv', delimiter = ',')
LoadTime=time.time()

print "One vs One: "
Yo_o=one_vs_one(Xtr,Ytr,100,Xts,Yts)
o_oTime=time.time()
print "One vs Rest: "
Yo_r=one_vs_rest(Xtr,Ytr,100,Xts,Yts)
o_rTime=time.time()

print "\nOneVsOne Scheme\n",confusion_matrix(Yts,Yo_o)
print "\nOneVsRest Scheme\n",confusion_matrix(Yts,Yo_r)

print "\nf1_score OneVsOne\t:",f1_score(Yts,Yo_o,average="macro")
print "f1_score SVM OneVsRest\t:",f1_score(Yts,Yo_r,average="macro")

print "\nRuntime SVM OneVsOne\t:",o_oTime-LoadTime,"seconds"
print "Runtime SVM OneVsRest\t:",o_rTime-o_oTime,"seconds"

print 'Loading mismatched images'
misMatchImage(np.loadtxt('../../USPSTest.csv', delimiter=','),Yts,Yo_o,'SVM OneVsOne Mismatch Count : ')
misMatchImage(np.loadtxt('../../USPSTest.csv', delimiter=','),Yts,Yo_r,'SVM OneVsRest Mismatch Count : ')