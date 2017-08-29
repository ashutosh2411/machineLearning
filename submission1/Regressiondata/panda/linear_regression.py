import numpy as np



hyper_dict = {}

def least_square(featurematrix,y):

    featurematrix_transpose = np.transpose(featurematrix)
    a = featurematrix_transpose.dot(featurematrix)
    a_inv = np.linalg.inv(a)
    b = featurematrix_transpose.dot(y)
    w = a_inv.dot(b)
    return w


def ridge_regression(featurematrix,y,lambd):
    """            cost_current += (h-y[i]) * (h-y[i])
            cost_current = cost_current /  m
            cost_prev = cost_current  """
    alpha = 0.1
    np.random.shuffle(featurematrix)
    w = np.zeros(featurematrix.shape[1])
    m = featurematrix.shape[0]

    for i in range(10):
        for i in range(m):
            h = w.dot(featurematrix[i])
            grad_ole_sgd = (h-y[i])*featurematrix[i]
            grad_ridge = lambd*w
            grad = grad_ole_sgd+grad_ridge
            w = w - alpha*grad
    return w


def featurenormalization(vector):

    mean = np.mean(vector,dtype=np.float64)
    std = np.std(vector,dtype=np.float64)
    meanvec = mean*np.ones(vector.shape)
    normvec = (vector - meanvec)/std
    return(normvec)


def createfeturematrix(feature):

    featurematrix = np.ones(feature.shape)

    for i in range(1,11):
        new = feature**i
        new = featurenormalization(new)
        featurematrix = np.hstack((featurematrix,new))

    return(featurematrix)

def leasterror(w,x,y):
    h = x.dot(w)
    least_error = np.transpose(h-y).dot(h-y)
    return least_error


def split(train,y,k,i):

    length = train.shape/k
    if i<k-1:
        cross_set = train[i*length:(i+1)*length]
        pre_cross = train[:i * length]
        post_cross = train[(i + 1) * length:]
        train_set = np.hstack((pre_cross,post_cross))
        y_cross = y[i * length:(i + 1) * length]
        y_pre = y[:i * length]
        y_post = y[(i + 1) * length:]
        y_train = np.hstack((y_pre,y_post))
    else:
        cross_set = train[i*length:]
        train_set = train[:i*length]
        y_cross = y[i * length:]
        y_train = y[:i * length]

    return (train_set,cross_set,y_train,y_cross)


def crossvalidation(k,train):

    error = np.zeros(1)
    m = train.shape
    min_error = float('inf')
    for key in hyper_dict:
        for i in range(k):
            (train_set,cross_set,y_train,y_cross) = split(train,k,i)
            x_train_set = createfeturematrix(train_set)
            x_cross_set = createfeturematrix(cross_set)
            w = ridge_regression(x_train_set,y_train,key)
            train_error = leasterror(w,x_train_set,y_train)
            cross_error = leasterror(w,x_cross_set,y_cross)
            error = error+cross_error
        hyper_dict[key] = w
        avg_error = error/m
        if avg_error < min_error:
            min_error = avg_error
            min_lamda = key

    return min_lamda









def main():

    fp_feature = open('x.txt','r')
    fp_y = open('y.txt','r')
    lst = []

    for i in fp_feature:
        i = i.strip().split(',')
        lst.append(i)

    #print(lst)
    a = np.array(lst,float)

    lst = []
    for i in fp_y:
        i = i.strip().split(',')
        lst.append(i)

    y = np.array(lst,float)
    list = []
    for i in range(21):
        list.append(2**(i-10))


    featurematrix = createfeturematrix(a)
    w_ole = least_square(featurematrix,y)
    w_ridge = ridge_regression(featurematrix,y,1)

    print(w_ole)
    print(w_ridge)


if __name__ == "__main__":
    main()
