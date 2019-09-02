import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import platform
import os

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float32')
        Y = np.array(Y)
        return X, Y
def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for i in range(1,6):
        f = os.path.join(ROOT, "data_batch_%d" % (i,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_tr = np.concatenate(xs)
    Y_tr = np.concatenate(ys)
    X_te, Y_te = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return X_tr,Y_tr,X_te,Y_te
def load_data(num_train = 49000, num_val = 1000, num_test = 1000, subtract_mean = True):
    cifar_10_dir = "cifar-10-python/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar_10_dir)
    mask = list(range(num_train, num_train + num_val))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_train))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    if subtract_mean:
        mean = np.mean(X_train, axis = 0)
        X_train -= mean
        X_val -= mean
        X_test -= mean
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


class K_nearest_neighbor:
    def __init__(self):
        pass
    def train(self,X,y):
        self.X_train = X
        self.y_train = y
    def l2_distance(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            dist[i,:]=np.sqrt(np.sum((X[i]-self.X_train)**2,axis=1))
        return dist
    def predict(self, X, k =1):
        dists = self.l2_distance(X)
        return self.predict_labels(dists,k)
    def predict_labels(self, dist, k=1):
        num_test = dist.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            top_k_index = np.argsort(dist[i])[:k]
            closest_y = self.y_train[top_k_index]
        vote = Counter(closest_y)
        count = vote.most_common()
        y_pred[i] = count[0][0]
        return y_pred

classifier = K_nearest_neighbor()
cifar_10_dir = "cifar-10-python/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar_10_dir)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)
classifier.train(X_train, y_train)
dists = classifier.l2_distance(X_test)
print(dists.shape)