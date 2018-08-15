import json
import numpy as np
from numpy import linalg as LA
from math import sqrt
#from multiprocessing import Pool


class PegasosSVM:

    def __init__(self):
        self.w_learned = None
        

    def __objective_function(self, X, y, w, lamb):
        N = len(y)

        #regularization part of objective function
        w_l2_norm = LA.norm(w)
        reg_param = 0.5*lamb*(w_l2_norm**2)

        #average hinge loss
        total = 0
        for n in range(N):
            total += max(0, (1-(y[n]*np.dot(w,X[n])))**2)
        avg_loss = total/N

        train_obj = reg_param + avg_loss

        return train_obj


    def fit(self, Xtrain, ytrain, lamb=0.0001, k=100, max_iterations=1000):
        #initialization
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)
        N = Xtrain.shape[0]
        D = Xtrain.shape[1]
        w = np.random.rand(D)*(1/(sqrt(lamb)*D))
        assert(LA.norm(w)<=1/sqrt(lamb))    # choose w such that ||W||<=1/sqrt(lamb)

        np.random.seed(0)

        train_obj = []


        for iter in range(1, max_iterations + 1):
            A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch
            lr = 1/(lamb*iter)
            
            grad = 0                                               #gradient
            for n in A_t:
                if ytrain[n]*np.dot(w,Xtrain[n])<1:
                    grad += ytrain[n]*Xtrain[n]
            grad = (lr/k)*grad

            w_half = ((1-(lr*lamb))*w)+grad
            w_next = min(1,1/(sqrt(lamb)*LA.norm(w_half)))*w_half
            train_obj.append(self.__objective_function(Xtrain,ytrain,w_next,lamb))
            w = w_next

        self.w_learned = w
        return w, train_obj


    def score(self, Xtest, ytest, t = 0.):
        Xtest = np.array(Xtest)
        ytest = np.array(ytest)
        N = Xtest.shape[0]
        y_pred = np.array([-1 for i in range(N)])

        for n in range(N):
            y_pred[n] = np.sign(np.dot(self.w_learned,Xtest[n]))

        acc = ytest*y_pred
        acc = acc.clip(min = 0)
        acc = np.sum(acc)
        test_acc = acc/N

        return test_acc

    def predict(self, X):
        return np.sign(np.dot(self.w_learned,X))
