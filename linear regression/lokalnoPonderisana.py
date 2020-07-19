import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from math import exp


#local weighted linear regression function
# x je test primer
# X su prediktori iz trening skupa Y su labele iz trening skupa
def lwl_regression(x, X, Y, tau=0.93):
    X = np.pad(X, ((0,0),(1,0)), mode='constant', constant_values=1)
    m = len(X)
    W = np.matrix(np.zeros((m,m)))
    x = np.pad(x, (1,0), mode='constant', constant_values=1)
    for i in range(m):
        xi = np.array(X[i])
        W[i, i] = exp(np.linalg.norm(x - xi) / (-2 * tau ** 2))
    #trazenje optimalnih koeficijenata
    # teta = (x.T * W * x)^-1 * x.T * W * y
    # e = (x.T * W * x)^-1
    e = np.linalg.inv(np.matmul(np.matmul(X.T, W), X))
    teta = np.matmul(np.matmul(np.matmul(e, X.T), W), Y)
    return teta

def predict(x_test, teta):
    #Dopuni X: u prvoj koloni sve 1 - 5 prediktora i 1 na pocetku (zbog teta0) za svaki primer
    x_test = np.pad(x_test, (1,0), mode='constant', constant_values=1)
    #predict y
    y_pred = np.matmul(teta, x_test)
    
    return y_pred



df = pd.read_csv('data.csv', header=None)
dataset = df.values
#Izdvajanje prediktora - prvih 5 kolona dataset-a
X = dataset[:,:5]
#standardizacija prediktora
x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)
x_stand = np.subtract(X, x_mean)
x_stand = np.divide(x_stand, x_std)
#Izdvanjanje oznaka - poslednja kolona dataset-a
Y = dataset[:,5]
#podela dataset-a na trening i validacioni
x_train = x_stand[69:,:]
x_valid = x_stand[:69,:]
y_train = Y[69:]
y_valid = Y[:69]

y_pred = np.zeros((len(x_valid), 1))
for i in range(len(x_valid)):
    teta = lwl_regression(x_valid[i], x_train, y_train)
    y_pred[i] = predict(x_valid[i], teta)
    x_train_new = np.matrix(x_valid[i]).reshape((1,5))
    y_train_new = np.array(y_valid[i]).reshape((1,))
    x_train = np.concatenate((x_train, x_train_new))
    y_train = np.concatenate((y_train, y_train_new))
    
y_pred = y_pred.ravel()
    
#Nadji srednju kvadratnu gresku
J = (1/2) * np.sum(np.square(np.subtract(y_pred, y_valid)))
print('Gubitak na validacionom skupu ', J)




