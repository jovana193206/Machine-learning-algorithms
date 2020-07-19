import pandas as pd
import numpy as np
import sklearn as sl
from matplotlib import pyplot as plt
from numpy.linalg import inv

teta = np.zeros((6,1), dtype=float)

def fit(x, y):
    #Dopuni X: u prvoj koloni sve 1 - 5 prediktora i 1 na pocetku (zbog teta0) za svaki primer
    x = np.pad(x, ((0,0),(1,0)), mode='constant', constant_values=1)
    #trazenje optimalnih koeficijenata
    # teta = (x.T * x)^-1 * x.T * y
    # e = (x.T * x)^-1
    e = np.linalg.inv(np.matmul(x.T, x))
    f = np.matmul(e, x.T)
    teta = np.matmul(f,y)
    return teta

def predict(x_test):
    #Dopuni X: u prvoj koloni sve 1 - 5 prediktora i 1 na pocetku (zbog teta0) za svaki primer
    x_test = np.pad(x_test, ((0,0),(1,0)), mode='constant', constant_values=1)
    #predict y
    y_pred = np.matmul(x_test, teta)
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
#podela obucavajuceg skupa 80/20 za validaciju sa izostavljanjem
x_train = x_stand[69:,:]
x_valid = x_stand[:69,:]
y_train = Y[69:]
y_valid = Y[:69]

teta = fit(x_train, y_train)
#predict y
#y_pred = np.matmul(x_valid, teta)
y_pred = predict(x_valid)

#Nadji srednju kvadratnu gresku
J = (1/2) * np.sum(np.square(np.subtract(y_pred, y_valid)))
print('Gubitak na validacionom skupu ', J)

teta = fit(x_stand, Y)
#predict y
#y_pred = np.matmul(x_valid, teta)
y_pred = predict(x_stand)
#Nadji srednju kvadratnu gresku
J = (1/2) * np.sum(np.square(np.subtract(y_pred, Y)))
print('Gubitak na obucavajucem skupu ', J)







