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
#Izdvanjanje oznaka - poslednja kolona dataset-a
Y = dataset[:,5]

teta = fit(X, Y)
print('fit teta: ')
print(teta)

#predict y
#y_pred = np.matmul(x_valid, teta)
y_pred = predict(X)

print('y: ')
print(Y)
print()
print('y_pred: ')
print(y_pred)

#Nadji srednju kvadratnu gresku
J = (1/2) * np.sum(np.square(np.subtract(y_pred, Y)))

print('J = ', J)


