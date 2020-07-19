import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from itertools import combinations_with_replacement

#Optimalna vrednost za parametar ridge regularizacije lamba
l = 14.737105263157895


def polyFeatures(x, degree):
    if(degree == 1):
        #obicna linearna regresija, ne menja se x
        return x
    if(degree == 2):
        #za svaki od m primera iz skupa x pravimo novi primer koji osim osnovnih 5 prediktora
        #sadrzi i sve njihove kombinacije reda 2
        #kao rezultat dobijamo novo x sa 20 prediktora po primeru
        x2 = np.matrix(list(combinations_with_replacement(x[1],2)))
        x2 = np.prod(x2, axis=1)
        x2 = x2.ravel()
        for i in range(len(x)):
            if(i != 1):
                x_con = np.matrix(list(combinations_with_replacement(x[i],2)))
                x_con = np.prod(x_con, axis=1)
                x_con = x_con.ravel() 
                x2 = np.concatenate((x2,x_con))
        x_poly = np.concatenate((x, x2), axis = 1)
        return x_poly
    #nalazimo x reda za 1 manjeg od zadatog i na njega dodajemo sve kombinacije 
    #prediktora iz x zadatog reda - degree
    x_prev = polyFeatures(x, degree - 1)
    x_degree = np.matrix(list(combinations_with_replacement(x[1],degree)))
    x_degree = np.prod(x_degree, axis=1)
    x_degree = x_degree.ravel()
    for i in range(len(x)):
        if(i != 1):
            x_con = np.matrix(list(combinations_with_replacement(x[i],degree)))
            x_con = np.prod(x_con, axis=1)
            x_con = x_con.ravel() 
            x_degree = np.concatenate((x_degree,x_con))
    x_poly = np.concatenate((x_prev, x_degree), axis = 1)
    return x_poly

def fit(x, y):
    #Dopuni X: u prvoj koloni sve 1 - 5 prediktora i 1 na pocetku (zbog teta0) za svaki primer
    x = np.pad(x, ((0,0),(1,0)), mode='constant', constant_values=1)
    #trazenje optimalnih koeficijenata
    #l = lambda
    # teta = (x.T*x + lambda*I)^-1 * x.T * y
    # e = (x.T*x + lambda*I)^-1
    e = np.matmul(x.T,x)
    e = np.add(e, l*np.identity(21))
    e = np.linalg.inv(e)
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

#podela dataset-a na trening i validacioni
x_train = x_stand[69:,:]
x_valid = x_stand[:69,:]
y_train = Y[69:]
y_valid = Y[:69]

#Optimalna vrednost za stepen polinoma je 2
x_poly = polyFeatures(x_stand, 2)
teta = fit(x_poly, Y)
print('teta: ')
print(teta)
print()
y_pred = predict(x_poly)
J = (1/2) * np.sum(np.square(np.subtract(y_pred, Y)))
print('J na obucavajucem skupu = ', J)

x_poly = polyFeatures(x_train, 2)
teta = fit(x_poly, y_train)
x_poly = polyFeatures(x_valid, 2)
y_pred = predict(x_poly)
J = (1/2) * np.sum(np.square(np.subtract(y_pred, y_valid)))
print('J na validacionom skupu = ', J)


