import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from itertools import combinations_with_replacement

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


def predict(X):
    #koeficijenti izracunati u polinomijalnom modelu stepena 2 sa ridge regresijom
    #i parametrom labda = 14.737105263157895
    #pomenuti model se nalazi u prilozenom fajlu polyRidgeRegularizacija_stepen2.py
    teta = np.array([129.69100491, 27.44933437, -10.24039469, 7.56578779, 26.74393517,
                     5.54774517, 5.97199108, -4.32042655, -1.76561106, 4.22765287,
                     -1.11024745, 2.80935916, -6.45800792, 9.63547487, -2.17489747,
                     2.80731247, -2.9391896, 4.24224039, 0.47662749, -5.76104341,
                     8.44905651])
    #standardizacija prediktora
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    x_stand = np.subtract(X, x_mean)
    x_stand = np.divide(x_stand, x_std)
    x_poly = polyFeatures(x_stand, 2)
    #Dopuni x_poly: u prvoj koloni sve 1 - 5 prediktora i 1 na pocetku (zbog teta0) za svaki primer
    x_test = np.pad(x_poly, ((0,0),(1,0)), mode='constant', constant_values=1)
    #predict y
    y_pred = np.matmul(x_test, teta)
    return y_pred




