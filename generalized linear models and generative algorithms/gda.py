import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv

def standardize(x):
    #standardizacija prediktora
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_stand = np.subtract(x, x_mean)
    x_stand = np.divide(x_stand, x_std)
    return x_stand

#Vraca vektor koji za svaki element vektora y daje indikatorsku funkciju od t
def indicatorFunc(y, t):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i] == t):
            result[i] = 1
        else:
            result[i] = 0
    return result    

def predict(x, mi, cov, fi):
    n = x.shape[1]
    y_pred = np.zeros(len(x))
    pi_part = np.power((2*np.pi), n/2)
    for i in range(len(x)):
        pr = np.zeros(3)
        cov_det = np.linalg.det(cov[0])
        exp_part = (-1/2) * np.matmul(np.matmul(x[i] - mi[0], np.linalg.inv(cov[0])), (x[i] - mi[0]).T)
        pr[0] = (fi[0]/(pi_part * np.power(cov_det, 1/2))) * np.exp(exp_part)
        cov_det = np.linalg.det(cov[1])
        exp_part = (-1/2) * np.matmul(np.matmul(x[i] - mi[1], np.linalg.inv(cov[1])), (x[i] - mi[1]).T)
        pr[1] = (fi[1]/(pi_part * np.power(cov_det, 1/2))) * np.exp(exp_part)
        cov_det = np.linalg.det(cov[2])
        exp_part = (-1/2) * np.matmul(np.matmul(x[i] - mi[2], np.linalg.inv(cov[2])), (x[i] - mi[2]).T)
        pr[2] = (fi[2]/(pi_part * np.power(cov_det, 1/2))) * np.exp(exp_part)
        y_pred[i] = np.argmax(pr)
    return y_pred


#Procitaj podatke iz csv fajla
df = pd.read_csv('multiclass_data.csv', header=None)
#promesaj podatke
df = df.sample(frac=1).reset_index(drop=True)
dataset = df.values
#podela dataset-a na trening i test skup u odnosu 80/20
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train = train.values
test = test.values

#trazimo u kojim redovima dataseta se nalaze podaci odredjenih klasa
train0_ind = np.where(train.T[5] == 0)
train1_ind = np.where(train.T[5] == 1)
train2_ind = np.where(train.T[5] == 2)

#Izdvajanje prediktora - prvih 5 kolona dataset-a, standardizacija i prosirenje sa kolonom 1
x_train = train[:,:5]
x_train = standardize(x_train)
x_test = test[:,:5]
x_test = standardize(x_test)
#Izdvanjanje oznaka - poslednja kolona dataset-a
y_train = train[:,5]
y_test = test[:,5]
#razdvajamo x_train po klasama
x_train0 = (x_train[train0_ind,:]).reshape(-1,5)
x_train1 = (x_train[train1_ind,:]).reshape(-1,5)
x_train2 = (x_train[train2_ind,:]).reshape(-1,5)

#trazimo parametre mi za svaku klasu podataka x
mi = np.mean(x_train0, axis=0).reshape(1,-1)
mi = np.concatenate((mi, np.mean(x_train1, axis=0).reshape(1,-1)))
mi = np.concatenate((mi, np.mean(x_train2, axis=0).reshape(1,-1)))

#kovarijacione matrice za svaku klase
n = x_train.shape[1]
cov = np.zeros((3, n, n))
cov[0] = (1/len(x_train0)) * np.dot((x_train0 - mi[0]).T,(x_train0 - mi[0]))
cov[1] = (1/len(x_train1)) * np.dot((x_train1 - mi[1]).T,(x_train1 - mi[1]))
cov[2] = (1/len(x_train2)) * np.dot((x_train2 - mi[2]).T,(x_train2 - mi[2]))

#pronalazenje parametara fi
fi = np.zeros(3)
fi[0] = (1/len(x_train)) * np.sum(indicatorFunc(y_train, 0))
fi[1] = (1/len(x_train)) * np.sum(indicatorFunc(y_train, 1))
fi[2] = (1/len(x_train)) * np.sum(indicatorFunc(y_train, 2))

#Testiranje klasifikatora na trening skupu
y_predicted = predict(x_train, mi, cov, fi)
#print(y_predicted)
correct = np.array(np.where(y_train == y_predicted)).ravel()
num_correct = len(correct)
print('Broj pogodjenih: ', num_correct)
print('Broj promasenih: ', len(y_train) - num_correct)
accuracy = (num_correct/len(y_train)) * 100
print('Tacnost na trening skupu: ', accuracy, '%')

#Testiranje klasifikatora na test skupu
y_predicted = predict(x_test, mi, cov, fi)
correct = np.array(np.where(y_test == y_predicted)).ravel()
num_correct = len(correct)
print('Broj pogodjenih: ', num_correct)
print('Broj promasenih: ', len(y_test) - num_correct)
accuracy = (num_correct/len(y_test)) * 100
print('Tacnost na test skupu: ', accuracy, '%')







