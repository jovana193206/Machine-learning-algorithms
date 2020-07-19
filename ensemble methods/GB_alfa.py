import pandas as pd
import numpy as np
import sklearn as sl
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# ispitujemo learning rate - alfa


def standardize(x):
    #standardizacija prediktora
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_stand = np.subtract(x, x_mean)
    x_stand = np.divide(x_stand, x_std)
    return x_stand

#Procitaj podatke iz csv fajla
df = pd.read_csv('data.csv', header=None)
#promesaj podatke
df = df.sample(frac=1).reset_index(drop=True)
dataset = df.values
#podela dataset-a na trening i test skup u odnosu 80/20
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
valid = df[~msk]
train = train.values
valid = valid.values

#Izdvajanje prediktora - prvih 6 kolona dataset-a, standardizacija i prosirenje sa kolonom 1
x_train = train[:,:6]
x_train = standardize(x_train)
x_valid = valid[:,:6]
x_valid = standardize(x_valid)
#Izdvanjanje oznaka - poslednja kolona dataset-a
y_train = train[:,6]
y_valid = valid[:,6]

# alfa = 0.1
train_accuracy1 = np.ones(10)
valid_accuracy1 = np.ones(10)
ans_size = np.arange(1, 100, 10)
# alfa = 0.3
train_accuracy3 = np.ones(10)
valid_accuracy3 = np.ones(10)
# alfa = 0.7
train_accuracy7 = np.ones(10)
valid_accuracy7 = np.ones(10)
for i in range(len(ans_size)):
    s = ans_size[i]
    model1 = GradientBoostingClassifier(n_estimators=s, learning_rate=0.1)
    model1.fit(x_train, y_train)
    valid_accuracy1[i] = model1.score(x_valid, y_valid)
    train_accuracy1[i]= model1.score(x_train, y_train)
    model3 = GradientBoostingClassifier(n_estimators=s, learning_rate=0.3)
    model3.fit(x_train, y_train)
    valid_accuracy3[i] = model3.score(x_valid, y_valid)
    train_accuracy3[i]= model3.score(x_train, y_train)
    model7 = GradientBoostingClassifier(n_estimators=s, learning_rate=0.7)
    model7.fit(x_train, y_train)
    valid_accuracy7[i] = model7.score(x_valid, y_valid)
    train_accuracy7[i]= model7.score(x_train, y_train)
    
plt.title("Tacnost na trening skupu")
plt.xlabel("broj stabala")
plt.ylabel("Tacnost")
plt.plot(ans_size, train_accuracy1, color="navy", label='alfa = 0.1')
plt.plot(ans_size, train_accuracy3, color="red", label='alfa = 0.3')
plt.plot(ans_size, train_accuracy7, color='green', label='alfa = 0.7')
plt.legend()
plt.show()

plt.title("Tacnost na validacionom skupu")
plt.xlabel("broj stabala")
plt.ylabel("Tacnost")
plt.plot(ans_size, valid_accuracy1, color="navy", label='alfa = 0.1')
plt.plot(ans_size, valid_accuracy3, color="red", label='alfa = 0.3')
plt.plot(ans_size, valid_accuracy7, color='green', label='alfa = 0.7')
plt.legend()
plt.show()