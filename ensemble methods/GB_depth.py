import pandas as pd
import numpy as np
import sklearn as sl
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# ispitujemo dubinu stabla


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

# depth = 2
train_accuracy2 = np.ones(10)
valid_accuracy2 = np.ones(10)
ans_size = np.arange(1, 100, 10)
# depth = 4
train_accuracy4 = np.ones(10)
valid_accuracy4 = np.ones(10)
# depth = 5
train_accuracy5 = np.ones(10)
valid_accuracy5 = np.ones(10)
for i in range(len(ans_size)):
    s = ans_size[i]
    model2 = GradientBoostingClassifier(n_estimators=s, max_depth=2)
    model2.fit(x_train, y_train)
    valid_accuracy2[i] = model2.score(x_valid, y_valid)
    train_accuracy2[i]= model2.score(x_train, y_train)
    model4 = GradientBoostingClassifier(n_estimators=s, max_depth=4)
    model4.fit(x_train, y_train)
    valid_accuracy4[i] = model4.score(x_valid, y_valid)
    train_accuracy4[i]= model4.score(x_train, y_train)
    model5 = GradientBoostingClassifier(n_estimators=s, max_depth=5)
    model5.fit(x_train, y_train)
    valid_accuracy5[i] = model5.score(x_valid, y_valid)
    train_accuracy5[i]= model5.score(x_train, y_train)
    
plt.title("Tacnost na trening skupu")
plt.xlabel("broj stabala")
plt.ylabel("Tacnost")
plt.plot(ans_size, train_accuracy2, color="navy", label='d = 2')
plt.plot(ans_size, train_accuracy4, color="red", label='d = 4')
plt.plot(ans_size, train_accuracy5, color='green', label='d = 5')
plt.legend()
plt.show()

plt.title("Tacnost na validacionom skupu")
plt.xlabel("broj stabala")
plt.ylabel("Tacnost")
plt.plot(ans_size, valid_accuracy2, color="navy", label='d = 2')
plt.plot(ans_size, valid_accuracy4, color="red", label='d = 4')
plt.plot(ans_size, valid_accuracy5, color='green', label='d = 5')
plt.legend()
plt.show()