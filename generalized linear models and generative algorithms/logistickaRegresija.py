import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def standardize(x):
    #standardizacija prediktora
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_stand = np.subtract(x, x_mean)
    x_stand = np.divide(x_stand, x_std)
    return x_stand

def logisticFunc(x):
    e = np.exp(-x)
    divisor = (np.ones(len(x)) + e)
    for i in range(len(divisor)):
        if(divisor[i] == 1):
            divisor[i] = 1.00000001
    h = np.divide(np.ones(len(x)), divisor)
    return h

def predict(teta, x):
    z = np.matmul(x, teta)
    return logisticFunc(z)

def loss(teta, x, y):
    m = len(x)
    y_pred = predict(teta, x)
    correct = np.multiply(y, np.log(y_pred))
    wrong = np.multiply(np.subtract(np.ones(m), y), np.log(np.subtract(np.ones(m), y_pred)))
    l = -np.sum(correct + wrong, axis=0)
    return l

def gradientDescent(x, y, teta, alfa = 0.1, iterations = 500):
    m = len(x)
    #medjurezultati za iscrtavanje grafika
    cost_history = np.zeros(iterations)
    teta_history = np.zeros((iterations, 6))
    iter_array = np.zeros(iterations)
    for it in range(iterations):
        y_pred = predict(teta, x)
        teta = teta - (1/m)*alfa*(x.T.dot((y_pred - y)))
        teta_history[it,:] = teta.T
        cost_history[it] = loss(teta, x, y)
        iter_array[it] = it
    return teta, teta_history, cost_history, iter_array

def createTrainForClass(y, myClass):
    y_tr = np.zeros(y.shape)
    for i in range(len(y)):
        if(y[i] == myClass):
            y_tr[i] = 1
    return y_tr

    
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

#Izdvajanje prediktora - prvih 5 kolona dataset-a, standardizacija i prosirenje sa kolonom 1
x_train = train[:,:5]
x_train = standardize(x_train)
x_train = np.pad(x_train, ((0,0),(1,0)), mode='constant', constant_values=1)
x_test = test[:,:5]
x_test = standardize(x_test)
x_test = np.pad(x_test, ((0,0),(1,0)), mode='constant', constant_values=1)
#Izdvanjanje oznaka - poslednja kolona dataset-a
y_train = train[:,5]
y_test = test[:,5]

#Klasifikator za klasu 0
y_train0 = createTrainForClass(y_train, 0)
teta_zeros = np.zeros(6,)
teta0, teta_history, cost_history, iterations = gradientDescent(x_train, y_train0, teta_zeros)
teta_alfaLarger, teta_history, cost_alfaLarger, iterations = gradientDescent(x_train, y_train0, teta_zeros, 70)
teta_alfaLower, teta_history, cost_alfaLower, iterations = gradientDescent(x_train, y_train0, teta_zeros, 0.01)
plt.title("Funkcija gubitka kroz iteracije")
plt.xlabel("Iteracija")
plt.ylabel("Gubitak")
plt.plot(iterations, cost_history, label="alfa = 0.1")
plt.plot(iterations, cost_alfaLarger, label="alfa = 70")
plt.plot(iterations, cost_alfaLower, label="alfa = 0.001")
plt.legend()
plt.show()

#Klasifikator za klasu 1
y_train1 = createTrainForClass(y_train, 1)
teta_zeros = np.zeros(6,)
teta1, teta_history, cost_history, iterations = gradientDescent(x_train, y_train1, teta_zeros)
plt.title("Funkcija gubitka kroz iteracije za klasifikator1")
plt.xlabel("Iteracija")
plt.ylabel("Gubitak")
plt.plot(iterations, cost_history)
plt.show()

#Klasifikator za klasu 2
y_train2 = createTrainForClass(y_train, 2)
teta_zeros = np.zeros(6,)
teta2, teta_history, cost_history, iterations = gradientDescent(x_train, y_train2, teta_zeros)
plt.title("Funkcija gubitka kroz iteracije za klasifikator1")
plt.xlabel("Iteracija")
plt.ylabel("Gubitak")
plt.plot(iterations, cost_history)
plt.show()

#Testiranje klasifikatora na trening skupu
y_predicted = np.zeros((len(y_train),3))
y_predicted[:,0] = predict(teta0, x_train)
y_predicted[:,1] = predict(teta1, x_train)
y_predicted[:,2] = predict(teta2, x_train)
y_predicted = np.argmax(y_predicted, axis=1)
correct = np.array(np.where(y_train == y_predicted)).ravel()
num_correct = len(correct)
print('Broj pogodjenih: ', num_correct)
print('Broj promasenih: ', len(y_train) - num_correct)
accuracy = (num_correct/len(y_train)) * 100
print('Tacnost na trening skupu: ', accuracy, '%')

#Testiranje klasifikatora na test skupu
y_predicted = np.zeros((len(y_test),3))
y_predicted[:,0] = predict(teta0, x_test)
y_predicted[:,1] = predict(teta1, x_test)
y_predicted[:,2] = predict(teta2, x_test)
y_predicted = np.argmax(y_predicted, axis=1)
correct = np.array(np.where(y_test == y_predicted)).ravel()
print(correct)
num_correct = len(correct)
print('Broj pogodjenih: ', num_correct)
print('Broj promasenih: ', len(y_test) - num_correct)
accuracy = (num_correct/len(y_test)) * 100
print('Tacnost na test skupu: ', accuracy, '%')
 





