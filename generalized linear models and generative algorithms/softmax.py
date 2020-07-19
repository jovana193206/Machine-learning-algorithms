import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

k = 3

def standardize(x):
    #standardizacija prediktora
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_stand = np.subtract(x, x_mean)
    x_stand = np.divide(x_stand, x_std)
    return x_stand

def loss(teta, x, y):
    m = len(x)
    totalsum = 0
    for i in range(m):
        y_i = int(y[i])
        tetaxi = np.matmul(x[i], teta[:,y_i])
        lnsum = np.log(np.sum(np.exp(np.matmul(x[i], teta))))
        totalsum = totalsum + (tetaxi - lnsum)
    return -totalsum

#Vraca vektor koji za svaki element vektora y daje indikatorsku funkciju od t
def indicatorFunc(y, t):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i] == t):
            result[i] = 1
        else:
            result[i] = 0
    return result    

def create_mini_batches(x, y, batch_size): 
    mini_batches = [] 
    y = np.reshape(y, (len(y),1))
    data = np.hstack((x, y)) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0;
    for i in range(n_minibatches): 
        mini_batch = data[i * batch_size : (i + 1) * batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if ((data.shape[0] % batch_size) != 0): 
        mini_batch = data[(i + 1) * batch_size : data.shape[0], :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches 

def gradient(x_mini, y_mini, teta):
    #teta ima (n+1) red i k kolona
    k = teta.shape[1]
    #batch size - bs = len(x_mini)
    bs = len(x_mini)
    divisor_sum = np.sum(np.exp(np.matmul(x_mini, teta)), axis=1)
    #za l = 0
    dividend = np.exp(np.matmul(x_mini, teta[:,0]))
    z = indicatorFunc(y_mini, 0) - np.divide(dividend, divisor_sum)
    grad = (1/bs) * (x_mini.T.dot(z))
    grad = np.reshape(grad, (len(grad), 1))
    #za l = 1,..,k-2, za l=k-1 odnosno teta_k se ne radi azuriranje, uvek je 0
    for i in range(k - 1):
        if (i == 0):
            continue
        dividend = np.exp(np.matmul(x_mini, teta[:,i]))
        z = indicatorFunc(y_mini, i) - np.divide(dividend, divisor_sum)
        grad_i = (1/bs) * (x_mini.T.dot(z))
        grad_i = np.reshape(grad_i, (len(grad_i), 1))
        grad = np.concatenate((grad, grad_i), axis = 1)
    teta_k = np.zeros((len(grad), 1))
    grad = np.concatenate((grad, teta_k), axis=1)
    return grad
        

def gradientDescent(x, y, teta, batch_size = 16, alfa = 0.1, iterations = 100):
    #medjurezultati za iscrtavanje grafika
    cost_history = np.zeros(iterations)
    iter_array = np.zeros(iterations)
    mini_batches = create_mini_batches(x, y, batch_size)
    for it in range(iterations):
        for mini_batch in mini_batches:
            #print('mini batch: ')
            #print(mini_batch)
            x_mini, y_mini = mini_batch 
            teta = teta + alfa*gradient(x_mini, y_mini, teta)
        cost_history[it] = loss(teta, x, y)
        iter_array[it] = it
    return teta, cost_history, iter_array

#softmax
def predict(teta, x):
    dividend = np.exp(np.matmul(x, teta))
    divisor = np.sum(dividend, axis=1)
    for j in range(len(dividend)):
        dividend[j] = np.divide(dividend[j], divisor[j])
    ET = dividend
    result = np.argmax(ET, axis=1)
    return result

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

#Treniranje
teta_zeros = np.zeros((6,k))
teta, cost_history, iterations = gradientDescent(x_train, y_train, teta_zeros)
teta_bsLarger, cost_bsLarger, iterations = gradientDescent(x_train, y_train, teta_zeros, 128)
teta_bsLower, cost_bsLower, iterations = gradientDescent(x_train, y_train, teta_zeros, 4)
plt.title("Funkcija gubitka kroz iteracije")
plt.xlabel("Iteracija")
plt.ylabel("Gubitak")
plt.plot(iterations, cost_history, label="bs = 16")
plt.plot(iterations, cost_bsLarger, label="bs = 128")
plt.plot(iterations, cost_bsLower, label="bs = 4")
plt.legend()
plt.show()

#Testiranje klasifikatora na trening skupu
y_predicted = predict(teta, x_train)
#print(y_predicted)
correct = np.array(np.where(y_train == y_predicted)).ravel()
num_correct = len(correct)
print('Broj pogodjenih: ', num_correct)
print('Broj promasenih: ', len(y_train) - num_correct)
accuracy = (num_correct/len(y_train)) * 100
print('Tacnost na trening skupu: ', accuracy, '%')

#Testiranje klasifikatora na test skupu
y_predicted = predict(teta, x_test)
correct = np.array(np.where(y_test == y_predicted)).ravel()
num_correct = len(correct)
print('Broj pogodjenih: ', num_correct)
print('Broj promasenih: ', len(y_test) - num_correct)
accuracy = (num_correct/len(y_test)) * 100
print('Tacnost na test skupu: ', accuracy, '%')


