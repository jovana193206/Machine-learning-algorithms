import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxopt
from matplotlib import cm
from mpl_toolkits import mplot3d


def standardize(x):
    #standardizacija prediktora
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_stand = np.subtract(x, x_mean)
    x_stand = np.divide(x_stand, x_std)
    return x_stand

# neka klase umesto sa 0 i 1 budu obelezene sa -1 i 1
def alterLabels(y):
    for i in range(len(y)):
        if(y[i] == 0):
            y[i] = -1
    return y

def confMat(y_pred, y):
    conf = np.zeros((2, 2))
    for i in range(len(y)):
        if(y[i] == 1):
            if(y_pred[i] == 1):
                conf[1,1] = conf[1,1] + 1
            else:
                conf[1,0] = conf[1,0] + 1
        else:
            if(y_pred[i] == 1):
                conf[0,1] = conf[0,1] + 1
            else:
                conf[0,0] = conf[0,0] + 1
    return conf

# Preciznost, kao argumente prima matricu konfuzije i klasu za koju se racuna: {-1, 1}
def precision(conf, c):
    if(c == 1):
        # true_pos / (true_pos + false_pos)
        p = conf[1,1] / (conf[1,1] + conf[0,1])
    elif(c == -1):
        # true_neg / (true_neg + false_neg)
        p = conf[0,0] / (conf[0,0] + conf[1,0])
    return p

# Osetljivost, kao argumente prima matricu konfuzije i klasu za koju se racuna: {-1, 1}
def recall(conf, c):
    if(c == 1):
        # true_pos / (true_pos + false_neg)
        o = conf[1,1] / (conf[1,1] + conf[1,0])
    elif(c == -1):
        # true_neg / (true_neg + false_pos)
        o = conf[0,0] / (conf[0,0] + conf[0,1])
    return o

def f1_score(prec, rec):
    # 2 / ((prec)^-1 + (recall)^-1)
    f1 = 2 / ((1/prec) + (1/rec))
    return f1

def accuracy(y_pred, y_true):
    num_correct = 0
    for i in range(len(y_pred)):
        if(y_pred[i] == y_true[i]):
            num_correct = num_correct + 1
    a = (num_correct/len(y_true)) * 100
    return a

#Gauss-ovski kernel
def K(x_i, x_j, sigma):
    beta = 1/(2*sigma*sigma)
    norm = (x_i-x_j).T.dot(x_i-x_j)
    kern = np.exp(-beta * norm)
    return kern

def predict(x_test, w, b):
    y_pred = np.sign(np.matmul(x_test, w) + b)
    return y_pred

def QP(x, y, C, sigma):
    m, n = x.shape
    #prevodimo dualni problem u oblik pogodan za kvadratno programiranje:
    #min (1/2)x.T*P*x + q.T*x
    #subject to G*x <= h and A*x = b
    #Kreiramo kernel matricu: Km[i,j] = K(x_i, x_j)
    Km = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            Km[i,j] = float(K(x[i,:], x[j,:], sigma))
    # y_i*y_j
    y= np.reshape(y, (m,1))
    yy = y.dot(y.T)
    #promenljiva koju optimizujemo ima oblik [alfa].T
    #matrica P mnozi alfe sa yy*Km
    P = yy * Km
    #q nam daje -sum(alfa_i)
    q = -np.ones((m,1))
    #uslov C >= alfa_i >= 0 transformisemo u alfa_i <= C i -alfa_i <= 0 za svako i
    G = np.zeros((2*m, m))
    #ovaj deo obezbedjuje alfa_i <= C
    G[:m,:] = np.eye(m)
    #ovaj deo obezbedjuje -alfa_i <= 0
    G[m:,:] = -np.eye(m)
    #uslov -alfa_i <= 0
    h = np.zeros((2*m,1))
    #uslov alfa_i <= C
    h[:m] = C
    #uslov sum(alfa_i * y_i) = 0
    y = np.reshape(y, (m,1))
    A = y.T
    b = np.array([[0]])
    #convert everything to cxvopt matrices
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    A = cvxopt.matrix(A, A.shape,'d')
    b = cvxopt.matrix(b, b.shape,'d')
    #set up cvxopt
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alfa = np.array(sol['x'])
    w = np.sum([alfa[i] * y[i] * x[i,:] for i in range(m)], axis=0)
    i = 0
    for i in range(len(alfa)):
        if((alfa[i] > 0) and (alfa[i] < C)):
            break
    b = y[i] - sum([alfa[j] * y[j] * K(x[j,:], x[i,:], sigma) for j in range(len(alfa))])
    return w, b


#Procitaj podatke iz csv fajla
df = pd.read_csv('data.csv', header=None)
#promesaj podatke
df = df.sample(frac=1).reset_index(drop=True)
dataset = df.values
#podela dataset-a na trening i test skup u odnosu 80/20
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train = train.values
valid = test.values

#Izdvajanje prediktora - prvih 5 kolona dataset-a, standardizacija i prosirenje sa kolonom 1
x_train = train[:,:6]
x_train = standardize(x_train)
x_valid = valid[:,:6]
x_valid = standardize(x_valid)
#Izdvanjanje oznaka - poslednja kolona dataset-a
y_train = alterLabels(train[:,6])
y_valid = alterLabels(valid[:,6])

Cs = np.arange(0, 4, 0.25)
sigmas = np.arange(1, 17)
accuracy_arr = np.zeros((16,16))
#Za svaki stepen iz degrees istreniraj teta i izracunaj J na validacionom skupu
for i in range(len(sigmas)):
    sigma = sigmas[i]
    for j in range(len(Cs)):
        C = Cs[j]
        #print('C = ', C)
        #istreniraj model - nadji w i b
        w, b = QP(x_train, y_train, C, sigma)
        y_pred = predict(x_valid, w, b)
        #nadji tacnost na validacionom skupu
        acc =  accuracy(y_pred, y_valid)
        conf = confMat(y_pred, y_valid)
        print('acc = ', acc)
        accuracy_arr[i,j] = acc

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(sigmas, Cs, accuracy_arr, 50, cmap=cm.coolwarm)
ax.set_xlabel('sigma')
ax.set_ylabel('C')
ax.set_zlabel('Tacnost [%]')

ind = np.argmax(accuracy_arr)
print('Optimalan parametar C: ', Cs[ind // 16])
print('Optimalno sigma: ', sigmas[ind % 16])


