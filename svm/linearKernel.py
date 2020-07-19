import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxopt

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

def QP(x, y, C):
    m, n = x.shape
    #prevodimo primarni problem u oblik pogodan za kvadratno programiranje:
    #min (1/2)x.T*P*x + q.T*x
    #subject to G*x <= h and A*x = b
    #promenljiva koju optimizujemo ima oblik [w, b, eps].T
    #matrica P izdvaja samo w deo iz promenljive za optimizaciju, da bismo dobili 1/2 ||w||^2
    P = np.zeros((m+n+1,m+n+1))
    for i in range(n):
        P[i,i] = 1
    #q izdvaja samo eps deo promenljive za optimizaciju da bismo dobili C * suma(eps_i)
    q = np.vstack([np.zeros((n+1,1)), C * np.ones((m,1))])
    #uslov primarnog problema transformisemo u -y_i(w.T*x_i + b)-eps_i <= -1 and -eps_i <= 0 za svako i
    G = np.zeros((2*m, m+n+1))
    #ovaj deo obezbedjuje y_i*w.T*x_i u uslovu -y_i(w.T*x_i + b)-eps_i <= -1
    y = np.reshape(y, (len(y), 1))
    G[:m, 0:n] = x*y
    #ovaj deo obezbedjuje y_i*b u uslovu -y_i(w.T*x_i + b)-eps_i <= -1
    G[:m, n] = y.T
    #ovaj deo obezbedjuje +eps_i u uslovu -y_i(w.T*x_i + b)-eps_i <= -1
    G[:m, n+1:]  = np.eye(m)
    #ovaj deo obezbedjuje eps_i >= 0
    G[m:, n+1:] = np.eye(m)
    G = -G
    #uslov za eps je <= od 0
    h = np.zeros((2*m,1))
    #uslov po w,b je <= -1
    h[:m] = -1
    #convert everything to cxvopt matrices
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    #set up cvxopt
    sol = cvxopt.solvers.qp(P, q, G, h)
    w = np.array(sol['x'][:n])
    b = sol['x'][n]
    eps = sol['x'][n+1:]
    print('w: ', w.T)
    print('b: ', b)
    print('eps: ', eps.T)
    return w, b

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
     

def predict(x_test, w, b):
    y_pred = np.sign(np.matmul(x_test, w) + b)
    return y_pred


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
test = test.values

#Izdvajanje prediktora - prvih 5 kolona dataset-a, standardizacija i prosirenje sa kolonom 1
x_train = train[:,:6]
x_train = standardize(x_train)
x_test = test[:,:6]
x_test = standardize(x_test)
#Izdvanjanje oznaka - poslednja kolona dataset-a
y_train = alterLabels(train[:,6])
y_test = alterLabels(test[:,6])

#optimalno C
C = 0.25
#treniranje modela
w, b = QP(x_train, y_train, C)
#testiranje modela
y_pred = predict(x_test, w, b)
conf = confMat(y_pred, y_test)
print('Matrica konfuzije: ')
print(conf)
print()
p0 = precision(conf, -1)
r0 = recall(conf, -1)
print('Preciznost za klasu 0: ', p0*100, '%')
print('Osetljivost za klasu 0: ', r0*100, '%')
print('F1 score za klasu 0: ', f1_score(p0, r0))
print()
p1 = precision(conf, 1)
r1 = recall(conf, 1)
print('Preciznost za klasu 1: ', p1*100, '%')
print('Osetljivost za klasu 1: ', r1*100, '%')
print('F1 score za klasu 1: ', f1_score(p1, r1))
acc = accuracy(y_pred, y_test)
print()
print('Ukupna tacnost: ', acc, '%')


