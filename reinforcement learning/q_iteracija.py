import numpy as np
from matplotlib import pyplot as plt

# Ns=11 stanja - od 0 do 10
Ns = 11
# Na=4 akcije: levo=0, desno=1, gore=2, dole=3
Na = 4
# matrica koja cuva Q vrednosti - NsxNa, inicijalno sve 0
Q = np.zeros((Ns, Na))
# vektor koji cuva nagradu svakog stanja R
R = np.full((Ns), -0.04)
R[9] = -1
R[10] = 1
# Politika - za svako stanje daje optimalnu akciju koju treba u tom stanju preduzeti
policy = np.full(Ns, -1)
# matrica P koja cuva verovatnoce prelaza izmedju stanja - NsxNaxNs
# P[s, a, s'] = verovatnoca da se iz stanja s pod dejstvom akcije a predje u stanje s'
P = np.zeros((Ns, Na, Ns))
# za s=0
P[0,0,:] = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
P[0,1,:] = np.array([0.1, 0.1, 0, 0.8, 0, 0, 0, 0, 0, 0, 0])
P[0,2,:] = np.array([0.1, 0.8, 0, 0.1, 0, 0, 0, 0, 0, 0, 0])
P[0,3,:] = np.array([0.9, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0])
# za s=1
P[1,0,:] = np.array([0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])
P[1,1,:] = np.array([0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])
P[1,2,:] = np.array([0, 0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0])
P[1,3,:] = np.array([0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# za s=2
P[2,0,:] = np.array([0, 0.1, 0.9, 0, 0, 0, 0, 0, 0, 0, 0])
P[2,1,:] = np.array([0, 0.1, 0.1, 0, 0.8, 0, 0, 0, 0, 0, 0])
P[2,2,:] = np.array([0, 0, 0.9, 0, 0.1, 0, 0, 0, 0, 0, 0])
P[2,3,:] = np.array([0, 0.8, 0.1, 0, 0.1, 0, 0, 0, 0, 0, 0])
# za s=3
P[3,0,:] = np.array([0.8, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0])
P[3,1,:] = np.array([0, 0, 0, 0.2, 0, 0.8, 0, 0, 0, 0, 0])
P[3,2,:] = np.array([0.1, 0, 0, 0.8, 0, 0.1, 0, 0, 0, 0, 0])
P[3,3,:] = np.array([0.1, 0, 0, 0.8, 0, 0.1, 0, 0, 0, 0, 0])
# za s=4
P[4,0,:] = np.array([0, 0, 0.8, 0, 0.2, 0, 0, 0, 0, 0, 0])
P[4,1,:] = np.array([0, 0, 0, 0, 0.2, 0, 0, 0.8, 0, 0, 0])
P[4,2,:] = np.array([0, 0, 0.1, 0, 0.8, 0, 0, 0.1, 0, 0, 0])
P[4,3,:] = np.array([0, 0, 0.1, 0, 0.8, 0, 0, 0.1, 0, 0, 0])
# za s=5
P[5,0,:] = np.array([0, 0, 0, 0.8, 0, 0.1, 0.1, 0, 0, 0, 0])
P[5,1,:] = np.array([0, 0, 0, 0, 0, 0.1, 0.1, 0, 0.8, 0, 0])
P[5,2,:] = np.array([0, 0, 0, 0.1, 0, 0, 0.8, 0, 0.1, 0, 0])
P[5,3,:] = np.array([0, 0, 0, 0.1, 0, 0.8, 0, 0, 0.1, 0, 0])
# za s=6
P[6,0,:] = np.array([0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0])
P[6,1,:] = np.array([0, 0, 0, 0, 0, 0.1, 0, 0.1, 0, 0.8, 0])
P[6,2,:] = np.array([0, 0, 0, 0, 0, 0, 0.1, 0.8, 0, 0.1, 0])
P[6,3,:] = np.array([0, 0, 0, 0, 0, 0.8, 0.1, 0, 0, 0.1, 0])
# za s=7
P[7,0,:] = np.array([0, 0, 0, 0, 0.8, 0, 0.1, 0.1, 0, 0, 0])
P[7,1,:] = np.array([0, 0, 0, 0, 0, 0, 0.1, 0.1, 0, 0, 0.8])
P[7,2,:] = np.array([0, 0, 0, 0, 0.1, 0, 0, 0.8, 0, 0, 0.1])
P[7,3,:] = np.array([0, 0, 0, 0, 0.1, 0, 0.8, 0, 0, 0, 0.1])
# za s=8
P[8,0,:] = np.array([0, 0, 0, 0, 0, 0.8, 0, 0, 0.1, 0.1, 0])
P[8,1,:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.1, 0])
P[8,2,:] = np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0.1, 0.8, 0])
P[8,3,:] = np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0.9, 0, 0])

# gama - zadat parametar
gama = 1
iterations = np.zeros(20)
# delta = usrednjena promena Q vrednosti, za svako Q[s, a]
deltas = np.zeros(20)
for i in range(20):
    iterations[i] = i
    Qmax =  np.amax(Q, axis=1)  
    newQ = gama * np.sum(np.multiply(P, Qmax), axis=2) + np.reshape(R, (Ns,1))
    deltas[i] = np.sum(np.absolute(Q - newQ)) / Q.size
    Q = newQ    
print('Gama = 1: ')
plt.title("Iteracija Q vrednosti")
plt.xlabel("iteracija")
plt.ylabel("delta")
plt.plot(iterations, deltas, color="navy")
plt.show()
policy = np.argmax(Q, axis=1)
print('Optimalna politika: ')
print(policy)
print()

Q = np.zeros((Ns, Na))
policy = np.full(Ns, -1)
# gama - zadat parametar
gama = 0.9
iterations = np.zeros(20)
# delta = usrednjena promena Q vrednosti, za svako Q[s, a]
deltas = np.zeros(20)
for i in range(20):
    iterations[i] = i
    Qmax =  np.amax(Q, axis=1)  
    newQ = gama * np.sum(np.multiply(P, Qmax), axis=2) + np.reshape(R, (Ns,1))
    deltas[i] = np.sum(np.absolute(Q - newQ)) / Q.size
    Q = newQ    
print('Gama = 0.9: ')
plt.title("Iteracija Q vrednosti")
plt.xlabel("iteracija")
plt.ylabel("delta")
plt.plot(iterations, deltas, color="navy")
plt.show()
policy = np.argmax(Q, axis=1)
print('Optimalna politika: ')
print(policy)
print()
print()

Vs = np.amax(Q, axis=1)
print(Vs)

    
    
    

