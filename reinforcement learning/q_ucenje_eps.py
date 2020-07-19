import numpy as np
from matplotlib import pyplot as plt
import random

# Ns=11 stanja - od 0 do 10
Ns = 11
# Na=4 akcije: levo=0, desno=1, gore=2, dole=3
Na = 4

class Simulator:
    def __init__(self):
        # stanje u kom se trenutno nalazimo
        self.state = 0
        # ukupan broj stanja
        self.Ns = 11
        # ukupan broj akcija
        self.Na = 4
        # nagrade svih stanja
        self.R = np.full((self.Ns), -0.04)
        self.R[9] = -1
        self.R[10] = 1
        # matrica prelaska iz jednog stanja u drugo pod dejstvom odredjene akcije
        # T[s, a] = s' (s' je broj izmedju 0 i 10, odnosno oznaka sledeceg stanja)
        self.T = np.zeros((self.Ns, self.Na), dtype=int)
        self.T[:,0] = (np.array([0, 1, 2, 0, 2, 3, 6, 4, 5, 0, 0])).astype(int)
        self.T[:,1] = (np.array([3, 1, 4, 5, 7, 8, 9, 10, 8, 0, 0])).astype(int)
        self.T[:,2] = (np.array([1, 2, 2, 3, 4, 6, 7, 7, 9, 0, 0])).astype(int)
        self.T[:,3] = (np.array([0, 0, 1, 3, 4, 5, 5, 6, 8, 0, 0])).astype(int)
    # metod koji se koristi za izvrsavanje poteza
    # u trenutnom stanju primenjuje zadatu akciju a i vraca sledece stanje i dobijenu nagradu
    def move(self, a):
        s_next = self.T[self.state, a]
        r = self.R[self.state]
        self.state = s_next
        return s_next, r



# matrica koja cuva Q vrednosti - NsxNa, inicijalno sve 0
Q = np.zeros((Ns, Na))
# gama - zadat parametar
gama = 0.9
# eps - verovatnoca da se akcija izabere nasumicno
eps = 0.5
# stopa ucenja - alfa
alfa = 1
# kreiramo simulator
sim = Simulator()
iterations = np.zeros(2000)
# delta = usrednjena promena Q vrednosti, za svako Q[s, a]
deltas = np.zeros(2000)
avgQs = np.zeros(2000)
for i in range(2000):
    iterations[i] = i
    # ocitavamo trenutno stanje
    s = sim.state
    # biramo akciju nasumicno
    if random.random() <= eps:
       a = random.randrange(4) 
    # biramo akciju koja daje max Q 
    else:
      a = np.argmax(Q[s,:]) 
    # pravimo potez - izvrsavamo izabranu akciju
    s_next, r = sim.move(a)
    # racunamo q(s,a)
    # ako je s == 9 ili 10, stigli smo do kraja epizode
    if (s == 9) or (s==10):
        q = r
    else:
        q = r + gama*np.amax(Q[s_next,:])
    # azuriramo Q(s,a)
    newQ = Q[s,a] + alfa*(q - Q[s,a])
    deltas[i] = np.absolute(Q[s,a] - newQ)
    Q[s,a] = newQ
    avgQs[i] = np.mean(Q)
    
print('Eps = 0.5, alfa = 1: ')
plt.title("Q ucenje")
plt.xlabel("iteracija")
plt.ylabel("delta")
plt.plot(iterations, deltas, color="navy")
plt.show()
plt.title("Q ucenje")
plt.xlabel("iteracija")
plt.ylabel("avg(Q)")
plt.plot(iterations, avgQs, color="navy")
plt.show()
policy = np.argmax(Q, axis=1)
print('Optimalna politika: ')
print(policy)
print()