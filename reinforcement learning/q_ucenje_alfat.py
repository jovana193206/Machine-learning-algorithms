import numpy as np
from matplotlib import pyplot as plt
import random

# Ns=11 stanja - od 0 do 10
Ns = 11
# Na=4 akcije: levo=0, desno=1, gore=2, dole=3
Na = 4
# V vrednosti dobijene u prvom zadatku
V_true = np.array([0.29646448, 0.39851054, 0.50941538, 0.25395899, 0.64958636, 0.34478784, 0.48644045, 0.79536224, 0.12994131, -1, 1])
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
# kreiramo simulator
sim = Simulator()
iterations = np.zeros(50000)
# delta = usrednjena promena Q vrednosti, za svako Q[s, a]
deltas = np.zeros(50000)
avgQs = np.zeros(50000)
br_epizoda = 0
Vs = np.zeros((7000,11))
for i in range(50000):
    iterations[i] = i
    # adaptivna stopa ucenja
    t = br_epizoda + 1
    alfa = np.log(t + 1) / (t + 1)
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
    kraj_epizode = False
    if (s == 9) or (s==10):
        q = r
        kraj_epizode = True
        Vs[br_epizoda] = np.amax(Q, axis=1)
        br_epizoda = br_epizoda + 1
    else:
        q = r + gama*np.amax(Q[s_next,:])
        kraj_epizode = False
    # azuriramo Q(s,a)
    newQ = Q[s,a] + alfa*(q - Q[s,a])
    deltas[i] = np.absolute(Q[s,a] - newQ)
    Q[s,a] = newQ
    avgQs[i] = np.mean(Q)
epizode = np.arange(0, br_epizoda)
    

plt.title("Q ucenje")
plt.xlabel("iteracija")
plt.ylabel("avg(Q)")
plt.plot(iterations, avgQs, color="navy")
plt.show()
policy = np.argmax(Q, axis=1)
print('Optimalna politika: ')
print(policy)
print()
print('Broj epizoda: ', br_epizoda)

Vs = Vs[:br_epizoda, :]
Vs = Vs.T
for i in range(11):
    plt.title("S = " +  str(i))
    plt.xlabel("epizoda")
    plt.ylabel("V vrednost")
    plt.plot(epizode, Vs[i], color="navy")
    plt.plot(epizode, np.full(len(epizode), V_true[i]), color="red")
    plt.show()
    print()