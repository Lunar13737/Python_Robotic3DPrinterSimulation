#Python 3.9
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import random as rd
import pandas as pd
def avg(list): #get the avg of a num list
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)
def mate(p1,p2): # return a list of offspring
    num = 2 * K // P #num of kids a couple has; also len(k)
    k = []
    for i in range(num):
        r = rd.random()
        k.append(r * p1 + (1 - r) * p2)
    return k
def CoFun(x): #cost function
    y = (x + (mt.pi / 2) * mt.sin(x)) ** 2
    return y
S = 50; P = 12; K = 12 #total strings, parents, offspring
TOL = 1E-6 #tolorance
G = 100 #max generations
dv = 1 #total design varialbles per string
Pi = []; Pi_min = []; Pi_avg = []
Lambda = []
new_pi = []
g = 1
for i in range(S):
    x = rd.uniform(-20,20)
    pi = CoFun(x)
    Lambda.append(x)
    new_pi.append(pi)
pf = pd.DataFrame([new_pi,Lambda])
pf = pd.DataFrame(pf.values.T)
pf.sort_values(by=[0],ascending=True,
               inplace=True,ignore_index=True)
Pi_min.append(pf[0][0])
Pi_avg.append(avg(new_pi))
Pi.append(Lambda) #index = g - 1
while g < G:
    g += 1
    new_pi.clear()
    Lambda.clear()
    for i in range(P): #get parents
        Lambda.append(pf[1][i])
    for i in range(P // 2): #mate
        cind = 0
        Lambda += mate(pf[1][cind],pf[1][cind+1])
        cind += 2
    while len(Lambda) < S: #add new chanllengers
        x = rd.uniform(-20,20)
        Lambda.append(x)
    for i in range(S): #calculate pi for every x in Lambda
        pi = CoFun(Lambda[i])
        new_pi.append(pi)
    pf = pd.DataFrame([new_pi,Lambda])
    pf = pd.DataFrame(pf.values.T)
    pf.sort_values(by=[0],ascending=True,
                   inplace=True,ignore_index=True)
    Pi_min.append(pf[0][0])
    Pi_avg.append(avg(new_pi))
    Pi.append(Lambda)
    if new_pi[0] <= TOL:
        break

fig, ax1 = plt.subplots()
ax1.set_xlabel('Generations',font = 'Times New Roman',fontsize = 14)
ax1.set_ylabel('Cost of the Best Design',
               color = 'darkorange',font = 'Times New Roman',fontsize = 14)
ax1.semilogy(Pi_min,linewidth = 0.8,color = 'darkorange',zorder = 2,label = 'minimum')
ax1.legend(loc=2)
if g < 40:
    ax1.semilogy(Pi_min,'.',color = 'darkorange',zorder = 2)
ax1.tick_params(axis='y',labelcolor = 'darkorange')

ax2 = ax1.twinx() 

ax2.set_ylabel(' Mean Cost of All the Designs',
               color = 'dodgerblue',font = 'Times New Roman',fontsize = 14) 
ax2.plot(Pi_avg,linewidth = 0.8,color = 'dodgerblue',label = 'average')
ax2.legend(loc=1)
ax2.tick_params(axis='y',labelcolor = 'dodgerblue')

fig.tight_layout()
plt.savefig('MATSCI_C286_Project_1_Q2b.pdf')
plt.show()