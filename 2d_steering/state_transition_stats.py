#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import sys

ipath = 'state_transitions.pickle'
with open(ipath, 'rb') as f:
    state_transitions = pickle.load(f)
Xnext = state_transitions['Xnext']
Costs = state_transitions['Costs']

pxs = state_transitions['pxs']
pys = state_transitions['pys']
qs = state_transitions['qs']
ws = state_transitions['ws']
us = state_transitions['us']

Npx = len(pxs)
Npy = len(pys)
Nq = len(qs)
Nw = len(ws)
Nu = len(us)

print('next omega range effected by u:')
print(Xnext[1, round(Nq/2), round(Nw/2), :])
print('next theta range effected by omega:')
print(Xnext[0, round(Nq/2), :, round(Nu/2)])
sys.exit()


print("computing index sets")
t0 = time.time()
Deltas = np.copy(Xnext)
delta_qs = []
delta_ws = []

QOptions = np.zeros((Nq, Nw), dtype=int)
WOptions = np.zeros((Nq, Nw), dtype=int)

QMax = np.zeros((Nq, Nw))
WMax = np.zeros((Nq, Nw))

for kq in range(Nq):
    for kw in range(Nw):
        q_set = set(Xnext[0, kq, kw, :])
        w_set = set(Xnext[1, kq, kw, :])

#        print(len(q_set))
#        print(len(w_set))
        QOptions[kq, kw] = len(q_set)
        WOptions[kq, kw] = len(w_set)

        QMax[kq, kw] = qs[max(q_set)]
        WMax[kq, kw] = ws[max(w_set)]

t1 = time.time()
print("computed delta indexes in %.1f seconds" % (t1 - t0))

plt.subplot(2, 2, 1)
X, Y = np.meshgrid(qs, ws)
S = plt.contourf(X, Y, QOptions.T)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('q opt')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('q options')


plt.subplot(2, 2, 2)
X, Y = np.meshgrid(qs, ws)
S = plt.contourf(X, Y, WOptions.T)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('w opt')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('w options')


plt.subplot(2, 2, 3)
X, Y = np.meshgrid(qs, ws)
S = plt.contourf(X, Y, QMax.T)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('max q')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('max q')


plt.subplot(2, 2, 4)
X, Y = np.meshgrid(qs, ws)
S = plt.contourf(X, Y, WMax.T)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('max w')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('max w')



#
#plt.subplot(2, 2, 4)
#plt.hist(delta_qs, bins=np.max(delta_qs) - np.min(delta_qs) + 1)
#plt.title('delta theta')
#
#plt.subplot(2, 2, 3)
#plt.hist(delta_ws, bins=np.max(delta_ws) - np.min(delta_ws) + 1)
#plt.title('delta omega')

plt.show()
