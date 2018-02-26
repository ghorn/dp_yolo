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

qs = state_transitions['qs']
ws = state_transitions['ws']
us = state_transitions['us']

Nq = len(qs)
Nw = len(ws)
Nu = len(us)


print("computing delta indexes")
t0 = time.time()
Deltas = np.copy(Xnext)
delta_qs = []
delta_ws = []

Delta_qs = np.zeros((Nq, Nw, Nu), dtype=int)
Delta_ws = np.zeros((Nq, Nw, Nu), dtype=int)

for kq in range(Nq):
    for kw in range(Nw):
        for ku in range(Nu):
            q_next = Xnext[kq, kw, ku, 0]
            w_next = Xnext[kq, kw, ku, 1]
            delta_qs.append(np.abs(q_next - kq))
            delta_ws.append(np.abs(w_next - kw))
            Delta_qs[kq, kw, ku] = q_next - kq
            Delta_ws[kq, kw, ku] = w_next - kw
t1 = time.time()
print("computed delta indexes in %.1f seconds" % (t1 - t0))
            
#plt.subplot(2, 2, 1)
#X, Y = np.meshgrid(qs, ws)
#S = plt.contourf(X, Y, Delta_qs.T, 300)
#cbar = plt.colorbar(S)
#cbar.ax.set_ylabel('delta q')
#plt.xlabel('theta [rad]')
#plt.ylabel('omega [rad/s]')
#plt.title('Delta q')
#
#
#plt.subplot(2, 2, 2)
#X, Y = np.meshgrid(qs, ws)
#S = plt.contourf(X, Y, Delta_ws.T, 300)
#cbar = plt.colorbar(S)
#cbar.ax.set_ylabel('delta w')
#plt.xlabel('theta [rad]')
#plt.ylabel('omega [rad/s]')
#plt.title('Delta w')




plt.subplot(2, 2, 4)
plt.hist(delta_qs, bins=np.max(delta_qs) - np.min(delta_qs) + 1)
plt.title('delta theta')

plt.subplot(2, 2, 3)
plt.hist(delta_ws, bins=np.max(delta_ws) - np.min(delta_ws) + 1)
plt.title('delta omega')

plt.show()
