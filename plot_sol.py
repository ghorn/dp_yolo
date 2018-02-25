#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

import make_state_transitions

Nq = make_state_transitions.Nq
Nw = make_state_transitions.Nw
Nu = make_state_transitions.Nu


ipath = 'solution.pickle'
with open(ipath, 'rb') as f:
    sol = pickle.load(f)
qs = sol['qs']
ws = sol['ws']
us = sol['us']
Value = sol['Value']
Ustar_k = sol['Ustar_k']


plt.subplot(1, 3, 1)
X, Y = np.meshgrid(qs, ws)
Z = Value
S = plt.contourf(X, Y, Z.T, 300)#, cmap=plt.cm.bone, origin=origin)


plt.subplot(1, 3, 2)
U = Value
for kq in range(Nq):
    for kw in range(Nw):
        U[kq, kw] = us[Ustar_k[kq, kw]]
Z = Value
S = plt.contourf(X, Y, U.T, 300)#, cmap=plt.cm.bone, origin=origin)

plt.subplot(1, 3, 3)
plt.hist(U.flatten(), bins=25)


plt.show()
