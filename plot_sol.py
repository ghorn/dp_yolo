#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

ipath = 'solution.pickle'
with open(ipath, 'rb') as f:
    sol = pickle.load(f)
qs = sol['qs']
ws = sol['ws']
us = sol['us']
Value = sol['Value']
Ustar_k = sol['Ustar_k']
changes = sol['changes']

Nq = len(qs)
Nw = len(ws)
Nu = len(us)


plt.subplot(2, 2, 1)
X, Y = np.meshgrid(qs, ws)
S = plt.contourf(X, Y, -Value.T, 300)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('-Value')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('Value function')


plt.subplot(2, 2, 2)
U = np.zeros((Nq, Nw))
for kq in range(Nq):
    for kw in range(Nw):
        U[kq, kw] = us[Ustar_k[kq, kw]]
S = plt.contourf(X, Y, U.T, 300)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('u')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('optimal action')

plt.subplot(2, 2, 3)
plt.hist(U.flatten(), bins=25)
plt.xlabel('u*')
plt.ylabel('count')
plt.title('action historgram')

plt.subplot(2, 2, 4)
plt.semilogy(changes, '.')
plt.grid(True)
plt.title('relative U changes')
plt.xlabel('iteration')
plt.ylabel('count')


plt.figure()
plt.subplot(1, 2, 1)
X, Y = np.meshgrid(qs, ws)
Z = -Value.T
S = plt.contourf(np.hstack((X - 2*np.pi, X, X + 2*np.pi)),
                 np.hstack((Y, Y, Y)),
                 np.hstack((Z, Z, Z)),
                 300)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('-Value')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('Value function')



plt.subplot(1, 2, 2)
X, Y = np.meshgrid(qs, ws)
Z = -Value.T
S = plt.contourf(np.hstack((X - 2*np.pi, X, X + 2*np.pi)),
                 np.hstack((Y, Y, Y)),
                 np.hstack((U.T, U.T, U.T)),
                 300)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('u*')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.title('u*(x)')



plt.show()
