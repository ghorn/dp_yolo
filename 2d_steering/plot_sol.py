#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import sys

ipath = 'solution.pickle'
with open(ipath, 'rb') as f:
    sol = pickle.load(f)
pxs = sol['pxs']
pys = sol['pys']
qs = sol['qs']
ws = sol['ws']
us = sol['us']

Npx = len(pxs)
Npy = len(pys)
Nq = len(qs)
Nw = len(ws)
Nu = len(us)

Value = sol['Value']
Ustar_k = sol['Ustar_k']
changes = sol['changes']
Xnext = sol['Xnext']
Costs = sol['Costs']

def get_values(xk):
    return np.array([pxs[xk[0]],
                     pys[xk[1]],
                     qs[xk[2]],
                     ws[xk[3]]])

def sim(x0):
    route = [x0]

    k = 1
    success = False
    while True:
        x = route[-1]
        trial_values = np.zeros(Nu)
        for ku in range(Nu):
            this_cost = Costs[x[0],x[1],x[2],x[3],ku]
            px_next = Xnext[0, x[0], x[1], x[2], x[3], ku]
            py_next = Xnext[1, x[0], x[1], x[2], x[3], ku]
            q_next  = Xnext[2, x[0], x[1], x[2], x[3], ku]
            w_next  = Xnext[3, x[0], x[1], x[2], x[3], ku]
        
            trial_values[ku] = this_cost + Value[px_next, py_next, q_next, w_next]
        ustar = np.argmin(trial_values, axis=0)
        x1 = Xnext[:, x[0], x[1], x[2], x[3], ustar]
        if (x1 == x).all():
            print('converged in %d iterations' % k)
            success = True
            break
        else:
            route.append(x1)

        # prevent infinite iteration
        if k > 2000:
            print('failed after %d iters iters' % k)
            break
        k += 1
    return (np.array([get_values(r) for r in route]), success)

plt.figure()
X, Y = np.meshgrid(pxs, pys, indexing='ij')
print(Value.shape)
Z = np.min(Value, axis=(2, 3))
print(Z.shape)
S = plt.contourf(Y, X, -Z, 300)
cbar = plt.colorbar(S)
cbar.ax.set_ylabel('-Value')
plt.xlabel('y [m]')
plt.ylabel('x [m]')
plt.title('Value function')

(xs, success) = sim([round(Npx/2), round(Npy/2), round(Nq/2), 1])
col = 'g.' if success else 'r.'
plt.plot(xs[1,:], xs[0,:], col)



plt.show()





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
plt.subplot(2, 1, 1)
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



plt.subplot(2, 1, 2)
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
