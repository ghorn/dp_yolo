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
Ustar = sol['Ustar']
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
    uroute = []

    k = 1
    while True:
        x = route[-1]
        #print(Costs[x[0], x[1], x[2], x[3]])
        ustar = Ustar[x[0], x[1], x[2], x[3]]
        x1 = Xnext[:, x[0], x[1], x[2], x[3], ustar]
        if (x1 == x).all():
            print('converged in %d iterations' % k)
            break
        else:
            route.append(x1)
            uroute.append(ustar)

        # prevent infinite iteration
        if k > 2000:
            print('failed after %d iters iters' % k)
            break
        k += 1
    print('route length %d' % (len(route)))
    return (np.array([get_values(r) for r in route]),
            np.array([us[uk] for uk in uroute]))

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
#plt.show()

(xs, u) = sim([round(Npx/2*0.25), round(Npy/2*0.25), round(Nq/2*(1-0.1)), 0])
plt.plot(xs[:,1], xs[:,0], 'r.')



plt.figure()
plt.subplot(5, 1, 1)
plt.plot(xs[:,0], '.')
plt.ylabel('x')
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(xs[:,1], '.')
plt.ylabel('y')
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(xs[:,2], '.')
plt.ylabel('q')
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(xs[:,3], '.')
plt.ylabel('w')
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(u, '.')
plt.ylabel('u')
plt.grid(True)


plt.show()
