#!/usr/bin/env ipython3

import numpy as np
import pickle
import time

import ocp_config

Npx2 = 50
Npy2 = 51

Npx = 1 + 2 * Npx2
Npy = 1 + 2 * Npy2
Nq = 36
Nw = 3
Nu = 3

h = 1.0

#w_min = -4.
#w_max =  4.
#
#u_min = -0.8
#u_max =  0.8

pxs = np.linspace(-500.0, 500.0, Npx)
pys = np.linspace(-500.0, 500.0, Npy)
qs = np.linspace(-np.pi, np.pi, Nq + 1)[:-1]
#ws = np.array([-10, -5, 0, -5, 10])#*np.pi/180.0
ws = np.linspace(-10, 10, 3) * np.pi/180.0
us = [-1, 0, 1]

#print(ws[-1]*h)
#print(qs[-1] - qs[-2])
#sys.exit(1)

print("computing state transition tensor")
Pxs, Pys, Qs, Ws, Us = np.meshgrid(pxs, pys, qs, ws, us, indexing='ij')

t0 = time.time()
(next_x, costs, terminal_radius) = ocp_config.integrate(np.array([Pxs, Pys, Qs, Ws]), Us, h)
t1 = time.time()
print("computed state transition tensor in %.2f seconds" % (t1 - t0))
#print(next_x.shape)
#print(costs.shape)

print("Really quickly finding closest indices")
t0 = time.time()
XsNext = np.argmin(np.abs(next_x[0,:,:,:,:,:,None] - pxs), axis=5)
YsNext = np.argmin(np.abs(next_x[1,:,:,:,:,:,None] - pys), axis=5)
QsNext = np.argmin(np.abs(next_x[2,:,:,:,:,:,None] - qs), axis=5)
WsNext = np.argmin(np.abs(next_x[3,:,:,:,:,:,None] - ws), axis=5)
t1 = time.time()
#print(RFastQNext.shape)
#print(RFastWNext.shape)
print("Really quickly found closest indices in %.2f seconds" % (t1 - t0))


opath = 'state_transitions.pickle'
print('pickling to ' + opath)
ret = {'Xnext':np.array([XsNext, YsNext, QsNext, WsNext]),
       'Costs':costs,
       'terminal_radius':terminal_radius,
       'pxs':pxs,
       'pys':pys,
       'qs':qs,
       'ws':ws,
       'us':us}
with open(opath, 'wb') as f:
    pickle.dump(ret, f)
print('pickled state transitions to ' + opath)
