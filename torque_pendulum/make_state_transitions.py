#!/usr/bin/env ipython3

import numpy as np
import pickle
import time

import ocp_config

Nq2 = 100
Nw2 = 101
Nu2 = 25

Nq = 1 + 2 * Nq2
Nw = 1 + 2 * Nw2
Nu = 1 + 2 * Nu2
h = 0.4

# fudge factor to make evenly spaced over angle wraps
y = float(2*(Nq2)) / float(2*(Nq2) + 1)
q_min = -y*np.pi
q_max =  y*np.pi

w_min = -4.
w_max =  4.

u_min = -0.8
u_max =  0.8

qs = np.linspace(q_min, q_max, Nq)
ws = np.linspace(w_min, w_max, Nw)
us = np.linspace(u_min, u_max, Nu)

# check the fudge factor
assert abs((qs[0] - qs[-1] + 2*np.pi) - (qs[-1] - qs[-2])) < 1e-9

print("computing state transition tensor")
Qs, Ws, Us = np.meshgrid(qs, ws, us, indexing='ij')

t0 = time.time()
(next_x, costs) = ocp_config.integrate(np.array([Qs, Ws]), Us, h)
t1 = time.time()
print("computed state transition tensor in %.2f seconds" % (t1 - t0))
#print(next_x.shape)
#print(costs.shape)

print("Really quickly finding closest indices")
t0 = time.time()
RFastQNext = np.argmin(np.abs(next_x[0,:,:,:,None] - qs), axis=3)
RFastWNext = np.argmin(np.abs(next_x[1,:,:,:,None] - ws), axis=3)
t1 = time.time()
#print(RFastQNext.shape)
#print(RFastWNext.shape)
print("Really quickly found closest indices in %.2f seconds" % (t1 - t0))


if False:
    print("Quickly finding closest indices")
    t0 = time.time()

    Qerrs = np.zeros((Nq, Nq, Nw, Nu), dtype=float)
    for k,q in enumerate(qs):
        Qerrs[k,:,:,:] = next_x[0,:,:,:] - q
    FastQNext = np.argmin(np.abs(Qerrs), axis=0)

    Werrs = np.zeros((Nw, Nq, Nw, Nu), dtype=float)
    for k,w in enumerate(ws):
        Werrs[k,:,:,:] = next_x[1,:,:,:] - w
    FastWNext = np.argmin(np.abs(Werrs), axis=0)

    t1 = time.time()
    print("Quickly found closest indices in %.2f seconds" % (t1 - t0))


    print("Finding closest indices")
    Qnext = np.zeros((Nq, Nw, Nu), dtype=int)
    Wnext = np.zeros((Nq, Nw, Nu), dtype=int)
    t0 = time.time()
    for kq in range(Nq):
        for kw in range(Nw):
            for ku in range(Nu):
                kq1 = np.argmin(np.abs(qs - next_x[0, kq, kw, ku]))
                kw1 = np.argmin(np.abs(ws - next_x[1, kq, kw, ku]))
                Qnext[kq, kw, ku] = kq1
                Wnext[kq, kw, ku] = kw1
    t1 = time.time()
    print("computed closest indices in %.2f seconds" % (t1 - t0))

    assert (Qnext == FastQNext).all(), np.max(np.abs(Qnext-FastQNext))
    assert (Wnext == FastWNext).all(), np.max(np.abs(Wnext-FastWNext))

    assert (FastQNext == RFastQNext).all(), np.max(np.abs(Qnext-FastQNext))
    assert (FastWNext == RFastWNext).all(), np.max(np.abs(Wnext-FastWNext))

opath = 'state_transitions.pickle'
print('pickling to ' + opath)
ret = {'Xnext':np.array([RFastQNext, RFastWNext]),
       'Costs':costs,
       'qs':qs,
       'ws':ws,
       'us':us}
with open(opath, 'wb') as f:
    pickle.dump(ret, f)
print('pickled state transitions to ' + opath)
