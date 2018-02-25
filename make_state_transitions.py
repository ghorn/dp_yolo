#!/usr/bin/env python3

import numpy as np
import pickle
import time

import ocp_config

Nq = 101
Nw = 103
Nu = 51

qs = np.linspace(-np.pi, np.pi, Nq)
ws = np.linspace(-2*np.pi, 2*np.pi, Nw)
us = np.linspace(-5., 5., Nu)

if __name__ == '__main__':
    Xnext = np.zeros((Nq, Nw, Nu, Nu), dtype=int)
    Costs = np.zeros((Nq, Nw, Nu), dtype=float)

    print("computing state transition tensor")
    t0 = time.time()
    for kq in range(Nq):
        q = qs[kq]
        for kw in range(Nw):
            w = ws[kw]
            for ku in range(Nu):
                u = us[ku]
                next_x = ocp_config.rk4(np.array([q, w, 0.]), np.array(u))

                kq1 = np.argmin(np.abs(qs - next_x[0]))
                kw1 = np.argmin(np.abs(ws - next_x[1]))
                Xnext[kq, kw, ku, 0] = kq1
                Xnext[kq, kw, ku, 1] = kw1
                Costs[kq, kw, ku] = next_x[-1]
    t1 = time.time()
    print("computed state transition tensor in %.2f seconds" % (t1 - t0))

    opath = 'state_transitions.pickle'
    print('pickling to ' + opath)
    if __name__ == '__main__':
        with open(opath, 'wb') as f:
            pickle.dump([Xnext, Costs], f)
    print('succesfully pickled state transitions')
