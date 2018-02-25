#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import sys

import make_state_transitions

Nq = make_state_transitions.Nq
Nw = make_state_transitions.Nw
Nu = make_state_transitions.Nu

qs = np.linspace(-np.pi, np.pi, Nq)
ws = np.linspace(-3*np.pi, 3*np.pi, Nw)
us = np.linspace(-1, 1, Nu)

ipath = 'state_transitions.pickle'
with open(ipath, 'rb') as f:
    [Xnext, Costs] = pickle.load(f)

max_iter = 5000
Value = np.zeros((Nq, Nw))
Ustar_k = np.zeros((Nq, Nw), dtype=int)

print("running DP iterations")
t0 = time.time()
for k in range(max_iter):
    changes = 0
    sys.stdout.write('iteration %5d...' % k)
    sys.stdout.flush()
    for kq in range(Nq):
        for kw in range(Nw):
            old_ustar_k = Ustar_k[kq, kw]
            trial_values = []
            for ku in range(Nu):
                this_cost = Costs[kq, kw, ku]
                q_next = Xnext[kq, kw, ku, 0]
                w_next = Xnext[kq, kw, ku, 1]
                trial_values.append(this_cost + Value[q_next, w_next])

            ustar_k = np.argmin(trial_values)
            Ustar_k[kq, kw] = ustar_k
            Value[kq, kw] = trial_values[ustar_k]

            # convergence
            if old_ustar_k != ustar_k:
                changes += 1

    sys.stdout.write('%7d changes\n' % changes)
    sys.stdout.flush()
    if changes == 0:
        break
t1 = time.time()

if changes > 0:
    print("failed to converge in %.2f seconds, %d iterations" % (t1 - t0, max_iter))
    sys.exit()
else:
    print("converged in %.2f seconds, %d iterations" % (t1 - t0, k))
    opath = 'solution.pickle'
    if __name__ == '__main__':
        with open(opath, 'wb') as f:
            sol = {'qs':qs, 'ws':ws, 'us':us,
                   'Value':Value, 'Ustar_k': Ustar_k}
            pickle.dump(sol, f)
    print('succesfully pickled state transitions')
