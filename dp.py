#!/usr/bin/env ipython3

import numpy as np
import pickle
import matplotlib.pyplot as plt
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

max_iter = 5000
Value = np.zeros((Nq, Nw))
Ustar_k = np.zeros((Nq, Nw), dtype=int)

print("running DP iterations")
t0 = time.time()
changes = []
for k in range(max_iter):
    sys.stdout.write('iteration %5d...' % k)
    sys.stdout.flush()

    trial_values = np.zeros((Nq, Nw, Nu))
    for ku in range(Nu):
        this_cost = Costs[:, :, ku]
        q_next = Xnext[0, :, :, ku]
        w_next = Xnext[1, :, :, ku]

        trial_values[:,:,ku] = this_cost + Value[q_next, w_next]

    Ustar_k_old = np.copy(Ustar_k)
    Ustar_k = np.argmin(trial_values, axis=2)

    NewValue = np.min(trial_values, axis=2)

    # convergence
    changes.append(np.sum(Ustar_k != Ustar_k_old))


    Value = NewValue
    Value -= np.min(Value)
    not_inf = np.logical_not(np.isinf(Value))
    sys.stdout.write('   max value %.2e' % np.max(Value[not_inf]))
    if k > 0:
        sys.stdout.write('%7d changes (%5.1f %%)\n' %
                         (changes[-1], 100.*float(changes[-1])/float(changes[-2])))
    else:
        sys.stdout.write('%7d changes\n' % changes[-1])
    sys.stdout.flush()

    if changes[-1] == 0:
        break
t1 = time.time()

if changes[-1] > 0:
    print("failed to converge in %.2f seconds, %d iterations" % (t1 - t0, max_iter))
    sys.exit()
else:
    print("converged in %.2f seconds, %d iterations" % (t1 - t0, k))
    opath = 'solution.pickle'
    if __name__ == '__main__':
        with open(opath, 'wb') as f:
            sol = {'qs':qs, 'ws':ws, 'us':us,
                   'Value':Value, 'Ustar_k': Ustar_k,
                   'changes':changes}
            pickle.dump(sol, f)
    print('pickled solution to ' + opath)
