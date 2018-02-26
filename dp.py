#!/usr/bin/env python3

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
    changes.append(0)
    sys.stdout.write('iteration %5d...' % k)
    sys.stdout.flush()
    NewValue = np.copy(Value)
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
            NewValue[kq, kw] = trial_values[ustar_k]

            # convergence
            if old_ustar_k != ustar_k:
                changes[-1] += 1

    Value = NewValue
    Value -= np.min(Value)
    sys.stdout.write('   max value %.2e' % np.max(Value))
    if k > 0:
        sys.stdout.write('%7d changes (%5.1f %%)\n' % (changes[-1], 100.*float(changes[-1])/float(old_changes)))
    else:
        sys.stdout.write('%7d changes\n' % changes[-1])
    sys.stdout.flush()
    old_changes = changes[-1]
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
