#!/usr/bin/env ipython3

import numpy as np
import sys

def _cost_function(x, u):
    return 1 + 1e-4*(x[0]**2 + x[1]**2 + u**2)


# differential equation
def _ode(x, u):
    q = x[0]
    w = x[1]

    ddt_q = w
    ddt_w = -0.*w + np.sin(q) + u

    return np.array([ddt_q, ddt_w]), _cost_function(x, u)


def _rk4(x0, u, h):
    k1, ck1 = _ode(x0,            u)
    k2, ck2 = _ode(x0 + 0.5*h*k1, u)
    k3, ck3 = _ode(x0 + 0.5*h*k2, u)
    k4, ck4 = _ode(x0 +     h*k3, u)

    next_x = x0 + h/6. * (k1 + 2.*k2 + 2.*k3 + k4)
    cost = h/6. * (ck1 + 2.*ck2 + 2.*ck3 + ck4)
    return (next_x, cost)


def _wrap_angle(q):
    return np.remainder(q + np.pi, 2*np.pi) - np.pi

def _ghetto_wrap_angle(q):
    while q > np.pi:
        q -= 2*np.pi
    while q < -np.pi:
        q += 2*np.pi
    return q


def integrate(x0, u, h):
    (next_x, cost) = _rk4(x0, u, h)
    next_x[0] = _wrap_angle(next_x[0])

    # fudge factor for rounding
    delta1 = x0[1,0,1,0] - x0[1,0,0,0]
    too_small = next_x[1] < np.min(x0[1]) - 0.5*delta1
    too_large = next_x[1] > np.max(x0[1]) + 0.5*delta1
    oob = np.logical_or(too_small, too_large)
    cost[oob]=np.inf
    return (next_x, cost)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
#    qs = np.linspace(-9*np.pi, 12*np.pi, 5000)
#
##    plt.plot(qs, qs)
##    plt.plot(qs, [_ghetto_wrap_angle(q) for q in qs])
##    plt.plot(qs, [_wrap_angle(q) for q in qs])
##    plt.grid(True)
##    plt.show()
##    sys.exit()
#
#    for q in qs:
#        assert abs(_ghetto_wrap_angle(q) - _wrap_angle(q)) < 1e-9, str((_ghetto_wrap_angle(q), _wrap_angle(q)))
#    print('success!!!')
#    sys.exit()
    h = 0.1

    Xs = [np.array([0.0, 0.1])]
    costs = []
    for k in range(round(20./h)):
        next_x, cost = integrate(Xs[-1], 0, h)
        Xs.append(next_x)
        costs.append(cost)
    Xs = np.array(Xs)

    ts = h * np.array(range(0, Xs.shape[0]))

    plt.subplot(3, 1, 1)
    plt.plot(ts, np.sin(Xs[:,0]), '.')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(ts, np.cos(Xs[:,0]), '.')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ts, Xs[:,1])
    plt.grid(True)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(ts, Xs[:,0])
    plt.ylabel('theta')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(ts, Xs[:,1])
    plt.ylabel('omega')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ts[:-1], costs)
    plt.ylabel('cost')
    plt.grid(True)

    plt.show()
