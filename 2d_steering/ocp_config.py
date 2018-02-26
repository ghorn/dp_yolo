#!/usr/bin/env ipython3

import numpy as np
import sys

terminal_radius = 100.0

def _wrap_angle(q):
    return np.remainder(q + np.pi, 2*np.pi) - np.pi


def _cost_function(x, u):
    return 1 + 2*u**2

#   return 1 + 1e-2*(x[0]**2 + x[1]**2 + _wrap_angle(x[2])**2 + x[3]**2 + u**2)
#    height = np.cos(x[0]) - 1
#    return 1 + 1e-2*(height**2 + x[1]**2 + 1e-6*u**2)


# differential equation
def _ode(x, u):
    px = x[0]
    py = x[1]
    q  = x[2]
    w  = x[3]

    v = 30.0
    ddt_x = v*np.cos(q)
    ddt_y = v*np.sin(q)
    ddt_q = w
    ddt_w = 0*ddt_q

    return np.array([ddt_x, ddt_y, ddt_q, ddt_w]), _cost_function(x, u)


def _rk4(x0, u, h):
    k1, ck1 = _ode(x0,            u)
    k2, ck2 = _ode(x0 + 0.5*h*k1, u)
    k3, ck3 = _ode(x0 + 0.5*h*k2, u)
    k4, ck4 = _ode(x0 +     h*k3, u)

    next_x = x0 + h/6. * (k1 + 2.*k2 + 2.*k3 + k4)
    cost = h/6. * (ck1 + 2.*ck2 + 2.*ck3 + ck4)
    return (next_x, cost)


def integrate(x0, u, h):
    (_, Npx, Npy, Nq, Nw, Nu) = x0.shape
    #print(x0.shape)
    #print(u.shape)
    (next_x, cost) = _rk4(x0, u, h)
    #print(next_x.shape)

    # handle turn rates
    dw = x0[3,0,0,0,1,0] - x0[3,0,0,0,0,0]
    #print(next_x[3, round(Npx/2), round(Npy/2), round(Nq/2), round(Nw/2), :])
    next_x[3][u ==  1] += dw
    next_x[3][u == -1] -= dw
    #print(next_x[3, round(Npx/2), round(Npy/2), round(Nq/2), round(Nw/2), :])
    #sys.exit(1)

    # wrap angles
    next_x[2] = _wrap_angle(next_x[2])

    # fudge factor for rounding
    dpx = x0[0,1,0,0,0,0] - x0[0,0,0,0,0,0]
    dpy = x0[1,0,1,0,0,0] - x0[1,0,0,0,0,0]

    xtoo_small = next_x[0] < np.min(x0[0]) - 0.5*dpx
    xtoo_large = next_x[0] > np.max(x0[0]) + 0.5*dpx

    ytoo_small = next_x[1] < np.min(x0[1]) - 0.5*dpy
    ytoo_large = next_x[1] > np.max(x0[1]) + 0.5*dpy
    oob = np.logical_or(xtoo_small, xtoo_large)
    oob = np.logical_or(oob, ytoo_small)
    oob = np.logical_or(oob, ytoo_large)
    cost[oob]=np.inf

    inbnds = x0[0]**2 + x0[1]**2 < terminal_radius**2
    obnds = np.logical_not(inbnds)
#    print(next_x.shape)
#    print(x0.shape)
#    print(inbnds.shape)

    next_x[0] = next_x[0]*obnds + x0[0]*inbnds
    next_x[1] = next_x[1]*obnds + x0[1]*inbnds
    next_x[2] = next_x[2]*obnds + x0[2]*inbnds
    next_x[3] = next_x[3]*obnds + x0[3]*inbnds

    cost = cost*obnds + _wrap_angle(x0[2])**2*inbnds # heading

#    if np.isnan(next_x).any():
#        sys.exit(1)
#    if np.isnan(cost).any():
#        sys.exit(1)
#    print('success!')
#    sys.exit(1)
    return (next_x, cost, terminal_radius)


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

    Xs = [np.array([0, 0, 0, 10*3.14/180.0])]
    costs = []
    for k in range(round(20./h)):
        #next_x, cost, _ = integrate(Xs[-1], 0, h)
        ddtx, cost = _ode(Xs[-1], 0)
        Xs.append(Xs[-1] + h * ddtx)
        costs.append(cost)
    Xs = np.array(Xs)

    ts = h * np.array(range(0, Xs.shape[0]))

    plt.figure()
    plt.subplot(6, 1, 1)
    plt.plot(ts, Xs[:,0])
    plt.ylabel('x')
    plt.grid(True)

    plt.subplot(6, 1, 2)
    plt.plot(ts, Xs[:,1])
    plt.ylabel('y')
    plt.grid(True)

    plt.subplot(6, 1, 3)
    plt.plot(ts, Xs[:,2])
    plt.ylabel('q')
    plt.grid(True)

    plt.subplot(6, 1, 4)
    plt.plot(ts, Xs[:,3])
    plt.ylabel('w')
    plt.grid(True)

    plt.subplot(6, 1, 6)
    plt.plot(ts[:-1], costs)
    plt.ylabel('cost')
    plt.grid(True)

    plt.show()
