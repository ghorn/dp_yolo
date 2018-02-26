#!/usr/bin/env python3

import numpy as np
import time

def cost_function(x, u):
    return x[0]**2 + x[1]**2 + u**2


# differential equation
def ode(x, u):
    q = x[0]
    w = x[1]

    ddt_q = w
    ddt_w = -0.*w + np.sin(q) + u
    return np.array([ddt_q, ddt_w, cost_function(x, u)])


def rk4(x0, u, h):
    k1 = ode(x0,            u)
    k2 = ode(x0 + 0.5*h*k1, u)
    k3 = ode(x0 + 0.5*h*k2, u)
    k4 = ode(x0 +     h*k3, u)
    return x0 + h/6. * (k1 + 2.*k2 + 2.*k3 + k4)


def wrap_angle(q):
    while q > np.pi:
        q -= 2*np.pi
    while q < -np.pi:
        q += 2*np.pi
    return q


def integrate(x0, u):
    next_x = rk4(x0, u)
    next_x[0] = wrap_angle(next_x[0])
    return next_x

## state dimensions and ranges
##states_config = [(-np.pi, np.pi, 50), (-10, 10, 70)]
#states_config = [(-np.pi, np.pi, 3), (-10, 10, 4)]
#
## control dimensions
#control_config = [(-0.5, 0.5, 30)]
#
##print(wats.shape)
##sys.exit()
##print(wats)
#Xs = np.meshgrid(*[np.linspace(x0,x1,n) for (x0,x1,n) in states_config])
#print(len(Xs))
#print(Xs[0].shape)
#print(Xs[1].shape)
#
#print(Xs[0])
#print(Xs[1])
#
#ode(Xs, Us)
#next_x = []



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Xs = [np.array([0.0, 0.1, 0])]
    for k in range(round(20./h)):
        Xs.append(integrate(Xs[-1], 0))
    Xs = np.array(Xs)

    ts = h * np.array(range(0, Xs.shape[0]))

    plt.subplot(3, 1, 1)
    plt.plot(ts, np.sin(Xs[:,0]))
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(ts, np.cos(Xs[:,0]))
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ts, Xs[:,1])
    plt.grid(True)

    plt.show()
