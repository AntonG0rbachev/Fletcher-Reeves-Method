from matplotlib import pyplot
from sympy import *
import numpy as np


def function(x1, x2):
    return x1 ** 2 + 7 * x2 ** 2 - x1 * x2 + x1


X, Y = np.meshgrid([i for i in range(-1024, 1024)], [i for i in range(-512, 512)])
Z = function(X, Y)

fig = pyplot.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('function for minimization')
ax.plot_surface(X, Y, Z, cmap='inferno')
pyplot.show()
