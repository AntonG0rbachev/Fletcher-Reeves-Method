import math
import numpy
from sympy import *
from autograd import grad

function = lambda x1, x2: x1 ** 2 + 7 * x2 ** 2 - x1 * x2 + x1
calc_t = lambda x1, x2, d1, d2: (((d1 * (x2 - 2 * x1 - 1) + d2 * (x1 - 14 * x2)) /
                                  (2 * (d1 ** 2 + 14 * d2 ** 2 - 2 * d1 * d2))))
x0 = numpy.array([1.1, 1.1])
eps1 = 0.1
eps2 = 0.15
iterCounts = 10


def norm(args):
    return math.sqrt(sum(map(lambda x: x ** 2, args)))


def fletcher_reeves_method(epsilon_1, epsilon_2, max_iter_counts, init_approximation, func, calc_step):
    x_min = numpy.array([])
    gradient = grad(func, [0, 1])
    iter_counts = 0
    check_counts = 0
    new_approximation = init_approximation.copy()
    iter_counts += 1
    gradient_in_point = numpy.array(gradient(new_approximation[0], new_approximation[1]))
    if norm(gradient_in_point) < eps1:
        x_min = new_approximation
        return {"x_min": list(x_min), "f(x_min)": func(x_min[0], x_min[1]), "iterations": iter_counts}
    d = -1 * gradient_in_point
    t = calc_step(new_approximation[0], new_approximation[1], d[0], d[1])
    prev_approximation = new_approximation.copy()
    new_approximation = new_approximation + t * d
    while True:
        iter_counts += 1
        gradient_in_point = numpy.array(gradient(new_approximation[0], new_approximation[1]))
        gradient_norm = norm(gradient_in_point)
        if gradient_norm < epsilon_1:
            x_min = new_approximation
            break
        if iter_counts >= max_iter_counts:
            x_min = new_approximation
            break
        prev_gradient_in_point = gradient(prev_approximation[0], prev_approximation[1])
        betta = (norm(gradient_in_point) ** 2) / (norm(prev_gradient_in_point) ** 2)
        d = -1 * gradient_in_point + betta * d
        t = calc_step(new_approximation[0], new_approximation[1], d[0], d[1])
        prev_approximation = new_approximation.copy()
        new_approximation = new_approximation + t * d
        if (norm(new_approximation - prev_approximation) < epsilon_2 and abs(
                func(new_approximation[0], new_approximation[1]) - func(prev_approximation[0], prev_approximation[1]))):
            if check_counts == 2:
                x_min = new_approximation
                break
            else:
                check_counts += 1
    return {"x_min": list(x_min), "f(x_min)": func(x_min[0], x_min[1]), "iterations": iter_counts}


print(fletcher_reeves_method(eps1, eps2, iterCounts, x0, function, calc_t))
