# -*- coding: utf-8 -*-
import numpy as np
import copy
import math
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def sumError(a, b, c, d, t):
    return a + b * t + c * t ** 2 + d * t ** 3



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    count = 2000
    # Create random input and output data
    x = np.linspace(-math.pi, math.pi, count)
    y_prim = copy.deepcopy(x)
    y = np.sin(x)
    plt.plot(x, y, 'o')
    plt.show()
   # plt.plot(x, y, 'o')
    #plt.show()
    # Randomly initialize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    e = np.random.randn()

    learning_rate = 1e-6
    for t in range(2 * count):
        # Forward pass: compute predicted y
        # y = a + b x + c x^2 + d x^3
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if t % 100 == 0:
            print(t, loss)
            print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()
        #grad_e = (grad_y_pred * x ** 4).sum()
        # Update weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        #e -= learning_rate * grad_e
    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
    step = 0
    for i in x:
         result = sumError(a,b,c,d,i)
         np.put(y_prim, step, result)
        #y_prim.flat[step] = result
         step = step + 1
    #plt.plot(x, y, 'o')
    #plt.show()

    #plt.plot(x, y, 'o')
    #plt.show()
    #y_prim = y_prim/math.pi
    plt.plot(x, y, 'p')
    plt.plot(x, y_prim, 'p')
    plt.show()
    #print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
    #print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3 + {e} x^4')

