# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:08:44 2023

@author: teddy
"""
import numpy as np
import matplotlib.pyplot as plt

def f_x(x):
    return 1 / (x**2 + 1)

def divided_diff(x, y):
    '''
    function to calculate the divided
    differences table
    '''
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def newton_poly(coef, x_data, x):
    '''
    evaluate the newton polynomial 
    at x
    '''
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p

n_values = [11, 21, 31, 41]

for n in n_values:
    x_nodes = np.linspace(-5, 5, n + 1)
    y_nodes = f_x(x_nodes)

    dd_table = divided_diff(x_nodes, y_nodes)
    coef = dd_table[0, :]

    x_values = np.linspace(-5, 5, 100)
    true_values = f_x(x_values)
    interpolating_values = newton_poly(coef, x_nodes, x_values)

    plt.figure(figsize=(12, 8))
    plt.plot(x_values, true_values, label='f(x)')
    plt.plot(x_values, interpolating_values, label=f'Interpolating Polynomial n={n}')
    plt.scatter(x_nodes, y_nodes, label='Interpolation Nodes')
    plt.legend()
    plt.title(f'Interpolating Polynomial vs. f(x) for n={n}')
    plt.show()
'''
As I increase the values for n, the interpolation becomes more and more acurate. However, the first 
two points and the last two points still have very large stretches compared to where they should be. 
However, within the interval, the newton interpolated polynomial aligns perfectly with the real f(x)
function value. This is known as the Runge's phenomenon and is a problem of oscillation at the edges
of an interval that occurs when using interpolation of polynomial of high degree over a set of 
equidistant points.
'''