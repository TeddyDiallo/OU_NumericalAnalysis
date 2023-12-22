# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 08:14:08 2023

@author: teddy
"""

import numpy as np
import matplotlib.pyplot as plt


def sin_x(x):
    return np.sin(x)

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

#Generate n+1 equally spaced nodes in the interval [0, 5]
#n = 10
#n = 20
n = 5
x_nodes = np.linspace(0, 5, n+1)
y_nodes = sin_x(x_nodes)

# Calculate the divided difference coefficients
dd_table = divided_diff(x_nodes, y_nodes)
coef = dd_table[0, :]  # Extract the coefficients for the Newton polynomial

#Define the function for the Taylor polynomial of sin(x) around x=0
def taylor_sin(order, x_values):
    """
    Evaluates the taylor polynomial of sin(x)
    for an array of x values.
    :param order: order of the polynomial
    :param x_values: array of values at which we approximate sin(x)
    :return: array of values of the taylor polynomial
    """
    result = x_values.copy()  # Start with the first term of the series
    single_term = x_values.copy()  # This will keep track of the current term
    
    for i in range(1, order + 1):
        single_term *= (-1) * (x_values**2) / ((2 * i + 1) * (2 * i))
        result += single_term
        
    return result

#Plot sin(x), Newton interpolating polynomial, and Taylor polynomial
x_values = np.linspace(0, 5, 50)  # 50 points to plot
true_values = sin_x(x_values)
newton_values = newton_poly(coef, x_nodes, x_values)
taylor_values = taylor_sin(n, x_values)

plt.figure(figsize=(12, 8))
plt.plot(x_values, true_values, label='sin(x)')
plt.plot(x_values, newton_values, label=f'Newton Polynomial n={n}')
plt.plot(x_values, taylor_values, label=f'Taylor Polynomial n={n}')
plt.scatter(x_nodes, y_nodes, label='Interpolation Nodes')
plt.legend()
plt.show()

# Step 5: Plot the errors
plt.figure(figsize=(12, 8))
plt.plot(x_values, np.abs(true_values - newton_values), label='Newton Polynomial Error')
plt.plot(x_values, np.abs(true_values - taylor_values), label='Taylor Polynomial Error')
plt.legend()
plt.show()

'''
The general plots of the polynomials:
I note that the plots for the newton polynomial and the taylor polynomial are completely aligned. 

The error plots:
I started with n = 10 
The error has large fluctuations for small values and large values of x in the case of the newton polynomial. 
However, the error stays pretty consistent with the taylor polynomial. 

When I try with n =20
The error is still very large for x values between 0 and 0.5 and 4.5 and 5 for the newton polynomial. 
However, in between there is an almost perfect alignment between the error values for the taylor polynomial 
and the newton polynomial.

When n = 5
Something completely different occurs. The taylor polynomial suddenly presents an exponentially larger error once 
x is larger than 3. Meanwhile the newton polynomial has a low and consistent error value accross the 
considered interval.

Overall, for large values of n, we get oscillations problems at the edges of the prescibed intervals. However, 
that problem disappears for smaller values of n and we have a newton polynomial with better accuracy compared
to the taylor polynomial.
'''













