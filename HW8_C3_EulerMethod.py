import numpy as np
import matplotlib.pyplot as plt

#Part A

# Define the differential equation
def f(t, x):
    return -x + t + 1

# Euler's method function
def euler_method(h, t_final):
    t = np.arange(0, t_final + h, h)
    x = np.zeros(len(t))
    x[0] = 1  # initial condition
    for i in range(0, len(t) - 1):
        x[i + 1] = x[i] + h * f(t[i], x[i])
    return x[-1]

# Approximations for different step sizes
h_values = [0.2, 0.1, 0.05]
approximations = {h: euler_method(h, 5) for h in h_values}
approximations


#Part B

# Exact value at t=5
exact_value = np.exp(-5) + 5

# Calculate the absolute errors for each step size
errors = {h: abs(exact_value - euler_method(h, 5)) for h in h_values}

# Plotting the errors
plt.figure(figsize=(10, 6))
plt.plot(h_values, list(errors.values()), 'o-', color='red', label='Error in Approximation')
plt.xlabel('Step size (h)')
plt.ylabel('Absolute error')
plt.title('Error in Euler Approximations at t=5')
plt.legend()
plt.gca().invert_xaxis()  # Invert x-axis to show decreasing h values
plt.grid(True)
plt.show()
errors


#Part C

# Constants for Theorem 1
L = 1  # Lipschitz constant (assuming |f_x(t, x)| <= L)
M = 2  # Upper bound on the second derivative of the function (assuming |f''(t, x)| <= M)
a = 0  # Initial value of t

# Calculate error bounds using Theorem 1
error_bounds = {
    h: (h * M / (2 * L)) * (np.exp(L * (5 - a)) - 1) for h in h_values
}

# Plotting the actual errors and the error bounds
plt.figure(figsize=(10, 6))
plt.plot(h_values, list(errors.values()), 'o-', color='red', label='Actual Error')
plt.plot(h_values, list(error_bounds.values()), 's-', color='blue', label='Theorem 1 Error Bound')
plt.xlabel('Step size (h)')
plt.ylabel('Error')
plt.title('Comparison of Actual Error and Theorem 1 Error Bound at t=5')
plt.legend()
plt.gca().invert_xaxis()  # Invert x-axis to show decreasing h values
plt.grid(True)
plt.show()
error_bounds

'''
Note from the plot that the error obtained from theorem 1 is indeed an upper bound to the actual error.
'''

#Part D

# Given values
delta = 1e-6
M = 2  # This should be calculated or estimated based on the specifics of your problem

# Calculate the optimal h

optimal_h = np.sqrt((2 * delta) / M)
print(f"The optimal value of h is approximately: {optimal_h}")
'''
Note that I found this formula for h by finding the minimum analytically (visible on HW document)
'''




