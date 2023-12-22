import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, a, b, alpha, h):
    """
    Euler method for approximating solution of the
    initial value problem x' = f(t, x) for t in [a,b]
    with x(a) = alpha using step size h.
    """
    steps = int((b - a) / h)
    t_values = np.linspace(a, b, steps + 1)
    x_values = np.zeros(steps + 1)
    x_values[0] = alpha

    for i in range(steps):
        t = t_values[i]
        x = x_values[i]
        x_values[i + 1] = x + h * f(t, x)

    return t_values, x_values

# Define the right-hand side functions for each differential equation
def rhs_a(t, x):
    return 1 + (t - x)**2

def rhs_b(t, x):
    return t**2 * np.sin(2*t) - 2*t*x

# Solve problem a
a, b, alpha, h = 2, 3, 1, 0.05
t_values_a, x_values_a = euler_method(rhs_a, a, b, alpha, h)

# Solve problem b
a, b, alpha, h = 1, 2, 2, 0.025
t_values_b, x_values_b = euler_method(rhs_b, a, b, alpha, h)

# Plotting the results for problem a
plt.figure(figsize=(10, 5))
plt.plot(t_values_a, x_values_a, label="Euler's method for Problem a", marker='o')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title("Euler's Method Approximation for Problem a")
plt.legend()
plt.grid(True)
plt.show()

# Plotting the results for problem b
plt.figure(figsize=(10, 5))
plt.plot(t_values_b, x_values_b, label="Euler's method for Problem b", marker='o')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title("Euler's Method Approximation for Problem b")
plt.legend()
plt.grid(True)
plt.show()


#Problem C2

# Actual solution for problem a
def actual_solution_a(t):
    return t + 1 / (1 - t)

# Actual solution for problem b
def actual_solution_b(t):
    return (4 + np.cos(2) - np.cos(2 * t)) / (2 * t**2)

# Function to calculate error
def calculate_error(actual_solution, euler_approximation, t_values):
    actual_values = actual_solution(t_values)
    errors = actual_values - euler_approximation
    return errors

# Calculate errors for problem a
errors_a = calculate_error(actual_solution_a, x_values_a, t_values_a)

# Calculate errors for problem b
errors_b = calculate_error(actual_solution_b, x_values_b, t_values_b)

# Plotting the errors for problem a
plt.figure(figsize=(10, 5))
plt.plot(t_values_a, errors_a, label="Error for Problem a", marker='o')
plt.xlabel('t')
plt.ylabel('Error')
plt.title("Error at Each Step for Problem a")
plt.legend()
plt.grid(True)
plt.show()

# Plotting the errors for problem b
plt.figure(figsize=(10, 5))
plt.plot(t_values_b, errors_b, label="Error for Problem b", marker='o')
plt.xlabel('t')
plt.ylabel('Error')
plt.title("Error at Each Step for Problem b")
plt.legend()
plt.grid(True)
plt.show()

