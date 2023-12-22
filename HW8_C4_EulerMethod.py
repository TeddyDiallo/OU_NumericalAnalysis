import numpy as np
import matplotlib.pyplot as plt

# Constants for the problem
C = 0.3  # farads
R = 1.4  # ohms
L = 1.7  # henries

# The impressed voltage function and its derivatives
def E(t):
    return np.exp(-0.06 * t) * np.sin(2 * t - np.pi)

def dE_dt(t):
    return np.exp(-0.06 * t) * (0.06 * np.sin(np.pi - 2 * t) + 2 * np.cos(2 * t - np.pi))

def d2E_dt2(t):
    return np.exp(-0.06 * t) * (-4 * np.sin(2 * t - np.pi) - 0.12 * np.cos(2 * t - np.pi)) + 0.0036 * np.exp(-0.06 * t) * np.sin(2 * t - np.pi)

# Differential equation to solve: di/dt = C*d2E/dt2 + (1/R)*dE/dt + (1/L)*E
def di_dt(t, i):
    return C * d2E_dt2(t) + (1/R) * dE_dt(t) + (1/L) * E(t)

# Euler's method for ODE
def euler_method(f, t0, i0, t_end, steps):
    h = (t_end - t0) / steps
    t_values = np.linspace(t0, t_end, steps + 1)
    i_values = np.zeros(steps + 1)
    i_values[0] = i0

    for j in range(steps):
        i_values[j + 1] = i_values[j] + h * f(t_values[j], i_values[j])

    return t_values, i_values

# Solve the ODE
t0 = 0
i0 = 0
t_end = 10
steps = 1000  # This gives a step size of h = 0.01, which is smaller than the 0.1j step
t_values, i_values = euler_method(di_dt, t0, i0, t_end, steps)

# Plotting the result
plt.plot(t_values, i_values, label='Current i(t)')
plt.xlabel('Time t')
plt.ylabel('Current i(t)')
plt.title('Current in an RLC Circuit')
plt.legend()
plt.grid(True)
plt.show()
