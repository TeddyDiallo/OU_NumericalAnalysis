import numpy as np
import matplotlib.pyplot as plt


def taylor_e(order, x):
    """
    Evaluates the taylor polynomial of e^x
    :param order: of the polynomial
    :param x: value at which we approximate e^x
    :return: the value of the taylor polynomial
    """

    single_term = 1
    result = single_term
    for i in range(1, order + 1):
        single_term = single_term * x / i
        result += single_term
    return result


def find_smallest_order(epsilon=1e-6):
    n_points = 1000
    x_values = np.linspace(0, 2, n_points)
    
    m = 1  # Starting with the first order
    max_error = epsilon + 1  # Initialize max_error to be greater than epsilon
    
    while max_error > epsilon:
        y_values = np.zeros(n_points)
        for n in range(n_points):
            y_values[n] = taylor_e(m, x_values[n])
        
        max_error = np.max(np.abs(y_values - np.exp(x_values)))
        m += 1
    
    return m - 1  # Subtract 1 because we want the smallest order



if __name__ == '__main__':

    # Evaluate the Taylor polynomial of order 3 on 1000 uniformly spaced points in the interval [0,2]
    # Compare with the exponential function implemented in mumpy

    # Set the points in the interval [0,1]
    n_points = 1000
    x_values = np.linspace(0, 2, n_points)
    y_values = np.zeros(1000)

    # Set the order of the polynomial and compute at the given points
    m = 7
    for n in np.arange(n_points):
        y_values[n] = taylor_e(m, x_values[n])

    # Plot the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))
    plt.plot(x_values, y_values,linewidth=3, label=("Taylor Polynomial of order %d" % m))
    plt.plot(x_values, np.exp(x_values),linewidth=3, label="Numpy value of the exponential")
    plt.title('Taylor series')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()



    # Plot the absolute differences between the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))
    plt.plot(x_values, np.abs(y_values - np.exp(x_values)), linewidth=3)
    plt.title('Difference between Taylor series and Numpy')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
    
    
    smallest_order = find_smallest_order()
    print(f"The smallest order for which all errors are bounded by 10^-6 is: {smallest_order}")
    
    """
    By testing different order of the polynomial, I noticed that for smaller order, there is a bigger difference
    between the real value of e^x produced by numpy and the approximation. However, for larger order for the 
    polynomial, the difference become smaller and smaller. By order 5, the approximation and numpy's representation are 
    close to being fully merged. 
    With the find_smallest_function that I added, I was able to do iterations until finding the order for which all the 
    errors are bounded by 10^-6 which is 13. The representation of our approximation at order 13 shows that the approximation
    and the precise value of e^x are fully merged.
    """
    
    