import numpy as np

#Q2
def poly_1(x):
    return np.sqrt(x) - np.cos(x)

#Q3
def poly_2(x):
    return 9*(x**4) + 18*(x**3) + 38*(x**2) - 57*x - 1

#Q4
def poly_3(x):
    return x**3 - 3*x + 1

#Q5
def poly_4(x):
    return np.tan(x) - x

#Q6
def poly_5(x):
    return x**2 -3

#Q7. a
def poly_7(x):
    return x + 0.9

#Q7. b
def poly_8(x):
    return x**2 - 1.8*x + 0.71

#This function allows to find the number of steps to reach the desired accuracy 
def num_step(a, b, tolerance):
    x = np.log2((b-a)/tolerance)
    return x

def bisection(f, a, b, max_iter, tolerance):
    """
    Approximates a root p in [a,b] of the function f with error smaller than 
    a given tolerance.
    Note that f(a)f(b) has to be negative for this method to work
    :param f: function whose root is approximated
    :param a: left bound on the position of the root
    :param b: right bond on the position of the root
    :param max_iter: maximum number of iteration to reach the tolerance
    :param tolerance: maximal allowed error

    :return: converged - flag indicating if the method reached the precision
             root - approximation of the root
    """

    fa = f(a)
    fb = f(b)

    if np.sign(fa) == np.sign(fb):
        return False, 0
    
    if num_step(a, b, tolerance) > max_iter: #C1-Checks if the precision can be achieved
        return False, 0

    error = b - a
    for n in np.arange(max_iter):
        error /= 2
        p = a + error
        fp = f(p)

        if fp == 0 or error < tolerance:
            return True, p, n #Return n so that I can analyse the efficiency for C7
        if np.sign(fa) == np.sign(fp):
            a = p
            fa = fp

    return False, 0


def false_position(f, a, b, max_iter, tolerance):
    fa = f(a)
    fb = f(b)
    
    if np.sign(fa) == np.sign(fb):
        return False, 0
    
    p_prev = a
    for n in range(max_iter):
        p = (a*fb - b*fa) / (fb - fa)
        fp = f(p)
        
        if fp == 0 or np.abs(p - p_prev) < tolerance:
            return True, p , n #I made it return n so that I can check the efficiency for C7
        p_prev = p
        
        if np.sign(fa) == np.sign(fp):
            a = p
            fa = fp
        else:
            b = p
            fb = fp
            
    return False, p
    
#C2
n = num_step(0, 1, 10**-6)
n = np.ceil(n)
print(bisection(poly_1, 0, 1, n, 10**-6))

#C3
print(bisection(poly_2, 0, 1, n, 10**-6))

#C4  
print(bisection(poly_3, 0, 1, n, 10**-6))
'''
We were not given an interval but after testing with [0,1], we note that 
the bisection method works. As long as the product of the evaluation of the 
intervals at the bounds is negative, the bisection method will work 
even if it does not guarantee the truth of the root that we find.
'''

#C5
print(bisection(poly_4, 4, 5, n, 10**-6))# We get False
print(bisection(poly_4, 1, 2, n, 10**-6))# We get True

print(np.sign(poly_4(4)) == np.sign(poly_4(5)))#True => signs are equal
print(np.sign(poly_4(1)) == np.sign(poly_4(2)))#False => signs are not the same
'''
We notice that we can not find a root using the bisection method on one interval but we can on 
the other. This is because the primary condition for the bisection method to work is not respected. 
The product of the values needs to be negative. Meaning our interval needs to contain zero which
will suggest that we might have crossed the x-axis and allow for the bisection method to work! 
'''
#Q5
'''
Finding sqrt(3) is the same as solving for the positive root of the quadratic x^2 - 3 = 0. To be able 
to accurately evaluate sqrt(3), we can pick an interval in the positive quadrant and apply the 
bisection method
'''
print(bisection(poly_5, 1, 3, 100, 10**-6))
'''
I made the evaluation with an arbitrary 10^-6 accuracy and gave room for as much iteration 
as possible to find the best approximation. Using the calculator, we note that the value, 
from the bisection is accurate to at least five digits. 
'''
#Q7 a.
print(bisection(poly_7, -1, 1, 100, 10**-6), "Test1 for bissection") #We find the root after 20 iterations
print(false_position(poly_7, -1, 1, 100, 10**-6), "Test1 for false position") #We find the root after 1 iteration  

#Q7 b.
print(bisection(poly_8, -1, 1, 100, 10**-6), "Test2 for bissection") #We find the root after 20 iterations
print(false_position(poly_8, -1, 1, 100, 10**-6), "Test2 for false position") #We find the root after 38 iterations
'''
It seems that with a linear function the false position method works better while with quadratic functions, 
the bisection works better.
THEOREM ON THE A^K(B-A) FOR FALSE POSITION. In bisection A is 1/2 - based on 1st n 2nd derivation
'''
