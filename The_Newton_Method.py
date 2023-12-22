import numpy as np

#C1 _ Function 
def f_C1(x):
    return x**5 - 9*(x**4) - x**3 + 17*(x**2) - 8*x - 8
def df_C1(x):
    return 5*(x**4) - 36*(x**3) - 3*(x**2) + 34*x - 8

#C2 _ Function 
def f_C2(x):
    return x - np.tan(x)
def df_C2(x):
    return 1 - 1/np.cos(x)**2

#C3 _ Function 
def f_C3(x):
    return x**4 + 2*(x**3) - 7*(x**2) + 3
def df_C3(x):
    return 4*(x**3) + 6*(x**2) - 14*x

#C4 _ Function
def f_C4(x, m):
    return (x - 1)**m
def df_C4(x, m):
    return m*(x - 1)**(m-1)

#C5 _ Function
def f_C5(x):
    return np.exp(-x**2) - np.cos(x) - 1
def df_C5(x):
    return -2*x*np.exp(-x**2) + np.sin(x)

#C6 _ Function 
def f_formula1(x, m, R): #First formula
    return x**m - R
def df_formula1(x,m):
    return m*x**(m-1)

def f_formula2(x,m,R): #Second formula
    return 1 - R/x**m
def df_formula2(x):
    return x

#C7 _ Function 
def f_C7(x):
    return 4*x**3 - 2*x
def df_C7(x):
    return 12*x**2 - 2

def newton (f, df, x, max_iterations, epsilon, delta):
    
    for n in range(max_iterations):
        fx = f(x)
        dfx = df(x)
        
        if np.abs(dfx) < delta:
            return False, 0
        
        h = fx/dfx
        x -= h
        
        if np.abs(h) < epsilon:
            return True, x, n
        print(f"n = {n}, x = {x}, f(x) = {fx}")
    
    return False, x

#Problem C1
print("Problem C1")
print(newton(f_C1, df_C1, 0, 10, 0.000001, 0.000001))
print('\n')
'''
In this problem, we witness a case of cycling where the values
of x oscillate between -1.0, 1.0 and 0.0. This could be due to several
factors such as the tolerance I picked, the root multiplcity, the initial value
that I picked or the characteristics of the function and its derivatives.
'''

#Problem C2
print("Problem C2")
#Generating values around x = 99
x_values = np.linspace(98.5, 99.5, 100)
f_values = f_C2(x_values)

for x, fx in zip(x_values, f_values):
    print(f"| x_c2 = {x:.4f} | f_c2 = {fx: .6f} |")
'''
I note that the function takes big jumps between values of f_C2 as
I generates more and more x values. That is because the function is discontinuous.
I have to generate several values until I spot one for which f_C2 is very close to 0.
I did not manage to spot a value that would converge for the Newton method. For the 
bisection method, I considered a range where f_C2(a)>0 and f_C2(b)<0 but that did not converge
either. It could be because my function is discontinuous and even tho the f values have opposite 
signs, they still do not have a root in between them.
'''    
print(newton(f_C2, df_C2, 99, 10, 0.000001, 0.000001))
print('\n')

#Problem C3
print("Problem 3")
'''
Solving graphically, I was able to find the positive roots of the function 
we have 0.791 and 1.62. 
We must now choose our newton initial guest close to these values to be able to
approximate the root.
We can estimate the number of significant digits by adjusting our error tolerance to 
10^-6 and just hope it respects the number of significant digits by (x_n+1 - x_n).
we set epsilion below 10^-5 and we hope that the number of significant digits will be 
significant to at least 5 digits.
'''  
print(newton(f_C3, df_C3, 0.5, 10, 10**-6, 10**-6))
#This returns true and gives us the 1st root  0.791 within 5 digits accuracy

print(newton(f_C3, df_C3, 1.5, 10, 10**-6, 10**-6))
#This initial value provides us with the second root, accurate to 5 decimal digit
print('\n')

#Problem C4
print("Problem 4")
print(newton(lambda x: f_C4(x, 1), lambda x: df_C4(x, 1), 1.1, 10, 10**(-5), 10**(-5)))
'''
When m=1, it converges with 10 iterations but when I test m=2, it returns false in 10 iterations.
When m=2 converges if I allow 20 iterations but it does not if I leave it at 10 iterations.
We need different iterations for different values of m. That is because of the theorem of 
delta; because when the derivative is zero at the root, we do not have the quadratic convergence 
but we get a linear convergence otherwise. You need more iterations when x is larger than 1
because the c(delta) formula breaks down otherwise.
'''
print('\n')

#Problem C5
print("Problem C5")
print(newton(f_C5, df_C5, 0, 10, 0.000001, 0.000001)) 
'''
When we start at zero, the newton method fails because the derivative becomes smaller than delta.
'''
print('\n')
print(newton(f_C5, df_C5, 1, 10, 0.000001, 0.000001))
'''When we start with 1, the newton method also fails but this time because it just did not converge to a root
by the 10th iteration'''
print('\n')

#Problem C6
print("Problem 6")
print("Formula 1: R = 400, m = 4 and x0 = 10")
print(newton(lambda x : f_formula1(x,4,400), lambda x : df_formula1(x,4), 10, 10, 0.000001, 0.000001))

print("Formula 2: R = 400, m = 4 and x0 = 10")
print(newton(lambda x : f_formula2(x,4,400), df_formula2, 10, 10, 0.000001, 0.000001))

print('\n')

print("Formula 1: R = 400, m = 4 and x0 = 1")
print(newton(lambda x :f_formula1(x,4,400),lambda x : df_formula1(x,4), 1, 10, 0.000001, 0.000001))

print("Formula 2: R = 400, m = 4 and x0 = 1")
print(newton(lambda x : f_formula2(x,4,400),df_formula2, 1, 10, 0.000001, 0.000001))

print('\n')

print("Formula 1: R = 0.5, m = 4 and x0 = 2")
print(newton(lambda x :f_formula1(x,4,0.5),lambda x : df_formula1(x,4), 2, 10, 0.000001, 0.000001))

print("Formula 2: R = 0.5, m = 4 and x0 = 2")
print(newton(lambda x :f_formula2(x,4,0.5), df_formula2, 2, 10, 0.000001, 0.000001))

print('\n')

print("Formula 1: R = 0.5, m = 4 and x0 = 1")
print(newton(lambda x :f_formula1(x,4,0.5), lambda x :df_formula1(x,4), 1, 10, 0.000001, 0.000001))

print("Formula 2: R = 400, m = 4 and x0 = 1")
print(newton(lambda x : f_formula2(x,4,0.5), df_formula2, 10, 1, 0.000001, 0.000001))

print('\n')

print("Formula 1: R = 0.5, m = 4 and x0 = 0.1")
print(newton(lambda x :f_formula1(x,4,0.5),lambda x : df_formula1(x,4), 0.1, 10, 0.000001, 0.000001))

print("Formula 2: R = 0.5, m = 4 and x0 = 0.1")
print(newton(lambda x : f_formula2(x,4,0.5), df_formula2, 0.1, 10, 0.000001, 0.000001))

print('\n')

'''
We notice that formula two never converges to a root given the number of iterations that we
allowed. But formula1 converges in several instances. This is the same as what we said
in problem 4. 
'''

#Problem C7
print(newton(f_C7, df_C7, 0, 10, 10**-5, 10**-5))    
print(newton(f_C7, df_C7, 0.5, 10, 10**-5, 10**-5))   
print(newton(f_C7, df_C7, -0.5, 10, 10**-5, 10**-5))    
'''
The equation is a cubic function therefore, we can find the 3 roots using the newton's method. 
These are some of the points close to (1,0).
'''

    
    