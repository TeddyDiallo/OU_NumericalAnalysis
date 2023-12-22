import numpy as np
"""
Created on Tue Aug 29 18:09:53 2023

@author: Teddy Diallo
"""

"Problem 1"
def find_maximum(x, y, z):
    if x > y and x > z :
        return x
    elif y > x and y > z :
        return y
    else:
        return z
    
# Test 1
if __name__=='__main__':
    x = [1,2,3]
    y = [3,2,1]
    z = [1,3,2]
    for i in range(len(x)):
        print("The maximum value between ", x[i], ",",y[i], " and ",z[i],
              " is ", find_maximum(x[i], y[i], z[i]))

print("\n")

"Problem 2 : Sum of the squares of consecutive integers"

def long_sum_i_plus_one_squared(n, m):
    ans = 0 
    for i in range(n, m+1):
        ans += (i + 1)**2
    return ans

# Test 2
if __name__ == '__main__':
    n = [0, -3]
    m = [3, 2]
    for i in range(len(n)):
        print("The sum of squares from ", n[i], " to ", m[i], " is ",
              long_sum_i_plus_one_squared(n[i], m[i]))

print("\n")

"Problem 3 "
def cen_to_far (np_array):
    return (np_array * (9/5) ) + 32

# Test 3 
temp_list = [1,3.4,75.5,100.3, -35.2]
np_temp_list = np.array(temp_list)
print(cen_to_far(np_temp_list))

print("\n")

"Problem 5"
def num_even(np_array):
    even = np_array%2
    return len(even[even == 0])

#Test 4
num_list1 = [1,10,1,20,4,30,8,70]
num_list2 = [20,10,3,-4,70,80]

np_num_list1 = np.array(num_list1)
np_num_list2 = np.array(num_list2)

print("The number of even number in list 1 is ", num_even(np_num_list1))
print("The number of even number in list 2 is ",num_even(np_num_list2))

print("\n")
    
"Problem 6"

def common_values_long(arr1, arr2):
    ans=[]
    for val in arr1:
        if val in arr2 and val not in ans:
            ans.append(val)
    return ans

#Test 6
print("The common values between list 1 and list 2 are ",
      common_values_long(np_num_list1, np_num_list2))

print("\n")

"Problem 7 - Finding solutions of a polynomial ax^2 + bx + c = 0"

def solve_eqn(a,b,c):
    if a == 0 :
        if b == 0:
            print("This is not allowed")
            return []
        else:
            print("a equal 0")
            return -c/b
    else:
        delta = b**2 - 4*a*c
        if delta > 0:
            x_1 = (-b + delta**0.5) / 2*a
            x_2 = (-b - delta**0.5) / 2*a
            print("a is different of zero and delta is greater than 0")
            return [x_1, x_2]
        elif delta == 0:
            print("a is different of zero and delta is equal 0")
            return -b/(2*a)
        else:
            print("a is different of zero and delta is less than 0")
            return []

#Test 7
print(solve_eqn(1, -3, 2),"\n")
print(solve_eqn(1, 0, -1),"\n")
print(solve_eqn(1, 0, 1),"\n")
print(solve_eqn(0, -3, 2))            



"Problem 8"


















