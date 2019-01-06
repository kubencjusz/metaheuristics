import numpy as np

# defining testing functions, R2n -> R:
# functions taken from https://en.wikipedia.org/wiki/Test_functions_for_optimization

def simple(vec):
    return vec[0]**2 + 10*vec[1]**2

def schaffer(vec):
    #assert all([-100<x<100 for x in vec])
    x = vec[0]
    y = vec[1]
    return .5 + (np.sin(x**2-y**2)**2 - .5) / (1+.001*(x**2+y**2))**2

def himmelblau(vec):
    #assert all([-5<x<5 for x in vec])
    return (vec[0]**2+vec[1]-11)**2 + (vec[0]+vec[1]**2 -7)**2

def levi(vec):
    #assert all([-10<x<10 for x in vec])
    x=vec[0]
    y=vec[1]
    pi = np.pi
    return np.sin(3*pi*x)**2 + (x-1)**2 *(1+np.sin(3*pi*y)**2)+(y-1)**2 *(1+np.sin(2*pi*y)**2)
    
# visualizing functions
sa = 
pso = 
gen = 

# comparing execution time of different algorithms