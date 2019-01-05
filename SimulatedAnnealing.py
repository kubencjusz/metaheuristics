from datetime import datetime
import numpy as np

def simulatedAnnealing(f, x, alpha=.99, t=10000, delta=.1, maxIter=1000):
        
    ''' Simulated Annealing Algorithm (objective: find global minimum):
    f - objective function, R^n -> R
    x - inital solution, starting point, R^n
    alpha - annealing schedule parameter
    t - inital temperature
    delta - neighborhood radius
    maxIter - maximum no. of iterations  ''' 
    
    # initializing starting parameters
    results = {'x_opt':x, 'f_opt':f(x), 'x_hist':[x], 'time':[0],
               'f_hist':[f(x)], 'temp':[t], 'transProb':[0]}
    
    currIter = 1
    finished = False
    x_s = x    
        
    while not finished:
        
        time_0 = datetime.now() # to measure speed
        
        # x_c - uniformly drawing a candidate solution from neighborhood of x_s
        unif = np.random.rand(len(x_s))
        x_c = x_s + (-delta + 2*delta*unif)
    
        # A - calculating Metropolis activation function
        A = np.minimum(1, np.exp(-(f(x_c) - f(x_s)) / t))
    
        # transition to candidate solution
        if bool(np.random.rand(1) < A):
            x_s = x_c
        
        # temperature update for the next iteration
        t = alpha * t
    
        if currIter < maxIter:
            
            # if better solution, update results
            if f(x_s) < f(results['x_opt']):
                results['x_opt'] = x_s
                results['f_opt'] = f(x_s)
            
            # update results history
            results['x_hist'].append(x_s)
            results['f_hist'].append(f(x_s))
            results['temp'].append(t)
            results['transProb'].append(A)
        else:
            finished = True
        
        if currIter % 250 == 0:
            print(f"f_opt after {currIter} iterations: {results['x_opt']} \n")
        
        currIter += 1

    return results



# testing a function
res=simulatedAnnealing(f=schaffer, x=[14, 10], delta=.3)    

from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iks = np.linspace(-20, 20, 100)
igr = np.linspace(-20, 20, 100)
x = []
y = []
vals = []

for pair in product(iks, igr):
    x.append(pair[0])
    y.append(pair[1])
    vals.append(schaffer([pair[0], pair[1]]))
    
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, vals, cmap=plt.cm.viridis, linewidth=0.2)
plt.show()

# contour plot on range [-50, 50]
path_x = [x[0] for x in res['x_hist']]
path_y = [y[1] for y in res['x_hist']]


# contourplot with line
fig, ax = plt.subplots()
cmap = plt.contourf(iks, igr, np.array(vals).reshape(100,100))
fig.colorbar(cmap)
ax.plot(path_x, path_y, 'r')
ax.scatter(14,10, s=20, c='k')
plt.show(fig)



