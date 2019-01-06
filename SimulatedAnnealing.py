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
    time_0 = datetime.now() # to measure speed
    
    while not finished:
        
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
            results['time'].append((datetime.now() - time_0).microseconds)

            results['transProb'].append(A)
        else:
            finished = True
        
        if currIter % 250 == 0:
            print(f"f_opt after {currIter} iterations: {results['x_opt']} \n")
        
        currIter += 1

    return results

