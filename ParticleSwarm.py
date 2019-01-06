from datetime import datetime
import numpy as np

def PSO(f, swarm_size=20, max_iter=200, x_min=[-20,-20], x_max=[20,20],
        c1=1, c2=1, omega=.5):

    '''
    # Particle Swarm Optimization
    # INPUT
    - f: objective function, R^n -> R,
    - x_min: vector of the minimum values of coordinates, 
    - x_max: vector of the maximum values of coordinates,
    - swarm_size: number of particles in the swarm
    - max_iter: maximum number of iterations
    - c1: weight of personal best result of a particule
    - c2: weight of global best result of a swarm
    - omega: weight of current velocity
    '''
    
    dim = len(x_min) 
    r = np.empty((swarm_size, dim))
    s = np.empty((swarm_size, dim))
    
    # initializing swarm
    swarm = np.empty((swarm_size, dim))
    velocity = np.empty((swarm_size, dim))
    p_best = np.empty((swarm_size, dim))
    swarm_result = np.empty((swarm_size, 1))
    
    # for each particle and each dimension initialize starting points
    for i in range(swarm_size):
        for j in range(dim):
            swarm[i, j] = np.random.uniform(low=x_min[j], high=x_max[j], size=1)
        velocity[i,:] = np.random.uniform(low=0, high=1, size=dim)
        p_best[i,:] = swarm[i,:]
        swarm_result[i] = f(swarm[i,:])
        g_best = swarm[np.argmin(swarm_result),:] # updating global best solution
    
    results = {'x_opt': [g_best], 'f_opt':[f(g_best)], 'x_hist':[g_best],
               'f_hist':[f(g_best)], 'time':[0]}
    
    pocz_iter = datetime.now()
    
    for k in range(max_iter):
        
        for m in range(swarm_size):
            r[m, :] = np.random.uniform(low=0, high=1, size=dim)
            s[m, :] = np.random.uniform(low=0, high=1, size=dim)
            
            # calculating components of new velocity
            old_vel = omega * velocity[m,:] # old velocity comp.
            best_pers_vel = c1 * r[m,:] * (p_best[m,:]-swarm[m,:]) # personal best comp.
            best_glob_vel = c2 * s[m,:] * (g_best-swarm[m,:]) # global best comp.
            # calculating new velocity
            velocity[m, :] = old_vel + best_pers_vel + best_glob_vel 
            # moving a particle in a new direction
            swarm[m, :] += velocity[m, :]
            
            # updating best solution for particle m
            if f(swarm[i,]) < f(p_best[i,]):
                p_best[m, :] = swarm[m, :]
            swarm_result[m] = f(swarm[m, :])
        
        # updating global best particle in iteration k
        if min(swarm_result)[0] < f(g_best):
            g_best = swarm[np.argmin(swarm_result), :]
        
        # saving history
        if results['f_opt'] > f(g_best):
            results['x_opt'] = swarm[np.argmin(swarm_result),:]
            results['f_opt'] = f(swarm[np.argmin(swarm_result),:])
        
        results['x_hist'].append(g_best)
        results['f_hist'].append(f(g_best))
        results['time'].append((datetime.now()-pocz_iter).microseconds)
        
    return results

