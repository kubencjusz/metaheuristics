from datetime import datetime
import numpy as np

def PSO(f, swarm_size, max_iter, x_min, x_max, c1, c2, omega):

    dim = len(x_min)
    r = np.empty((swarm_size, dim))
    s = np.empty((swarm_size, dim))
    
    # initial swarm
    swarm = np.empty((swarm_size, dim))
    velocity = np.empty((swarm_size, dim))
    p_best = np.empty((swarm_size, dim))
    swarm_result = np.empty((swarm_size, 1))
    
    for i in range(swarm_size):
        for j in range(dim):
            swarm[i, j] = np.random.uniform(low=x_min[j], high=x_max[j], size=1)
            velocity[i,:] = np.random.uniform(low=0, high=1, size=dim)
            p_best[i,:] = swarm[i,:]
            swarm_result[i] = f(swarm[i,:])
            g_best = swarm[np.argmin(swarm_result),:]
    
    result = {'x_opt': g_best, 'f_opt':f(g_best), 'x_hist':g_best, 'f_hist':f(g_best), 'time':None}
    
    # iterujemy
    for k in range(max_iter):
        pocz_iter = datetime.now()
        for m in range(swarm_size):
            r[m,:] = np.random.uniform(low=0, high=1, size=dim)
            s[m,:] = np.random.uniform(low=0, high=1, size=dim)
            velocity[m,:] = omega*velocity[m,:] + c1*r[m,:]*(p_best[m,:]-swarm[m,:]) + c2*s[m,:]*(g_best-swarm[m,:])
            if f(swarm[i,])<f(p_best[i,]):
                p_best[m,:] = swarm[m,:]
            else:
                p_best[m,:]<-p_best[m,:]
            
        
        swarm_result[m] = f(swarm[m,:])
        if min(swarm_result)<f(g_best):
            g_best = swarm[np.argmin(swarm_result),:]

        if result['f_opt'] > f(g_best):
            result['x_opt'] = swarm[np.argmin(swarm_result),:]
            result['f_opt'] = f(swarm[np.argmin(swarm_result),:])
        
        result['x_hist'] = np.vstack([result['x_hist'], (g_best)])
        result['f_hist'] = np.vstack([result['f_hist'], f(g_best)])
        
        czas_iter = datetime.now()-pocz_iter
        result['time'] = np.vstack([result['time'],czas_iter])
        
    return result


# testy
ub_iter = 1000
omega = 0.5
c1 = 1
c2 = 1
swarm_size = 20
maxIter = 100
x_min = [-20,-20]
x_max = [20,20]

# Schaffer function
def myFun(x):
    return(0.6 + ((np.sin(x[0]**2-x[1]**2))**2-0.5)/((1+0.001*(x[0]**2+x[1]**2))**2))



pso = PSO(f=myFun, swarm_size=20, max_iter=ub_iter, x_min=x_min, x_max=x_max, c1=1, c2=1, omega=0.5)

