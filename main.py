import numpy as np
import pandas as pd
from itertools import product
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# import functions
os.chdir(os.path.join(os.getcwd(), "Desktop/PyProjects/metaheuristics"))

from SimulatedAnnealing import simulatedAnnealing
from GeneticAlgorithm import geneticAlgorithm
from ParticleSwarm import PSO


# defining testing functions, R^2 -> R:
# functions taken from:
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

# global minimum: 0.0 (0,0)
def simple(vec):
    return vec[0]**2 + 10*vec[1]**2

# global minimum: 0.2925 (0, 1.25313)
def schaffer(vec):
    #assert all([-100<x<100 for x in vec])
    x = vec[0]
    y = vec[1]
    return .5 + (np.sin(x**2-y**2)**2 - .5) / (1+.001*(x**2+y**2))**2

# global minimum: 0.0 [(3,2), (-2.8051, 3,1313), (-3,7793, -3,2831), (3.5844, -1.8481)]
def himmelblau(vec):
    #assert all([-5<x<5 for x in vec])
    return (vec[0]**2+vec[1]-11)**2 + (vec[0]+vec[1]**2 -7)**2

# global minimum: 0.0 (1, 1)
def levi(vec):
    #assert all([-10<x<10 for x in vec])
    x=vec[0]
    y=vec[1]
    pi = np.pi
    return np.sin(3*pi*x)**2 + (x-1)**2 *(1+np.sin(3*pi*y)**2)+(y-1)**2 * \
           (1+np.sin(2*pi*y)**2)

# global minimum: -1 (pi, pi)
def easom(vec):
    x=vec[0]
    y=vec[1]
    pi=np.pi
    return -np.cos(x)*np.cos(y)*np.exp(-((x-pi)**2 + (y-pi)**2))

# SET PARAMETERS
F = himmelblau
X_MIN = [-20,-20]
X_MAX = [20,20]
N_ITER = 1000
# for PSO
SWARM_S = 20
C1 = 1
C2 = 1
OMEGA = .5
# for SA
X_INIT = [5, -3]
DELTA = .3
# for GA
CEL = 50
POP_SIZE = 1000


# Optimize
sa = simulatedAnnealing(f=F, x=X_INIT, delta=DELTA, maxIter=N_ITER)
pso = PSO(f=F, swarm_size=SWARM_S, max_iter=N_ITER, x_min=X_MIN, x_max=X_MAX,
          c1=C1, c2=C2, omega=OMEGA)
gen = geneticAlgorithm(f=F, x_min=X_MIN, x_max=X_MAX, maxIter=N_ITER)


# MAKING VISUALISATIONS

# prepare grid of values
GRID_DENS = 100

iks = np.linspace(X_MIN[0], X_MAX[0], GRID_DENS)
igr = np.linspace(X_MIN[1], X_MAX[1], GRID_DENS)

x = []
y = []
vals = []

for pair in product(iks, igr):
    x.append(pair[0])
    y.append(pair[1])
    vals.append(F([pair[0], pair[1]]))

# surface plot 3D
fig_surf = plt.figure()
ax_surf = fig_surf.gca(projection='3d')
ax_surf.plot_trisurf(x, y, vals, cmap=plt.cm.viridis, linewidth=0.2)
plt.show()

# contour plot 2D
def draw_path(hist = sa, f_opt_hist=False):
    
    if f_opt_hist:
        new_f_hist = pd.Series(hist['f_hist']).cummin().values
        idx = [np.argwhere(hist['f_hist']==x)[0][0] for x in new_f_hist]
        hist['x_hist'] = [hist['x_hist'][i] for i in idx]
        hist['f_hist'] = new_f_hist
        

    path_x = [x[0] for x in hist['x_hist']]
    path_y = [y[1] for y in hist['x_hist']]

    fig, ax = plt.subplots()
    cmap = plt.contourf(iks, igr, np.array(vals).reshape(GRID_DENS,-1),
                        levels = 40)
    fig.colorbar(cmap)
    ax.plot(path_x, path_y, 'r', linewidth=0.7)
    ax.scatter(hist['x_hist'][0][0], hist['x_hist'][0][1], s=15, c='k')
    plt.show(fig)

draw_path(hist = gen)
draw_path(hist = sa, f_opt_hist=False)
draw_path(hist = pso, f_opt_hist=True)

# DRAWING ANIMATION
def draw_anim(hist = sa, save_to_disk = False,
              file_name = 'animated_path.gif'):
    
    # drawing static contourplot of function
    fig, ax = plt.subplots()
    cmap = plt.contourf(iks, igr, np.array(vals).reshape(GRID_DENS, -1))
    fig.colorbar(cmap)
    ax.scatter(hist['x_hist'][0][0], hist['x_hist'][0][1], s=20, c='k')
    line, = ax.plot([], [], 'r-', linewidth=1)
    
    # defining update function
    def update_gif(i):
        x_tmp = [x[0] for x in hist['x_hist']]
        y_tmp = [x[1] for x in hist['x_hist']]
        
        line.set_xdata(x_tmp[:i])
        line.set_ydata(y_tmp[:i])
        return line,
        
    ani = FuncAnimation(fig, update_gif, interval=100, blit=True,
                        frames=np.arange(1, len(hist['x_hist'])))
    
    if save_to_disk:
        ani.save(file_name) # writer='imagemagick'
    else:
        plt.show()


# drawing execution time
draw_path(pso)
pso['f_hist']
