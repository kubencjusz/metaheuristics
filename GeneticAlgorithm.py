from datetime import datetime
import numpy as np

# two helper functions
def binToInt(x):
    # Translate the binary chromosome to real values
    flipped = np.flipud(x)
    idx = np.argwhere(flipped==1).reshape(-1,)
    return (2**idx).sum()
  
def getCoords(population, cel, x_min, x_max):
    # Transform the binary chromosome of size 'cel' into real values of size 2
    coords = np.zeros((population.shape[0], 2))
    for i in range(population.shape[0]):
        for j in range(2): # test for more dimensions in spare time
            coordTemp = binToInt(population[i, (j*cel):((j+1)*cel)])
            # ensuring we are not leaving bounding box
            coords[i, j] = ((x_max[j]-x_min[j])/(2**cel))*coordTemp + x_min[j]
            
    return(coords)


def geneticAlgorithm(f, x_min=[-20, -20], x_max=[20, 20], cel=50,
                     popSize=30, pMut=0.05, maxIter=1000):
    
    '''
    # geneticAlgorithm
    # INPUT
    - f: objective function, R^n -> R
    - x_min: vector of the minimum values of coordinates, 
    - x_max: vector of the maximum values of coordinates
    - cel: coordinate encryption length, number of genes in a single chromosome
    - popSize: size of the population
    - pMut: probability of single genome mutation
    - maxIter: number of generations
    '''
  
    # initializing history
    results = {'x_opt':[], 'f_opt':[], 'x_hist':[], 'f_mean':[], 
               'f_hist':[], 'time':[]}

    # Check the number of dimensions
    d = len(x_min) 
        
    # Initialize population
    population = np.zeros((popSize, cel*d))
      
    for i in range(popSize):
        # .5 chosen arbitrarily
        population[i,] = np.random.uniform(size=cel*d) > .5 
    
    coordinates = getCoords(population, cel, x_min, x_max)
      
    # Calculate fittness of individuals
    objFunction = np.zeros((popSize,))
    for i in range(popSize):
        objFunction[i] = f(coordinates[i,])
    
    # Assign the first population to output 
    results['x_opt'] = coordinates[np.argmin(objFunction),]
    results['f_opt'] = f(coordinates[np.argmin(objFunction),])
      
    # The generational loop
    finished = False
    currIter = 1
    time_0 = datetime.now() # to measure speed
    
    while not finished:
        # Assign the output
        if currIter <= maxIter:
            if results['f_opt'] > f(coordinates[np.argmin(objFunction),]):
                results['x_opt'] = coordinates[np.argmin(objFunction),]
                results['f_opt'] = f(coordinates[np.argmin(objFunction),])
          
            results['f_hist'].append(results['f_opt'])
            results['x_hist'].append(coordinates[np.argmin(objFunction),])
            results['f_mean'].append(np.mean(objFunction))
            results['time'].append((datetime.now() - time_0).microseconds)
        else:
          finished = True
        
        # Translate binary coding into real values to calculate function value
        coordinates = getCoords(population, cel, x_min, x_max)
        
        # Calculate fittness of the individuals
        objFunction = np.zeros((popSize,))
        for i in range(popSize):
            objFunction[i] = f(coordinates[i,])
              
        rFitt = np.divide(min(objFunction),  objFunction,
                          out=np.zeros_like(objFunction), where=objFunction!=0)  # relative fittness
        nrFitt = rFitt / sum(rFitt) # relative normalized fittness (sum up to 1) 
        
        # Selection operator (roulette wheel), analogy to disk
        selectedPool = np.zeros((popSize,))
        for i in range(popSize):
            selectedPool[i] = np.argmin(np.random.uniform(size=1) > np.cumsum(nrFitt))
        
        # Crossover operator (for selected pool)
        nextGeneration = np.zeros((popSize, cel*d))
        for i in range(popSize):
            parentId = int(np.round(np.random.uniform(1, popSize-1, 1)))
            cutId = int(np.round(np.random.uniform(1, d*cel-2, 1)))
            # Create offspring
            nextGeneration[i, :cutId] = population[int(selectedPool[i]), :cutId]
            nextGeneration[i, cutId:(d*cel)] = population[int(selectedPool[parentId]), cutId:(d*cel)]
        
        # Mutation operator
        for i in range(popSize):
            # Draw the genomes that will mutate
            genomeMutId = np.argwhere(np.random.rand(d*cel) < pMut)
            for j in range(len(genomeMutId)):
                nextGeneration[i, genomeMutId[j]] = not nextGeneration[i, genomeMutId[j]] 
        
        # Replace the old population
        population = nextGeneration
        currIter += 1


    return results
  
