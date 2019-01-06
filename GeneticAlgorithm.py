import numpy as np

# zmienic funkcje seq i which rev

def intbin(x):
    # Translate the binary coding to real values numbers
    flipped = np.flipud(x)
    idx = np.argwhere(flipped==1).reshape(-1,)
    return (2**idx).sum()
  
def getCoordinates(population, cel, x_min, x_max):
    
    # Transform the binary coding into coordinates
    coordinates = np.zeros((population.shape[0], 2))
    for i in range(population.shape[0]):
        for j in range(2): # tu wlasciwie powinno byc po wszystkich wymiarach
            coordinatesTemp = intbin(population[i, (j*cel):((j+1)*cel)])
            # ensuring we are not leaving bounding box
            coordinates[i, j] = ((x_max[j]-x_min[j])/(2**cel))*coordinatesTemp + x_min[j]
            
    return(coordinates)


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
  
    # initializing starting values
    results = {'x_opt':[], 'f_opt':[], 'x_hist':[], 'f_mean':[], 
               'f_hist':[], 'time':[]}

    # Check the number of dimensions
    d = len(x_min) 
        
    # Initialize population
    population = np.zeros((popSize, cel*d))
      
    for i in range(popSize):
        population[i,] = np.random.uniform(size=cel*d) > .5 # .5 chosen arbitrarily
    
    coordinates = getCoordinates(population, cel, x_min, x_max)
      
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
    while not finished:
        # Assign the output
        if currIter <= maxIter:
            if results['f_opt'] > f(coordinates[np.argmin(objFunction),]):
                results['x_opt'] = coordinates[np.argmin(objFunction),]
                results['f_opt'] = f(coordinates[np.argmin(objFunction),])
          
            results['f_hist'].append(results['f_opt'])
            results['x_hist'].append(coordinates[np.argmin(objFunction),])
            results['f_mean'].append(np.mean(objFunction))
        else:
          finished = True
        
        # Translate binary coding into real values to calculate function value
        coordinates = getCoordinates(population, cel, x_min, x_max)
        
        # Calculate fittness of the individuals
        objFunction = np.zeros((popSize,))
        for i in range(popSize):
            objFunction[i] = f(coordinates[i,])
              
        rFitt = min(objFunction) / objFunction # Relative Fittness
        nrFitt = rFitt / sum(rFitt) # Relative Normalized (sum up to 1) Fittness
        
        # Selection operator (Roulette wheel), analogy to disk
        selectedPool = np.zeros((popSize,))
        for i in range(popSize):
            selectedPool[i] = np.argmin(np.random.uniform(size=1) > np.cumsum(nrFitt))
        
        # Crossover operator (for selected pool)
        nextGeneration = np.zeros((popSize, cel*d))
        for i in range(popSize):
            parentId = int(np.round(np.random.uniform(1, popSize, 1)))
            cutId = int(np.round(np.random.uniform(1, d*cel-1, 1))) # Do not exceed the matrix sizes
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
  
  
# testing
