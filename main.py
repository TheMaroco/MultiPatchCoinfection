import numpy as np
from model import *
from population import patch, metaPopulation


t = np.linspace(0, 100, 100)


epsilon = 0.1
d = epsilon
M = np.array([[-1, 1], [1, -1]])
patch1 = patch('A', [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1, 6, 2, 1, epsilon)
patch1.define_b([1, 2]) 
#patch1.define_sgamma([5, 0.8])
#patch1.define_cgamma([0.3,0.1,0.2,0.1])

patch2 = patch('B', [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1, 5, 2, 1, epsilon)
#patch2.define_b([0.1, 0.4])

patch3 = patch('A', [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1, 2, 2, 1, epsilon)


patches = [patch1, patch2]

metapop = metaPopulation(patches, 0, M)
print(patch1.R0)
print(patch2.R0)
print(metapop.meanR0())




plot(metapop.measures(t), t)
plt.show()

