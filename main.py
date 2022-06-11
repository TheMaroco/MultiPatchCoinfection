import numpy as np
from model import *
from population import patch, metaPopulation


t = np.linspace(0, 10, 100)


epsilon = 0.1
d = epsilon
M = np.array([[-1, 1], [1, -1]])
patch1 = patch('A', [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1, 6, 2, 1, epsilon)
patch2 = patch('A', [1, 0, 0, 0, 0, 0, 0], 0.1, 4, 2, 1, epsilon)
patch3 = patch('A', [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1, 2, 2, 1, epsilon)


patches = [patch1, patch2]

metapop = metaPopulation(patches, d, M)
print(patch1.R0, patch2.R0)




plot(metapop.measures(t), t)
plt.show()

