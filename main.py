import numpy as np
from model import *
from population import patch, metaPopulation


t = np.linspace(0, 100, 100)
initial_conditions1 = [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
initial_conditions2 = [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
neutral_r = 1
neutral_beta = 6

epsilon = 0.01
d = epsilon
M = np.array([[-1, 1], [1, -1]])
patch1 = patch('A', initial_conditions1, 1, neutralbeta = 6, neutralgamma = 2, neutralk = 1, epsilon = epsilon)
b = [0.1, 0.6]
patch1.define_beta(b) 
#patch1.define_sgamma([5, 0.8])
#patch1.define_cgamma([0.3,0.1,0.2,0.1])
#alpha = [0.1, 0.4, 0.3, 0.1]
#patch1.define_K(alpha)

patch2 = patch('B', initial_conditions2, 1, neutralbeta = 5, neutralgamma = 2, neutralk = 1, epsilon = epsilon)
b = [0.1, 0.1]
patch2.define_beta(b)
#alpha = [0.1, 0.5, 0.5, 0.1]
#patch2.define_K(alpha)


patch1.describe()
patch2.describe()

patches = [patch1, patch2]
metapop = metaPopulation(patches, d, M)
print('Patch A R0:', patch1.R0)
print('Patch B R0:', patch2.R0)
print('Average R0', metapop.meanR0())
print('Average lambda1_2', metapop.meanInvasionfitness()[0])
print('Average lambda2_1', metapop.meanInvasionfitness()[1])
print('w:', metapop.measures(t)['w'])

print('error in the approximation:', metapop.measures(t)['error'])

plot(metapop.measures(t), t)
plt.show()



errors = []

epsilons = np.linspace(0.001, 1, 100)
for e in epsilons:
    patch1 = patch('A', [0.4, 0.3, 0.1, 0.0, 0.0, 0.1, 0.1], 1, 6, 2, 1, e)
    patch1.define_beta(b)
    patch2 = patch('B', [0.4, 0.3, 0.1, 0.0, 0.0, 0.1, 0.1], 1, 5, 2, 0.1, e)
    patch2.define_beta(b)
    patches = [patch1, patch2]
    metapop = metaPopulation(patches, d, M)
    measures = metapop.measures(t)
    
    errors.append(measures['error'])


plt.plot(epsilons, errors)
plt.show()


