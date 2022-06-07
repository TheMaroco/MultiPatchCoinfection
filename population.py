import numpy as np
from system import system, solve
import matplotlib.pyplot  as  plt


class patch:

    def __init__(self, v0, r, neutralbeta, neutralgamma, neutralk, epsilon):
        self.v0 = np.array(v0)
        self.v = np.array(v0)
        self.r = np.array(r)
        self.nbeta = np.array(neutralbeta)
        self.ngamma = np.array(neutralgamma)
        self.nk = np.array(neutralk)
        self.epsilon = epsilon
        self.m = np.array(r) + np.array(neutralgamma)
        self.R0 = np.array(neutralbeta)/(np.array(r) + np.array(neutralgamma))
    
    def b(self, bi = [0, 0]):
        """Function to generate the non-neutral betas. bi is the strain specific reproduction rates."""
        #To generalize this to more strains -> make bi be a higher dimensional vector, use a for (over i in range(len(bi))) and append with the formula nbeta*(1 + epsilon*bi[i])
        self.beta = np.array([self.nbeta*(1 + self.epsilon*bi[0]), self.nbeta*(1 + self.epsilon*bi[1])])
        return self.beta
    def sgamma(self, gammai = [0, 0]):
        """Function to generate the non-neutral single infectio gammas. gammai is the strain specific clearance rates."""
        self.sgamma = np.array([self.ngamma*(1 + self.epsilon*gammai[0]), self.ngamma*(1 + self.epsilon*gammai[1])])
        return self.sgamma


    def cgamma(self, cgammai = [0, 0, 0, 0]):
        self.cgamma = self.ngamma*np.array([1 + self.epsilon*cgammai[0], 1 + self.epsilon*cgammai[1], 1 + self.epsilon*cgammai[2], 1 + self.epsilon*cgammai[3]])
        return self.cgamma
    #Keep doing this for the remaining parameters
    
    def K(self, k = np.zeros(8)):
        self.K = [self.nk*np.array([1+ self.epsilon*k[0],1 + self.epsilon*k[1]]), self.nk*np.array([1 + self.epsilon*k[2], 1 + self.epsilon*k[3]]), self.nk*np.array([1 + self.epsilon*k[4], 1 + self.epsilon*k[5]]), self.nk*np.array([1 + self.epsilon*k[6], 1 + self.epsilon*k[7]])]
        return self.K
    def neutralmodel(self, t):
        print('im solving')
        return solve(system, t, self.v0, self.r, self.nbeta, self.ngamma, self.ngamma, self.nk, np.array([1, 0.5]), np.array([1, 0.5]))
    


    def describe(self):
        """Function to describe the patch."""
    


patch1 = patch([0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 2, 4, 2, 1, 0.1)
print(patch1.K())





