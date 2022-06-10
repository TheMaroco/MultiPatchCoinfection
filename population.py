import numpy as np
from scipy import integrate
from system import system, neutralsystem
import matplotlib.pyplot  as  plt


class patch:

    def __init__(self, name, v0, r, neutralbeta, neutralgamma, neutralk, epsilon):
        self.name = name
        self.v0 = np.array(v0)
        self.v = np.array(v0)
        self.r = np.array(r)
        self.nbeta = np.array(neutralbeta)
        self.b = self.nbeta
        self.ngamma = np.array(neutralgamma)
        self.sgamma = self.ngamma
        self.cgamma = self.ngamma*np.ones(4)
        self.nk = np.array(neutralk)
        self.k = self.nk
        self.p = self.q = 0.5*np.ones(4)
        self.epsilon = epsilon
        self.m = np.array(r) + np.array(neutralgamma)
        self.R0 = np.array(neutralbeta)/(np.array(r) + np.array(neutralgamma))
        self.nSstar = 1/self.R0
        self.nTstar = 1 - self.nSstar
        self.nIstar = self.nTstar/(1 + self.nk*(self.R0 -1))
        self.nDstar = self.nTstar - self.nIstar
        self.mu = self.nIstar/self.nDstar
        self.detP = np.linalg.det(np.array([[2*self.nTstar, self.nIstar], [self.nDstar, self.nTstar]]))
        self.theta1 = 2*self.nbeta*self.nSstar*self.nTstar**2/self.detP
        self.theta2 = self.ngamma*self.nIstar*(self.nIstar + self.nTstar)/self.detP
        self.theta3 = self.ngamma*self.nTstar*self.nDstar/self.detP
        self.theta4 = 2*self.m*self.nTstar*self.nDstar/self.detP
        self.theta5 = self.nbeta*self.nTstar*self.nIstar*self.nDstar/self.detP
        self.lambda1_2 = 0
        self.lambda2_1 = 0
    
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

    def p(self, p = np.zeros(2)):
        self.p = [1, 0.5 + self.epsilon*p[1], 0.5 + self.epsilon*p[2], 1]
        return self.p
    def q(self, q = np.zeros(2)):
        self.q = [1, 0.5 + self.epsilon*q[1], 0.5 + self.epsilon*q[2], 1]
        return self.q

    def invasion_fitness(self):
        self.lambda1_2 = self.theta1*(self.beta[0] - self.beta[1]) + self.theta2*(-self.sgamma[0] + self.sgamma[1]) + self.theta3*(-self.cgamma[1] - self.cgamma[2] + self.cgamma[3]) + self.theta4*(self.p[1] - self.p[2]) + self.theta5*(self.mu*(self.K[2] - self.K[1]) + self.K[2] - self.K[3])
        self.lambda2_1 = self.theta1*(self.beta[1] - self.beta[0]) + self.theta2*(-self.sgamma[1] + self.sgamma[0]) + self.theta3*(-self.cgamma[2] - self.cgamma[1] + self.cgamma[0]) + self.theta4*(self.p[2] - self.p[1]) + self.theta5*(self.mu*(self.K[1] - self.K[2]) + self.K[1] - self.K[0])
        return self.lambda1_2, self.lambda2_1

    def neutralmodel(self, t):
        return integrate.odeint(neutralsystem, self.v0, t, args = (self.r, self.nbeta, self.ngamma,  self.nk, self.p, self.q, 0))

    
    def describe(self, option = 'parameters'):
        """Function to describe the patch. Option decides things to be described."""
        if option == 'parameters':
            print('Summary of patch', self.name, 'parameteres:')
            print('Strain reproduction rates:', self.b)
            print('Single infection clearance rates:', self.sgamma)
            print('Double infection clearance rates:', self.cgamma)
            print('Altered susceptibilities:', self.K)
            print('Probabiltity that second strain is transmited in co-infection:', self.q)
            print('Probabiltity that first strain is transmited in co-infection:', self.p)
            print('Patch', self.name, 'R0:', self.R0)

        if option == 'neutral equilibria':
            print('Patch', self.name, 'neutral S*:', self.nSstar)
            print('Patch', self.name, 'neutral T*:', self.nTstar)
            print('Patch', self.name, 'neutral I*:', self.nIstar)
            print('Patch', self.name, 'neutral D*:', self.nDstar)
        
        if option == 'invasion fitness':
            print('Invasion fitness of 1 -> 2:', self.lambda1_2)
            print('Invasion fitness of 2 -> 1:', self.lambda2_1)
        pass









patch1 = patch('A', [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1, 4, 2, 1, 0.1)






