
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

  
def Animate(t, tau, parameter, sols, axis_labels = [], plt_labels = []):

    figure, ax = plt.subplots(2, 2, figsize = (10, 10))
    
    

    
    def plot(sol):
        """Wrapper function to plot the solutions of the system, both in quantities and frequencies (Solutions of system and replicator, respectively)."""
        solution = sol['solution']
        replicator = sol['replicator_solution']
        labels = ['S in 1', 'S in 2', 'I1 in 1', 'I1 in 2','I2 in 1','I2 in 2', 'I11 in 1', 'I11 in 2', 'I12 in 1', 'I12 in 2', 'I21 in 1', 'I21 in 2', 'I22 in 1', 'I22 in 2']
        

        
        #fig.subplots_adjust(wspace = 0.5)

        for i in range(7):
            ax[0, 0].plot(t, solution.y[2*i][:len(t)], label = labels[2*i], color = 'Blue')
            ax[0, 1].plot(t, solution.y[2*i + 1][:len(t)], label = labels[2*i+1], color = 'Red')


        ax[1, 0].stackplot(tau[:len(replicator[0])], [replicator[0], replicator[2]], labels = ['Strain 1', 'Strain 2'], colors = ['Red', 'Blue'])
        ax[1, 1].stackplot(tau[:len(replicator[0])], [replicator[1], replicator[3]], labels = ['Strain 1', 'Strain 2'], colors = ['Red', 'Blue'])
        #Labeling everything
        
        return ax
    
    def animation_function(i):
        axe = plot(sols[i])

        return axe
    
    

    
    animation = FuncAnimation(figure,
                            func = animation_function,
                            frames = range(len(parameter)), 
                            interval = 10, repeat = False)

    plt.legend()
    plt.show()






