
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
  
def animate(parameter, data1, data2):
    x = []
    y = []
    
    figure, ax = plt.subplots()
    
    ax.set_xlim(min(parameter), max(parameter))
    ax.set_ylim(min(min(data1), min(data2), max(max(data1), max(data2))))

    

    
    def animation_function(i):
        # x.append(parameter[i])
        # y.append(data[i])
        
        # line1.set_xdata(x)
        # line1.set_ydata(y)

        ax.plot(parameter[:i], data1[:i], color = 'red')
        ax.plot(parameter[:i], data2[:i], color = 'blue')

        return ax
    
    animation = FuncAnimation(figure,
                            func = animation_function,
                            frames = range(len(parameter)), 
                            interval = 10, repeat = False)
    
    plt.show()






