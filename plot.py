import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
data_DEIM = np.load('/grand/NeuralDE/hjkim/ROM-DEIM/Burgers_DEIM/POD_DEIM.npy')
data_true = np.load('/grand/NeuralDE/hjkim/ROM-DEIM/Burgers_DEIM/POD_True_.npy')
sampling_index_DEIM = np.load('/grand/NeuralDE/hjkim/ROM-DEIM/Burgers_DEIM/sampling_index_DEIM.npy')

param = 'batchtime_10_lr_1.0e-05'
data_DEIM_ML = np.load(f'/grand/NeuralDE/hjkim/ROM-DEIM/Burgers_DEIM/POD_DEIM_ML_{param}.npy')
sampling_index_DEIM_ML = np.load(f'/grand/NeuralDE/hjkim/ROM-DEIM/Burgers_DEIM/sampling_index_DEIM_ML_{param}.npy')
print(sampling_index_DEIM_ML.shape)
print(data_DEIM_ML.shape)



x = np.arange(0,128)


fig, ax = plt.subplots()
ax.set_xlim(0, 128) 
ax.set_ylim(np.min(data_true), np.max(data_true))  
line1, = ax.plot(x, data_true[:,0], label='true')
line2, = ax.plot(x, data_DEIM[:,0], label='DEIM')
line3, = ax.plot(x, data_DEIM_ML[:,0], label=f'DEIM_ML_{param}')
line4, = ax.plot(sampling_index_DEIM, data_DEIM[sampling_index_DEIM,0], 'o', label='DEIM_sampling')
line5, = ax.plot(sampling_index_DEIM_ML[:,0], data_DEIM_ML[sampling_index_DEIM_ML[:,0],0], 'x', label=f'DEIM_ML_sampling_{param}')
ax.legend(loc='upper right')
  


def update(frame):
    line1.set_ydata(data_true[:,frame])
    line2.set_ydata(data_DEIM[:,frame])
    line3.set_ydata(data_DEIM_ML[:,frame])
    line4.set_ydata(data_DEIM[sampling_index_DEIM,frame])
    line5.set_xdata(sampling_index_DEIM_ML[:,frame])
    line5.set_ydata(data_DEIM_ML[sampling_index_DEIM_ML[:,frame],frame])
    return line1, line2, line3, line4, line5


ani = FuncAnimation(fig, update, frames=np.arange(0, 300), interval=50)

ani.save(f'animation_{param}.gif', writer='pillow', fps=30)
plt.show()
