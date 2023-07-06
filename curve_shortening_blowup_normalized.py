import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import matplotlib.animation as animation

N=100

def grid():
    N = 10 # Change as needed, remember this is now the number of points along one side of the grid

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    x, y = np.meshgrid(x, y)

    # Flatten x and y to 1D arrays, to match the shape expected by the rest of the script
    x = x.flatten()
    y = y.flatten()
    
    return x,y

def random():
    N = 100

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    return x,y
def circle():
    N = 100
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)

    # Circle with radius 1
    r = 0.5

    x = r * np.cos(theta)+0.5
    y = r * np.sin(theta)+0.5
    
    return x,y
def triangle():
    N = 100
    t_AB = np.linspace(0, 1, N // 3, endpoint=False)
    t_BC = np.linspace(0, 1, N // 3, endpoint=False)
    t_CA = np.linspace(0, 1, N - 2 * (N // 3), endpoint=False)

    x_AB = 1 - t_AB
    y_AB = np.zeros(N // 3)

    x_BC = np.zeros(N // 3)
    y_BC = t_BC

    x_CA = t_CA
    y_CA = 1 - t_CA

    x = np.concatenate((x_AB, x_BC, x_CA))
    y = np.concatenate((y_AB, y_BC, y_CA))
    return x,y

selec="random"
funlist={"circle":circle(),"triangle":triangle(),"grid":grid(),"random":random()}
x,y = funlist[selec]

db = []
colors = ['red']*(N // 3) + ['green']*(N // 3) + ['blue']*(N - 2 * (N // 3))

# rest of the script...

fig, ax = plt.subplots(dpi=200)
sc = ax.scatter(x, y, c=colors)
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_aspect('equal', adjustable='box')  # This line sets the aspect ratio

plt.tight_layout()

step_size = 0.51  # Adjust the step size as needed

norm_data = []

def animate(i):
    global x, y
    
    if i == 0:  # Add this condition to handle the first frame
        sc.set_offsets(np.c_[x, y])
        return sc,
    
    x_new = np.roll(x, -1) - 2 * x + np.roll(x, 1)
    y_new = np.roll(y, -1) - 2 * y + np.roll(y, 1)

    x += step_size * x_new
    y += step_size * y_new
    
    # Normalize after Laplacian operator
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_range = x_max - x_min
    y_range = y_max - y_min

    x = (x - x_min) / x_range
    y = (y - y_min) / y_range

    inter = pd.DataFrame({'x': x, 'y': y, 'iteration': [i+1]*len(x), 'color': colors})
    db.append(inter)
    
    norm_data.append({'iteration': i+1, 'x_min': x_min, 'x_max': x_max, 'x_range': x_range, 
                      'y_min': y_min, 'y_max': y_max, 'y_range': y_range})

    sc.set_offsets(np.c_[x,y])

ani = animation.FuncAnimation(fig, animate, frames=2001, repeat=False)
ani.save('curve_animation_step_'+str(step_size).replace(".","_")+'_long_'+selec+'.mp4', writer='ffmpeg',dpi=200, bitrate=2000)
plt.close(fig)
df = reduce(lambda x,y:pd.concat([x,y]),db)
df.to_csv('discrete_normalized_laplacian_points_'+str(step_size).replace(".","_")+'_'+selec+'.csv', index=False)

norm_df = pd.DataFrame(norm_data)

fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs = axs.ravel()
for i, column in enumerate(norm_df.columns[1:]):
    axs[i].plot(norm_df['iteration'], norm_df[column])
    axs[i].set_title(column)
    axs[i].set_xlabel('iteration')
plt.tight_layout()
plt.savefig('norm_values_vs_iteration_'+str(step_size).replace(".","_")+'_'+selec+'.png')
plt.close(fig)