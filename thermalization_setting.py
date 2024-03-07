import numpy as np
import matplotlib.pyplot as plt

b_array = np.array([0.01,0.005,0.001,0.0005,0.0001])
b_steps = b_array.shape[0]

config_number = 100000
beta_steps = 20
beta_max = 1
beta_min = 0.2
N = 100
D = 2
beta_array = np.linspace(beta_max,beta_min,beta_steps)
initial_state = "hot"

magnetization_vs_beta = np.load(f'ising_output/magnetization_vs_b_{config_number}c_{beta_steps}beta.npy')
energy_vs_beta = np.load(f'ising_output/energy_vs_b_{config_number}c_{beta_steps}beta.npy')
print(magnetization_vs_beta.shape)
print(range(config_number))
thermalization_array = np.zeros((b_array.shape[0], beta_array.shape[0]), int)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.ion()
# Plot Magnetization
for i, b in enumerate(b_array):
    for j, beta in enumerate(beta_array):
        ax.scatter(range(config_number),magnetization_vs_beta[:,j,i], s = 1)
        ax.set_xticks(np.arange(0, config_number, 10000))
        ax.set_yticks([])
        plt.grid()
        # axes[1].scatter(range(config_number),energy_vs_beta[:,j,i], s = 1)
        plt.show()
        thermalization_array[i,j]= 1000*int(input("input thermalization start:"))
        plt.cla()
np.save(f'ising_output/thermalization_array_{config_number}.npy', thermalization_array)