import numpy as np


def metropolis_algorithm(N,D,beta,b,random_seed:int, config_n, initial_config_type):
    assert initial_config_type == "hot" or initial_config_type == "cold", f"initial_config_type must be either 'hot' or 'cold'"
    np.random.seed(random_seed)
    
    if initial_config_type == "cold":
        config = np.ones((N,)*D)
    elif initial_config_type == "hot":
        config =  np.random.choice([-1, 1], size=(N,)*D, p=[0.5, 0.5])


    magnetization_array = np.zeros((config_n,))
    energy_array = np.zeros((config_n,))

    def checkerboard_mask(shape):
        return np.indices(shape).sum(axis=0) % 2 == 0
    white_mask = checkerboard_mask((N,)*D)
    black_mask = np.logical_not(white_mask)
    

    # iterate over number of configurations (0.62seconds/1000 iters)
    for config_iter in range(config_n):
        # print(config_iter,'/', config_n, end = '\r')
        

        #checkerboard array approach

        # random number array
        random_numbers = np.random.rand(N**D)
        random_array = random_numbers.reshape((N,)*D)

        # white
        neighbours = np.zeros((N,)*D)
        for i in range(D):
            neighbours = neighbours + (np.roll(config,1, axis=i)+np.roll(config,-1, axis=i))

        dH_array_white = np.exp(-(2*beta*config*neighbours +2*b*config))
        #only look at white indexes
        dH_array_white[black_mask] = 0
        white_flip_mask = dH_array_white > random_array
        # now flip accordingly based off of white spins
        config[white_flip_mask]*=-1
        # print(config)
        # now repeat for black
        neighbours = np.zeros((N,)*D)
        for i in range(D):
            neighbours = neighbours + (np.roll(config,1, axis=i)+np.roll(config,-1, axis=i))

        dH_array_black = np.exp(-(2*beta*config*neighbours +2*b*config))
        #only look at even indexes
        dH_array_black[white_mask] = 0
        black_flip_mask = dH_array_black > random_array
        
        config[black_flip_mask]*=-1

        
        
        magnetization_array[config_iter] = np.sum(config)

        #calculate config energy
        neighbours = np.zeros((N,)*D)
        for i in range(D):
            #only add right neighbours
            neighbours = neighbours + np.roll(config,1, axis=i)
        energy_config = np.sum(-beta*config*neighbours -b*config)
        energy_array[config_iter] = energy_config

    return magnetization_array, energy_array


b_array = np.array([0.01,0.005,0.001,0.0005,0.0001])
b_steps = b_array.shape[0]



config_number = 100000
beta_steps = 100
beta_max = 1
beta_min = 0.2
N = 100
D = 2
beta_array = np.linspace(beta_max,beta_min,beta_steps)
initial_state = "cold"

Thermalization_start = 20000


magnetization_vs_beta = np.load(f'ising_output/magnetization_vs_b_{config_number}c_{beta_steps}beta.npy')
energy_vs_beta = np.load(f'ising_output/energy_vs_b_{config_number}c_{beta_steps}beta.npy')


average_M_vs_b_array = np.zeros((beta_steps,b_steps))
average_E_vs_b_array = np.zeros((beta_steps,b_steps))
for i, b in enumerate(b_array):
    for j, beta in enumerate(beta_array):
        average_M_vs_b_array[j,i] = np.mean(magnetization_vs_beta[Thermalization_start:,j, i])
        average_E_vs_b_array[j,i] = np.mean(energy_vs_beta[Thermalization_start:,j, i])



W_array = np.linspace(30,100,71)
W_steps = W_array.shape[0]
Nth = Thermalization_start
S_M_vs_b_array = np.zeros((beta_steps, b_steps, W_steps))
S_E_vs_b_array = np.zeros((beta_steps, b_steps, W_steps))
error_M_vs_b_array = np.zeros((beta_steps, b_steps, W_steps))
error_E_vs_b_array = np.zeros((beta_steps, b_steps, W_steps))
b_array = np.array([0.01])
for l,W in enumerate(W_array):
    for i, b in enumerate(b_array):        
        for j, beta in enumerate(beta_array):
            for n in range(Nth, config_number):
                for m in range(int(max(Nth, n - W)), int(min(config_number, n + W+1))):
                    S_M_vs_b_array[j,i,l] += magnetization_vs_beta[m,j, i]*magnetization_vs_beta[n,j, i]
                    S_E_vs_b_array[j,i,l] += energy_vs_beta[m,j, i]*energy_vs_beta[n,j, i]
S_M_vs_b_array = S_M_vs_b_array/((config_number-Nth)**2)
S_E_vs_b_array = S_E_vs_b_array/((config_number-Nth)**2)        


np.save(f'ising_output/S_M_vs_b_{config_number}c_{beta_steps}beta.npy', S_M_vs_b_array)
np.save(f'ising_output/S_E_vs_b_{config_number}c_{beta_steps}beta.npy', S_E_vs_b_array)