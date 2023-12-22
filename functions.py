import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
rng = np.random.Generator(np.random.PCG64(12345))

#### Utility #####
def generate_complex_ndarray(N:int,D:int)->np.ndarray:
    """
    Generates a numpy.ndarray of shape NxD with random complex floats as entries.
    The function produces two arrays: one real and one complex and adds them togther to create a mixed array.

    ## Parameters
    - N (int)>=1: Size of the array to generate.
    - D (int)>=1: Dimension of the array to generate.

    ## Returns
    numpy.ndarray: Complex array.
    """
    assert isinstance(N, int) and N >= 1 , "N=" + str(N) + " must be an integer >= 1"
    assert isinstance(D, int) and D >= 1 , "N=" + str(D) + " must be an integer >= 1"
    A = rng.standard_normal(size=(N,)*D) + 1j * rng.standard_normal(size=(N,)*D)
    return A

def position_array(axis:int)->np.ndarray:
    """    
    Provides the indices along the specified axis for the lattice defined by global variables N, D.

    ## Parameters
    - axis (int): Axis along which to define the indices.

    ## Returns
    numpy.ndarray: Array of indices along the specified axis.
    """
    global N, D
    position_array = np.zeros((N,)*D)
    indices = np.indices(position_array.shape, dtype=int)
    coordinates = indices[axis]-N/2
    return coordinates



def n2_array()->np.ndarray:
    """
    Generates an array with the positional indices centered on the middle of the array (for lattice defined by global variables N, D).
    The function generates an array of indices and shifts them towards ther center of the array. Then the squared distance from the center is calculated for each lattice point.
    ## Parameters
    - none

    ## Returns
    - numpy.ndarray: Array of indices with respect to origin centered on the middle of the array.
    """
    global N, D
    ix = np.ndindex((N,)*D)
    potential_array = np.zeros((N,)*D)
    for index in ix:
        shifted_index = np.array(index)-np.floor(N/2)
        potential_array[index] = np.dot(shifted_index, shifted_index)
    return potential_array



def generate_plane_wave(wave_number:tuple):
    """
    Generates an arrayre representing a plane wave (for lattice defined by global variables N, D).

    ## Parameters
    - wave_numbers: Tuple, the wave numbers for each dimension.

    ## Returns
    - plane_wave_array: N-dimensional array with complex values.
    """
    global N, D
    assert len(wave_number) == D, "wave_number has lenght "+str(len(wave_number))+", it must have shape D="+str(D)
    # Calculate the values using the plane wave formula
    ix = np.ndindex((N,)*D)
    plane_wave_array = np.zeros((N,)*D, dtype=complex)
    for index in ix:
        shifted_index = np.array(index)-np.floor(N/2)       
        entry = np.exp(2j * np.pi *np.vdot(shifted_index,np.array(wave_number))/N)
        plane_wave_array[index] = entry/(N**(D/2))
    

    return plane_wave_array

############ Hamilton ##############

def potential_array_calculator()->np.ndarray:
    """
    Calculates the quadratic potential described by mu & epsilon_2, centered on the lattice for given parameters  N, D. 
    This potential is constant for the given set of parameters and can therefore be computed once and reused, for instance in the potential_function.
    This function is to be called once for each new set of parameters to set the global variable "potential_array".
    ## Parameters
    - none

    ## Returns
    - numpy.ndarray: Array defining the quadratic potential.
    """
    global mu, epsilon_2, N, D
    squared_array_ = n2_array()       
    potential_array = (mu/8)*((epsilon_2)*squared_array_-1)**2
    return potential_array

def potential_wall_array_calculator(axis:int, height, origin, greater:int)->np.ndarray:
    """
    Generates a potential plateau along a given axis with specific height. Further the position and direction where the plateau starts can be set.
    This potential is used to showcase wavefunction behaviour against a potential wall.
    This function is to be called once for each new set of parameters to set the global variable "potential_array.
    ## Parameters
    - none

    ## Returns
    - numpy.ndarray: Array defining the quadratic potential.
    """
    global mu, epsilon_2, N, D
    position = position_array(axis)
    potential_wall_array = np.zeros((N,)*D)
    ix = np.ndindex((N,)*D)
    for index in ix:
        if greater*(position[index]-origin)>0:
            potential_wall_array[index] = height*(mu)
        else:
            potential_wall_array[index] = 0
    return potential_wall_array

def potential_function(psi: np.ndarray)->np.ndarray:
    """
    Applies the potential to the wavefunction.
    This function requires the global variable potential_array to be defined, as it is used for the calculation.
    The function consists of an array multiplication between the potential array and the wavefunction.
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD onto which the potential is to be applied.

    ## Returns
    - numpy.ndarray: result of the array multiplication between the potential and the wavefunction.
    """
    
    assert isinstance(psi,np.ndarray) , "psi must be a ndarray"

    global potential_array
    assert isinstance(potential_array,np.ndarray) , "potential_array must be a ndarray"
    V_psi = potential_array*psi

    return V_psi


def kinetic_energy_function(psi: np.ndarray)->np.ndarray:
    """Kinetic energy function, which takes ψ and returns Hψ for the particular case V = 0.
    mu, epsilon, N, D are requiered global variables.
    The function calculates the discretized gradient using the wavefunction value of the nearest neighbour lattice points.
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD onto which the potential is to be applied.

    ## Returns
    - numpy.ndarray: kinetic energy operator applied to the array of wavefunction.
    """

    assert isinstance(psi,np.ndarray) , "psi must be a ndarray"

    global mu, epsilon_2, N, D
    derivative = np.zeros((N,)*D)
    for i in range(D):
        derivative = derivative + (np.roll(psi,1, axis=i)+np.roll(psi,-1, axis=i)-2*psi)

    e_kin = -(1/(2*mu*epsilon_2))*derivative
    return e_kin



def hamiltonian_function(psi: np.ndarray)->np.ndarray:
    """Hamiltonian function, which takes ψ and returns Hψ.
    This function calls the  kinetic_energy_function  and potential_function and adds their outputs.
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD onto which the potential is to be applied.

    ## Returns
    - numpy.ndarray: hamiltonian operator applied to the array of wavefunction.
    """

    assert isinstance(psi,np.ndarray) , "psi must be a ndarray"

    H_psi = kinetic_energy_function(psi)+potential_function(psi)

    return H_psi



############ Eigenvalues/Eigenvektor ##############



def power_method(vshape,apply_A,epsilon,max_iters=1000000):
    #global random_vector
    assert callable(apply_A) , "apply_A must be a function"
    assert isinstance(epsilon, float) and epsilon > 0. , "epsilon=" + str(epsilon) + " must be a positive float"
    assert isinstance(vshape,tuple) , "vshape must be a tuple"
    v = rng.standard_normal(size=vshape) + 1j * rng.standard_normal(size=vshape)
    v= v/np.linalg.norm(v)
    print("start vector shape: ",v.shape)
   

    #initialize values before starting loop
    niters = 0
    mu = np.linalg.norm(apply_A(v))    
    w = apply_A(v)   
    res = np.linalg.norm(w - np.dot(mu,v))
   
    niters = 0
    while res>epsilon and niters < max_iters:        
    
        res = np.linalg.norm(w - np.dot(mu,v))        
        v =  np.dot((1 / mu),w)
        w = apply_A(v)
        mu = np.linalg.norm(w)
        niters+=1
        print(f"Power method iterations: {niters}/{max_iters}", end='\r')
    if niters >= max_iters:
        print(f"Maximum number of iterations reached during power_method. epsilon = {res}  ")        
        return mu, v              
            
   
    return mu, v #, niters 

def conjugate_gradient(apply_A,b,epsilon, max_iters = 10000):

    x = np.zeros(b.shape)
    r = b- apply_A(x)
    p = r
    niters = 0
    epsilon_norm_b = epsilon*np.linalg.norm(b)
    #calculate espilon norm b real part once, then square
    #norm r^2 .realinstead of root, saves caluclation
    rnew_2 = np.vdot(r, r).real
    
    while np.linalg.norm(r)>epsilon_norm_b and niters < max_iters:
        #store r and rnew vdot for next iteration
        Ap = apply_A(p)      
        r_2 = rnew_2
        alpha = r_2/np.vdot(p,Ap).real
        x = x + alpha * p
        rnew  = r - alpha * Ap
        rnew_2 = np.vdot(rnew, rnew).real
        beta = rnew_2/r_2
        p = rnew + beta*p
        r = rnew
        niters+=1        
    if niters >= max_iters:
        print(f"Maximum number of iterations reached during conjugate_gradient. epsilon = {np.linalg.norm(r)}")        
        return x
        
    return x


def lowest_eigenvalue_vector(hamiltonian:callable,  power_method_tolerance: float, conjugate_gradient_tolerance: float,
                                max_iters_power_method = 10000, max_iters_conjugate_gradient= 10000):
    """
    Function that calculates the smallest eigenvalue and corresponding eigenvector of the hamiltonian for the given lattice.
    This function calls the  power_method function and provides it the conjugate_gradient of the hamiltonian function applied on the wavefunction.
    Effectively the power method is applied on the inverse hamiltonian operator, resulting in the smalles eigenvalue and vector for the hamiltonian..
    mu, epsilon, N, D are requiered global variables.
    ## Parameters
    - hamiltonian: callable, the hamiltonian function of which the eigenvalue and eigenfunction will be calculated.
    - power_method_tolerance: float, required precision of the power method function.
    - conjugate_method_tolerance: float, required of the conjugate gradient function.
    - max_iters_power_method: int (optional), maximum iterations of the power method function
    - max_iters_conjugate_gradient: int (optional), maximum iterations of the conjugate gradient function

    ## Returns
    - mu: float, smallest eigenvalue of hamiltonian operator.
    - v: numpy.ndarray, eigenfunction corresponding to mu.
    """
    global A, D, N

   
    def wrapper_conjugate_gradient(b, apply_A = hamiltonian):  
        return conjugate_gradient(apply_A, b,  epsilon = conjugate_gradient_tolerance, max_iters = max_iters_conjugate_gradient)
    
    shape = (N,)*D

    return power_method(shape, wrapper_conjugate_gradient, power_method_tolerance, max_iters_power_method)
    



############ Observables ##############


def expectation_energy(psi:np.ndarray)->float:
    """
    Calculates expectation value of energy for wavefunction psi.
    Function calculates the array product of psi, H(psi).
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - float: expectation value of energy.
    """
    return np.vdot(psi, hamiltonian_function(psi)).real

def expectation_position(psi:np.ndarray)->float:
    """
    Calculates expectation value of position for wavefunction psi.
    Function calculates the array product of psi, position*psi.
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - numpy.ndarray: expectation value of position as an array.
    """
    global D
    position_vector = np.empty((D,1))*epsilon
    for i in range(0,D):
        position_vector[i]=(np.vdot(psi, position_array(i)*psi).real)*epsilon
    return position_vector

def expectation_momentum(psi:np.ndarray)->float:
    """
    Calculates expectation value of momentum for wavefunction psi.
    Function calculates the array product of psi, derivative(psi).
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - numpy.ndarray: expectation value of momentum as an array.
    """
    global mu, epsilon, N, D
    momentum_expct = np.zeros((D,1))
    for i in range(D):
        derivative =(np.roll(psi,[1 if index == i else 0 for index in range(D)], axis=tuple(range(D)))-np.roll(psi,[-1 if index == i else 0 for index in range(D)], axis=tuple(range(D))))
        momentum_expct[i]= np.vdot(psi, -(1/2)*1j*1/(mu*epsilon)*derivative).real
    return momentum_expct

def indetermination_position(psi:np.ndarray)->float:
    """
    Calculates indetermination of position for wavefunction psi.
    Function calculates the array product of psi, position^2*psi.
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - numpy.ndarray: indetermination of position as an array.
    """
    global D, epsilon
    position2_vector = np.empty((D,1))
    for i in range(0,D):
        position2_vector[i]=(np.vdot(psi, position_array(i)**2*psi).real)*(epsilon**2)
    return position2_vector-expectation_position(psi)**2

def indetermination_momentum(psi:np.ndarray)->float:
    """
    Calculates indetermination of momentum for wavefunction psi.
    Function calculates the array product of psi, derivative^2(psi).
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - numpy.ndarray: indetermination of momentum as an array.
    """
    global mu, epsilon_2, N, D
    second_derivative = np.zeros((D,1))
    for i in range(D):
        derivative_2= (np.roll(psi,[1 if index == i else 0 for index in range(D)], axis=tuple(range(D)))+np.roll(psi,[-1 if index == i else 0 for index in range(D)], axis=tuple(range(D)))-2*psi)
        second_derivative[i] = np.vdot(psi, -1/(4*mu**2*epsilon_2)*derivative_2).real
    return second_derivative-expectation_momentum(psi)**2

def probability_xg0(psi:np.ndarray)->float:
    """
    Calculates probability that the particle is found in x>0 for wavefunction psi.
    Function calculates the norm of the wavefunction for x>0 divided by the full norm.
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - float: probability that the particle is found in x>0.
    """
    psi_l0, psi_g0 = np.array_split(psi, 2, axis = 0)
    return np.vdot(psi_g0, psi_g0).real/np.vdot(psi, psi).real

def norm(psi:np.ndarray)->float:
    """
    Calculates norm of wavefunction psi.
    Function calculates the norm of the wavefunction as array product of psi,psi.
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - float: norm of the wavefunciton psi.
    """
    norm = np.vdot(psi,psi).real
    return norm
    
def expectation_kinetic_energy(psi:np.ndarray)->float:
    """
    Calculates expectation value of Kinetic energy for wavefunction psi.
    Function calculates the array product of psi, Ekin(psi).
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - float: expectation value of kinetic energy.
    """
    
    return np.vdot(psi, kinetic_energy_function(psi)).real

def expectation_potential_energy(psi:np.ndarray)->float:
    """
    Calculates expectation value of Potential energy for wavefunction psi.
    Function calculates the array product of psi, Epot(psi).
    ## Parameters
    - psi: numpy.ndarray, the wavefunction of size NxD.

    ## Returns
    - float: expectation value of potential energy.
    """
    return np.vdot(psi, potential_function(psi)).real

############ Time Evolution ##############

def generate_gauss(k0: np.ndarray, origin:np.ndarray, width: float)->np.ndarray:
    """
    Generate a square D-dimensional array of size N representing a gaussian wave packet.
    ## Parameters:
    - k0: np.ndarray, maximum of gaussian in momentum space.
    - origin: tuple, origin coordinates (relative to center) of the gaussian wave packet
    - width: float, width of the gaussian curve
    ## Returns:
    - np.ndarray: array for the wavefunction 
    """
    global N, D
    ix = np.ndindex((N,)*D)
    gaussian = np.zeros((N,)*D, dtype=complex)
    # Calculate the values using the plane wave formula
    ix = np.ndindex((N,)*D)
    for index in ix:
        position= (np.array(index-np.floor(N/2))-origin)#-1j*k0/2)
        #entry = np.sqrt(np.sqrt(2/np.pi))*np.exp(-0.25*np.vdot(k0, k0))*np.exp(-np.vdot(position- 1j*k0/2, position- 1j*k0/2))
        entry =  np.sqrt(np.sqrt(2/np.pi))*np.exp((-np.vdot(position, position)/width+1j*np.vdot(k0, position)))
        gaussian[index] =     entry 

    return gaussian



def integrator_euler(psi:np.ndarray, time_step: float, hamiltonian:callable)->np.ndarray:
    """
    Evolves function psi by provided timestep.
    The function calculates : psi-i*time_step*Hamiltonian(psi).
    ## Parameters:
    - psi: np.ndarray, wavefunction to be evolved.
    - time_step: float, time step of the evolution 
    - hamiltonian: callable, hamiltonian function to be used during evolution
    ## Returns:
    - np.ndarray: Array of the wavefunction evolved by time_step
    """
    psi_evolved = psi - 1j*time_step*hamiltonian(psi)
    return psi_evolved

def integrator_crank_nicolson(psi:np.ndarray, time_step: float, hamiltonian:callable, conjugate_gradient_tolerance= 1.e-10, max_iters_conjugate_gradient= 100000):
    """
    Evolves function psi by provided timestep.
    The function calls the conjugate gradient function to calculate the inverse of an operator array.
    ## Parameters:
    - psi: np.ndarray, wavefunction to be evolved.
    - time_step: float, time step of the evolution 
    - hamiltonian: callable, hamiltonian function to be used during evolution
    ## Returns:
    - np.ndarray: Array of the wavefunction evolved by time_step
    """
    
    def inversion_term(psi):
        """Term to be inverted by conjugate gradient"""
        return psi+0.25*time_step**2*hamiltonian(hamiltonian(psi))
    eta = conjugate_gradient(inversion_term,psi, epsilon=conjugate_gradient_tolerance, max_iters=max_iters_conjugate_gradient)
    
    psi_evolved = eta-(1j/2)*time_step*hamiltonian(eta)+0.25*time_step**2*hamiltonian(hamiltonian(eta))
    return psi_evolved

def kinetic_energy_array_calculator()->np.ndarray:
    """
    Calculates kinetic energy array by means of hamiltonian eigenvalues for given parameters.
    The function output is used in the integrator_strang_splitting function.
    The function requires N, D, mu, epsilon_2 as global variables.
    ## Parameters:
    - none
    ## Returns:
    - np.ndarray: kinetic energy array to be used in integrator_strang_splitting function
    """
    global N, D, mu, epsilon_2
    ix = np.ndindex((N,)*D)
    kinetic_energy = np.zeros((N,)*D,dtype=complex)
    for index in ix:
        shifted_index = np.array(index)-np.floor(N/2)
        shifted_index_flat = shifted_index.ravel()
        shifted_index_k_flat = np.fft.fft(shifted_index_flat)
        shifted_index_k = shifted_index_k_flat.reshape(shifted_index.shape)
        sum = 0
        for i in range(D):
            sum += ((np.sin(np.pi*shifted_index_k[i]/N)))**2
        kinetic_energy[index] = (2/(mu*epsilon_2))*sum
    return kinetic_energy

def integrator_strang_splitting(psi:np.ndarray, time_step: float, hamiltonian):
    """
    Evolves function psi by provided timestep.
    The function uses the kinetic_energy_array_calculator function as well as fast fourier transformation library provided by numpy.
    The function requires N, D, mu, epsilon, potential_array, kinetic_energy_array as global variables.
    ## Parameters:
    - psi: np.ndarray, wavefunction to be evolved.
    - time_step: float, time step of the evolution 
    - hamiltonian: callable (not used, compatibility), hamiltonian function to be used during evolution
    ## Returns:
    - np.ndarray: Array of the wavefunction evolved by time_step
    """
    global N, D, mu, epsilon, potential_array, kinetic_energy_array    

    eta_real = np.exp(-0.5j*time_step*potential_array)*psi
    eta_real_flat = eta_real.ravel()
    eta_freq_flat = np.fft.fft(eta_real_flat)
    eta_freq =  eta_freq_flat.reshape(eta_real.shape)   
    xi_freq = np.exp(-1j*time_step*kinetic_energy_array)*eta_freq
    xi_feq_flat =xi_freq.ravel()
    xi_real_flat = np.fft.ifft(xi_feq_flat)
    xi_real =  xi_real_flat.reshape(eta_real.shape) 
    psi_evolved = np.exp(-0.5j*time_step*potential_array)*xi_real
    return psi_evolved






def time_evolution(psi:np.ndarray, hamiltonian, integrator:callable, time_step: float, time_total: float)->pd.DataFrame:
    """
    Function that iteratively calculates the time evolution using the selected integrator.
    The function iterates until the target time is used, while storing the wavefunction array and the observables as a dataframe.

    ## Parameters:
    - psi: np.ndarray, wavefunction to be evolved.
    - time_step: float, time step of the evolution. 
    - time_total: flaot, total time to be reached through iteration.
    - integrator: callable, integrator to be used for the time evolution.
    - hamiltonian: callable, hamiltonian function to be used during evolution.
    ## Returns:
    - pandas.dataframe: dataframe containing the avefunction and all observables for each time_step
    """
    assert callable(integrator) , "ntegrator must be a function"
    assert isinstance(time_step, float) and time_step > 0. , "time_step=" + str(time_step) + " must be a positive float"
    assert isinstance(time_total, float) and time_total > 0. , "time_total=" + str(time_total) + " must be a positive float"
    
    
    time_elapsed = 0
    evolution_log = pd.DataFrame(columns = ['norm', 'exp_Etot', 'exp_Ekin','exp_Epot', 'exp_pos', 'indet_pos'])
    while time_elapsed <= time_total:
        evolved_psi = integrator( psi, time_step, hamiltonian)
        step_log = {'time':time_elapsed, 'function':evolved_psi,'norm':norm(evolved_psi), 'exp_Etot':expectation_energy(evolved_psi), 'exp_Ekin':expectation_kinetic_energy(evolved_psi), 'exp_Epot':expectation_potential_energy(evolved_psi),  'exp_pos':expectation_position(evolved_psi), 'indet_pos':indetermination_position(evolved_psi)}
        step_df = pd.DataFrame([step_log])
        evolution_log = pd.concat([evolution_log, step_df], axis= 0)
        psi = evolved_psi
        time_elapsed+=time_step
        print(f"Time evolution iterations: {time_elapsed}/{time_total}", end='\r')
    
    return evolution_log


def plot_animation(df, time_resolution, hamiltonian, file_name, title):
    """
    Function plots the wavefunction animation over time.
    The plotted function is the absolute square of the function as well as the potential.

    ## Parameters:
    - df: pandas.dataframe, contains all data on time evolved function.
    - time_resolution: float, factor of how many of all steps should be plotted. 
    - hamiltonian: callable, hamiltonian function that was used during evolution.
    - file_name: str, filename of the animation to be saved.
    - title: str, title to be displayed on the animation
    ## Returns:
    - pandas.dataframe: dataframe containing the avefunction and all observables for each time_step
    """

    global N, D, epsilon

    df_selected = df.iloc[::time_resolution]

    x = np.zeros((N,)*D)
    x_axis = np.linspace(-N/2,N/2,N)
    x_axis= x_axis*epsilon
    max_function_value = df_selected['function'].apply(lambda x: np.max(x**2) / epsilon).max()
    min_function_value = df_selected['function'].apply(lambda x: np.min(x**2) / epsilon).min()


    fig, ax = plt.subplots()

    # Set plot labels and title



    # Function to update the plot for each frame
    def update(frame):
        
        ax.clear()  # Clear the previous frame
        ax.set_ylabel(r"$|\Psi|^{2}/ \varepsilon$")
        ax.set_xlabel(r"$\frac{x}{r}$")
        ax.set_title(title)
        time = df_selected['time'].iloc[frame]
        function = df_selected['function'].iloc[frame]
        potential = potential_function(np.ones_like(x))

        # Plot the function and potential
        ax.plot(x_axis, function**2/ epsilon, label='Function', color='C0')
        if hamiltonian == hamiltonian_function:
            ax.plot(x_axis, potential.real, label=fr'Potential $\mu$={mu}, a.u.', color='C1')
        ax.set_ylim(min_function_value, max_function_value)
        ax.text(0.75,0.15, fr'$\tau$={time:.5}', bbox={'facecolor': 'white', 'pad': 1},transform=plt.gcf().transFigure)
        # Set plot labels and title
        

        ax.legend()

    # Create the animation
    num_frames = len(df_selected)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=False)

    # Save the animation as a GIF
    animation_file = file_name
    animation.save(animation_file, writer='imagemagick', fps = 30)
