import numpy as np
import numpy as np
rng = np.random.Generator(np.random.PCG64(12345))


def generate_positive_definite_matrix(N,kappa=10.):
    assert isinstance(N, int) and N > 1 , "N=" + str(N) + " must be an integer >= 2"
    assert isinstance(kappa, float) and kappa > 0. , "kappa=" + str(kappa) + " must be a positive float"
    
    rmat = np.asmatrix(rng.standard_normal(size=(N,N)) + 1j * rng.standard_normal(size=(N,N)))
    U , _ = np.linalg.qr(rmat,mode='complete')
    evalues = np.concatenate((1. + kappa*rng.random(N-2),[1.,kappa]))
    D = np.asmatrix(np.diag(evalues))
    A = np.matmul(np.matmul(U,D),U.getH())
    
    return A, U , evalues

def generate_complex_ndarray(N:int,D:int)->np.ndarray:
    assert isinstance(N, int) and N >= 1 , "N=" + str(N) + " must be an integer >= 1"
    assert isinstance(D, int) and D >= 1 , "N=" + str(D) + " must be an integer >= 1"
    A = rng.standard_normal(size=(N,)*D) + 1j * rng.standard_normal(size=(N,)*D)
    return A



def distance_array()->np.ndarray:
    """for given shape of array defined by global N, D  generate an array with the positional indexes centerd on the middle of the array.
    Used by potential_function to calculate the potential applied to ψ"""
    global N, D
    ix = np.ndindex((N,)*D)
    distance_array = np.zeros((N,)*D)
    for index in ix:
        shifted_index = np.array(index)-np.floor(N/2)
        distance_array[index] = np.sqrt(np.dot(shifted_index, shifted_index))
    return distance_array
def n2_array()->np.ndarray:
    """for given shape of array defined by global N, D  generate an array with the squared positional indexes centerd on the middle of the array.
    Used by potential_function to calculate the potential applied to ψ"""
    global N, D
    ix = np.ndindex((N,)*D)
    potential_array = np.zeros((N,)*D)
    for index in ix:
        shifted_index = np.array(index)-np.floor(N/2)
        potential_array[index] = np.dot(shifted_index, shifted_index)
    return potential_array



def generate_plane_wave(wave_number:tuple):
    """
    Generate a square D-dimensional array of size Nrepresenting a plane wave.

    Parameters:
    - wave_numbers: Tuple, the wave numbers for each dimension.

    Returns:
    - plane_wave_array: N-dimensional array with complex values.
    """
    global N, D
    assert len(wave_number) == D, "wave_number has lenght "+str(len(wave_number))+", it must have shape D="+str(D)
    # Calculate the values using the plane wave formula
    ix = np.ndindex((N,)*D)
    plane_wave_array = np.zeros((N,)*D)
    for index in ix:
        phase = np.dot((np.array(index)-np.floor(N/2)),np.array(wave_number))
        plane_wave_array[index] = phase
    plane_wave_array = np.exp(2j * np.pi * plane_wave_array/N)

    return plane_wave_array

############ Hamilton ##############

def potential_function(psi: np.ndarray)->np.ndarray:
    """Potential function which takes ψ and returns Vψ. Uses a fixed quartic potential V.
        mu, epsilon, N, D are requriered global variables"""
    
    assert isinstance(psi,np.ndarray) , "psi must me a ndarray"

    global mu, epsilon_2, N, D

      
    squared_array_ = n2_array()

    V_psi = (mu/8)*((epsilon_2)*squared_array_-1)**2*psi

    return V_psi


def kinetic_energy_function(psi: np.ndarray)->np.ndarray:
    """Kinetic energy function, which takes ψ and returns Hψ for the particular case V = 0.
    mu, epsilon, N, D are requriered global variables"""

    assert isinstance(psi,np.ndarray) , "psi must me a ndarray"

    global mu, epsilon_2, N, D
    derivative = np.zeros((N,)*D)
    for i in range(D):
        derivative = derivative + (np.roll(psi,[1 if index == i else 0 for index in range(D)], axis=tuple(range(D)))+np.roll(psi,[-1 if index == i else 0 for index in range(D)], axis=tuple(range(D)))-2*psi)

    e_kin = -(1/(2*mu*epsilon_2))*derivative

    return e_kin



def hamiltonian_function(psi: np.ndarray)->np.ndarray:
    """Hamiltonian function, which takes ψ and returns Hψ. This function uses kinetic_energy_function  and potential_function."""

    assert isinstance(psi,np.ndarray) , "psi must me a ndarray"

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
    if niters >= max_iters:
        raise ValueError("Maximum number of iterations reached.")        
            
   
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
        print(np.linalg.norm(r))
    if niters >= max_iters:
        raise ValueError(f"Maximum number of iterations reached during conjugate_gradient. epsilon = {np.linalg.norm(r)}")
    
    return x, niters


def smallest_eigenvalue_vector(apply_A:callable,  power_method_tolerance: float, conjugate_gradient_tolerance: float,
                                max_iters_power_method = 10000, max_iters_conjugate_gradient= 10000):
    """Function that calculates the smallest eigenvalue anhd corresponding eigenvector of a matrix"""
    global A, D, N
    c = rng.standard_normal(size=A.shape) + 1j * rng.standard_normal(size=A.shape)


    # adjust power method and conjugate gradient for local use

    def power_method_EV(vshape, apply_A, epsilon = power_method_tolerance, max_iters=max_iters_power_method):
        #global random_vector
        assert callable(apply_A) , "apply_A must be a function"
        assert isinstance(epsilon, float) and epsilon > 0. , "epsilon=" + str(epsilon) + " must be a positive float"
        assert isinstance(vshape,tuple) , "vshape must be a tuple"
        # v = rng.standard_normal(size=vshape) + 1j * rng.standard_normal(size=vshape)
        # v= v/np.linalg.norm(v)
        v = c
        print("power method start vector shape: ",v.shape)
    

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
        if niters >= max_iters:
            raise ValueError(f"Maximum number of iterations reached during power_method. epsilon = {res}  ")        
                
    
        return mu, v #, niters 

    def conjugate_gradient_EV(b, epsilon = conjugate_gradient_tolerance, max_iters = max_iters_conjugate_gradient):
        print(apply_A)
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
            # print(f'conj grad error = {np.linalg.norm(r)}')
        if niters >= max_iters:
            raise ValueError(f"Maximum number of iterations reached during conjugate_gradient. epsilon = {np.linalg.norm(r)}")    
        print("complete")
        return x # , niters


    #b = rng.standard_normal(size=A.shape) + 1j * rng.standard_normal(size=A.shape)
    k_vector = rng.standard_normal(size=(D,1))
    
    b = generate_plane_wave(tuple(k_vector))
    print(b)
    #conjugate_gradient(apply_A,b,conjugate_gradient_tolerance, max_iters_conjugate_gradient)

    return power_method_EV(b.shape,conjugate_gradient_EV)



############ Observables ##############


def expectation_energy(psi:np.ndarray)->float:
    """Caluculates expectation value of energy."""
    return np.vdot(psi, hamiltonian_function(psi))

def expectation_position(psi:np.ndarray)->float:
    """Caluculates expectation value of energy."""
    return np.vdot(psi, distance_array()*psi)

def expectation_momentum(psi:np.ndarray)->float:
    """Caluculates expectation value of energy."""
    derivative = np.zeros((N,)*D)
    for i in range(D):
        derivative = derivative + (np.roll(psi,[1 if index == i else 0 for index in range(D)], axis=tuple(range(D)))-np.roll(psi,[-1 if index == i else 0 for index in range(D)], axis=tuple(range(D))))

    return np.vdot(psi, derivative*scaling)
