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
    assert isinstance(N, int) and N > 1 , "N=" + str(N) + " must be an integer >= 2"
    A = rng.standard_normal(size=(N,)*D) + 1j * rng.standard_normal(size=(N,)*D)
    return A



############ Hamilton ##############

def potential_function(psi: np.ndarray)->np.ndarray:
    """Potential function which takes ψ and returns Vψ. Uses a fixed quartic potential V.
        mu, epsilon, N, D are requriered global variables"""
    
    assert isinstance(psi,np.ndarray) , "psi must me a ndarray"

    global mu, epsilon_2, N, D

    def squared_array()->np.ndarray:
        """for given shape of array defined by global N, D  generate an array with the squared positional indexes centerd on the middle of the array.
        Used by potential_function to calculate the potential applied to ψ"""
        global N, D
        ix = np.ndindex((N,)*D)
        potential_array = np.zeros((N,)*D)
        for index in ix:
            shifted_index = np.array(index)-np.floor(N/2)
            potential_array[index] = np.dot(shifted_index, shifted_index)
        return potential_array
        
    squared_array_ = squared_array()

    V_psi = (mu/8)*((epsilon_2)*squared_array_-1)**2*psi

    return V_psi


def kinetic_energy_function(psi: np.ndarray)->np.ndarray:
    """Kinetic energy function, which takes ψ and returns Hψ for the particular case V = 0.
    mu, epsilon, N, D are requriered global variables"""

    assert isinstance(psi,np.ndarray) , "psi must me a ndarray"

    global mu, epsilon_2, N, D

    e_kin = -1/(2*mu*epsilon_2)*(np.roll(psi,1)+np.roll(psi,-1)-2*psi)

    return e_kin



def hamiltonian_function(psi: np.ndarray)->np.ndarray:
    """Hamiltonian function, which takes ψ and returns Hψ. This function uses kinetic_energy_function  and potential_function."""

    assert isinstance(psi,np.ndarray) , "psi must me a ndarray"

    H_psi = kinetic_energy_function(psi)+potential_function(psi)

    return H_psi



############ Eigenvalues/Eigenvektor ##############