import numpy as np




def potential_function(psi: np.ndarray)->np.ndarray:
    """Potential function, that calculates the dimensionless potential.
        Or:  which takes ψ and returns Vψ. Uses a fixed quartic potential V"""
    global mu, epsilon, N, D
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
    V_psi = (mu/8)*((epsilon**2)*squared_array_-1)**2*psi
    return V_psi


def kinetic_energy_function():
    """Kinetic energy function, which takes ψ and returns Hψ for the particular case V = 0."""


def hamiltonian_function():
    """Hamiltonian function, which takes ψ and returns Hψ. This function uses kinetic_energy_function  and potential_function ."""