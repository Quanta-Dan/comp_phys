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



############ Hamilton ##############

def potential_function(psi: np.ndarray)->np.ndarray:
    """Potential function which takes ψ and returns Vψ. Uses a fixed quartic potential V.
        mu, epsilon, N, D are requriered global variables"""
    
    assert isinstance(psi,np.ndarray) , "psi must me a ndarray"

    global mu, epsilon_2, N, D, n_array, n2_array

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
        
    squared_array_ = n2_array()

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
    #DEBUGGING: store mu and pesi
    #mu_iters = [mu]
    #res_iters =[epsy]
    niters = 0
    while res>epsilon and niters < max_iters:        
    
        res = np.linalg.norm(w - np.dot(mu,v))        
        v =  np.dot((1 / mu),w)
        w = apply_A(v)
        mu = np.linalg.norm(apply_A(v))
        #DEBUGGING: store mu and residue each iterations
        #mu_iters.append(mu)
        #res_iters.append(res)
        niters+=1
            
    #DEBUGGING: output mu and residue of all iterations
    return mu, v, niters, #mu_iters, res_iters

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
    
    
    return x, niters

def smallest_eigenvalue():
    A:np.ndarray

    def apply_A(v):
        assert isinstance(v,np.ndarray) , "v must be an np.ndarray"
        assert v.shape==A.shape , "v has shape "+str(v.shape)+", it must have shape "+str(A.shape)
        return np.asarray(np.dot(A,v.flatten())).reshape(A.shape)
    
    b:np.ndarray
    power_method(b.shape,conjugate_gradient,)

