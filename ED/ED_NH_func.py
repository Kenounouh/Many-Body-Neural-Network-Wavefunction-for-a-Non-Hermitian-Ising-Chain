################################################################
#
# Exact diagonalization using sparse matrix method
#
################################################################

import scipy.sparse as sp
from functools import reduce
import numpy as np
#
#	**********  BASIC MATRIX DEFINITIONS  **********
#

Sz = sp.dia_matrix([[1.0,0.0],[0.0,-1.0]]) #Pauli matrix
Sx = sp.csc_matrix([[0.0,1.0],[1.0,0.0]])
Sy = sp.csc_matrix( [[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]] )


def Id(n): #identity  matrix of size 2^n
	return sp.eye(2**n)

def Szi(i,n): #Sz acting on the i^th site of a system of n sites
	A = Id(i) # Identity matrix of size 2^i acting to the left of the i^th site
	B = Id(n-i-1) # Identity matrix of size 2^(n-i-1) acting to the rigth of the i^th site
	D = reduce(sp.kron, [A,Sz,B])
    # Kronecker product (kron) iteratively over a list consisting of A, Sz and B and yields 
    #the matrix representation of Sz acting on the i-th site embedded in a larger system of n sites.
	return D

def Sxi(i,n): # Sx
	A = Id(i)  # Id(i)----->i^th<--------------Id(n-i-1)
	B = Id(n-i-1)
	D = reduce(sp.kron, [A,Sx,B])
	return D

def Syi(i,n):
	A = Id(i)
	B = Id(n-i-1)
	D = reduce(sp.kron, [A,Sy,B])
	return D

#
#	**********  HAMILTONIANS  **********
#

#--------------Interaction Hamiltonian---------
def interaction(c,i,j,n): #interaction term
	if (i == j):
		print("i and j must be distinct!")
	elif (j<i):
		return interaction(c,j,i,n)
	else:
		A = Id(i)
		B = Id(j-i-1)
		C = Id(n-j-1)
		D = reduce(sp.kron, [A,Sz,B,Sz,C]) #Sz_i*Sz_i+1
	return c*D #'c' = coupling J=1 

#------Impose Boundary conditions on interaction Hamiltonian---------
def NNterm(c,n,bc):
	if(bc == 0): #OBC
		#print("Using Open BC")
		for i in range(n-1):
			if(i == 0):
				H = interaction(c,i,i+1,n)  #J=1
			else:
				H = H + interaction(c,i,i+1,n) #sumation of interaction term
		return H
	if(bc==1): #PBC
		#print("Using Periodic BC")
		for i in range(n):
			if(i == 0):
				H = interaction(c,i,i+1,n)
			else:
				H = H + interaction(c,i,(i+1)%n,n)
		return H
	if(bc==2): #APBC
		#print("Using Anti-Periodic BC")
		for i in range(n):
			if(i == 0):
				H = interaction(c,i,i+1,n)
			if(i == n-1):
				H = H - interaction(c,i,0,n)
			if(i != 0 and i != n-1):
				H = H + interaction(c,i,i+1,n)
		return H

def TrField(n,even):  #kinetic or transverse field part for spin on sublattice "o"
	for i in range(n):
		if (((i%2)==0)==even): # apply the field on the corresponding site
			if (i == 0 or i == 1):
				F = Sxi(i,n)
			else:
				F = F + Sxi(i,n)
	return F

#Let's now compute the magnetization

# Binary BitArray representation of the given integer `num`, padded to length `N`.
def bit_rep(num, N):
    """
    Converts the given integer `num` to a binary bit array (list of bools), 
    padded to length N.
    """
    # Get the binary representation of `num`, padded to length `N`, and convert to list of bools
    return np.array([bool(int(b)) for b in np.binary_repr(num, width=N)])

# Generates a basis spanning the Hilbert space of `N` spins.
def generate_basis(N):
    """
    Generate a basis (list of bit arrays) for the Hilbert space of N spins.
    """
    nstates = 2 ** N
    basis = []
    for i in range(nstates):
        basis.append(bit_rep(i, N))
    return basis

 

def magnetization(state, basis):
    """
    Calculate the magnetization of the given quantum state.

    Parameters:
    - state: A numpy array of coefficients corresponding to the quantum state in the given basis.
    - basis: A list of spin configurations (each configuration is a list of boolean values).

    Returns:
    - M: The total magnetization.
    """
    M = 0.0  # Initialize total magnetization

    for i, bstate in enumerate(basis):  # Loop over basis states
        bstate_M = 0.0  # Magnetization for the current basis state
        for spin in bstate:
            # Add contribution from each spin in the basis state
            bstate_M += (state[i]**2 * (1 if spin else -1)) / len(bstate)
        
        # Assert that the magnetization per state is within bounds (-1 to 1)
        assert abs(bstate_M) <= 1
        M += abs(bstate_M)  # Add absolute value of magnetization for the basis state

    return M
import numpy as np

def localenergy(N, J, g_real, g_imag, samples, logpsi, compute_logpsi):
    """
    Computes the local energy for the staggered non-Hermitian transverse field Ising model.
    
    Parameters:
    - N: Number of spins.
    - J: Coupling constant.
    - g_real: Real part of the transverse field.
    - g_imag: Imaginary part of the transverse field.
    - samples: Sample configurations (shape: [batch_size, N]).
    - logpsi: Log wavefunction values for the samples (shape: [batch_size]).
    - compute_logpsi: Function to compute log wavefunction for given samples.

    Returns:
    - eloc: Local energy values (shape: [batch_size]).
    """
    eloc = np.zeros(samples.shape[0], dtype=np.complex64)

    # Diagonal contribution: Classical interaction term
    for n in range(N-1):
        eloc += -J * (2 * samples[:, n] - 1) * (2 * samples[:, (n + 1) % N] - 1)

    # Off-diagonal contributions from staggered transverse fields
    g_A = np.complex(g_real, g_imag)  # Complex g for sites in A
    g_B = np.complex(g_real, -g_imag)  # Complex conjugate for sites in B

    for j in range(N):
        # Flip the j-th spin
        flip_samples = np.copy(samples)
        flip_samples[:, j] = 1 - samples[:, j]

        # Compute log wavefunction for flipped samples
        flip_logpsi = compute_logpsi(flip_samples)

        # Contribution from sites in A and B
        if j % 2 == 0:  # Even index (site in A)
            eloc += -g_A * np.exp(flip_logpsi - logpsi)
        else:  # Odd index (site in B)
            eloc += -g_B * np.exp(flip_logpsi - logpsi)

    return eloc

