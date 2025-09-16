
################################################################
#
# Exact diagonalization using sparse matrix method
#
################################################################
import numpy as np
import scipy.sparse.linalg as spa  #Uses sparse matrix technique
#from matrices import *
from ED_NH_func import *
import matplotlib.pyplot as plt


#bc = bondary condition, OBC = Open Boundary condition, PBC = Periodic boundary condition
bc = 1 # ( bc = 0 for OBC, bc = 1 for PBC, bc = 2 for APBC )
N = 10 # number of spins
c =1 #J

if bc == 0:
    print('Using OBC')  # Open Boundary Condition
elif bc == 1:
    print('Using PBC')  # Periodic Boundary Condition
else:
    print('Using APBC')  # Antiperiodic Boundary Condition

def gs(nval,n,eta,epsi):  #ground state eigen value and vector
    #'nval' = nber of eigen values and vectors to be computed 
    # which='SA' computes the eigenvalues nearest to zero
    #'maxiter' sets the maximum number of iterations allowed for the
    #iterative solver to converge based on the size of the matrix H.
    #'v0' specifies an initial guess for the eigenvectors.'None' means that no initial guess is provided.
    #'tol' sets the tolerance for convergence of the iterative solver. When the residual
    #error in the eigenvalue equation falls below this value, the solver considers the solution converged.
	#g = eta  + i*epsi is the NH transverse field
    #'spa.eigsh' is for Hermitian matrices and for non-Hermitian matrices, use 'spa.eigs'
    # "which='SR'": This argument specifies that you want the eigenvalues with the smallest real part. You 
	# can adjust this based on which eigenvalues you're interested in (e.g., 'LR' for largest real part, 'LI' 
	# for largest imaginary part, 'LM': Largest magnitude, 'SM': Smallest magnitude).
    # 'complex128' ensures that complex numbers are stored using 128 bits (64 bits for the real part and 64 bits 
	# for the imaginary part), providing double precision for both real and imaginary components.
    
    #Functions
    trfield_odd = TrField(n,False).astype(np.complex128) #Kinetic 
    trfield_even = TrField(n,True).astype(np.complex128) #Kinetic
    ising_term = NNterm(c,n, bc).astype(np.complex128)  #interacting
    g = np.complex128(eta + 1j*epsi)
    g1 = np.complex128(eta - 1j*epsi)
    H_problem = -ising_term
    H_driver  = -g*trfield_even -g1*trfield_odd  # g, g1 = transverse field on sublattice "o" and "." respectively
    H = (H_problem + H_driver).astype(np.complex128) #non-Hermitian matrix for which eigenvalues and eigenvectors are to be computed
    basis = generate_basis(N)
    # val, vec = spa.eigs(H,k=nval,which='SR',maxiter=n*1000000,v0=None,tol=1E-8, return_eigenvectors=True)#[0]
    import scipy
    val, vec = scipy.linalg.eig(H.todense())#[0]
    ground_state =vec[:,np.argmin(val)]
    val = sorted(val.tolist(), key = lambda x: x.real) #to sort the eigen values from the smallest real part
    # ground_state =vec[:,0]
    m = magnetization(ground_state, basis) #magnetization
    pe = np.dot(ground_state.conj().T, H_problem.dot(ground_state)) # Average potential energy per spin

    return val[0], val[1], val[2], vec,m,pe

file = open('NH_TFIM'+str(N)+'.txt', "w+") #create a file to save data

for n in [N]:
	cnt = 0
	lambd = 1.0 
	eta =0.0
	epsi=0.0
	
	for i in range(30+1):
		E0, E1, E2, vec, m, pe = gs(3,n,eta,epsi) # compute the ground and first two excited states
		file.write("{} {} {} {} {} {} {}\n".format(eta, epsi, E0, E1, E2, m, pe )) #write in the file s, E0, E1 and E2
		print(vec[:,0], eta, epsi, m)
		# plt.plot(vec[:,0])
		# break
		eta  += 0.1
		epsi += 0.01
file.close()

#----------------- Ground State Wave Function ----------------
fig, ax = plt.subplots(figsize=(10, 6))

# Get eigenvalues and eigenvectors for the current eta, epsi
val1, val2, val3, vec, m, pe = gs(3, n, 0, 0)

# Extract the ground state eigenvector (the first column of vec)
ground_state = -vec[:, 0]  # The first column corresponds to the ground state eigenvector
basis = generate_basis(N)
m = magnetization(ground_state, basis) #magnetization
#print(vec[:,0])
#print('===', m)
# Plot the real part of the ground state eigenvector

plt.plot(np.real(ground_state), label="Ground state")

# Label axes
ax.set_xlabel(r'Basis index', fontsize=25)
ax.set_ylabel('Amplitude', fontsize=25)
plt.legend()

# Save and display the plot
plt.savefig('NH_ground_state.png')
plt.show()

print('eta:', eta, 'epsi:', epsi)
print('Job done !!!')


#Ignore everything bellow

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from ED_NH_func import *  # Your model-specific functions

# ---------------- Parameters ----------------
bc = 1   # 0=OBC, 1=PBC, 2=APBC
N = 10   # number of spins
c = 1    # Ising coupling

def gs(n, eta, epsi):
    # Build operators
    trfield_odd = TrField(n, False).astype(np.complex128)
    trfield_even = TrField(n, True).astype(np.complex128)
    ising_term = NNterm(c, n, bc).astype(np.complex128)

    g = np.complex128(eta + 1j * epsi)
    g1 = np.complex128(eta - 1j * epsi)

    H_problem = -ising_term
    H_driver  = -g * trfield_even - g1 * trfield_odd
    H = (H_problem + H_driver).astype(np.complex128)

    basis = generate_basis(n)

    # Dense diagonalization
    vals, vecs = la.eig(H.todense())

    # Find index of smallest real part
    idx = np.argmin(vals.real)
    E0 = vals[idx]
    gs_vec = vecs[:, idx]

    # Magnetization & potential energy for that eigenvector
    m = magnetization(gs_vec, basis)
    pe = np.dot(gs_vec.conj().T, H_problem.dot(gs_vec))

    return E0, gs_vec, m, pe

# ---------------- Scan loop ----------------
etas = []
epsis = []
E0_real = []
E0_imag = []

# Choose a base eta value to define xi
eta_base = 1.0

for xi_frac in [0.1, 0.5]:  # 10% and 50%
    eta = eta_base
    epsi = xi_frac * eta_base  # non-Hermitian parameter scaling
    E0, _, _, _ = gs(N, eta, epsi)
    print(f"xi fraction={xi_frac:.2f}  E0={E0.real:.4f} + {E0.imag:.4f}i")
    etas.append(eta)
    epsis.append(epsi)
    E0_real.append(E0.real)
    E0_imag.append(E0.imag)

# ---------------- Full sweep plot ----------------
xi_vals = np.linspace(0.0, 1.0, 30)  # sweep fraction from 0 to 1
E0_real_sweep = []
E0_imag_sweep = []

for xi_frac in xi_vals:
    eta = eta_base
    epsi = xi_frac * eta_base
    E0, _, _, _ = gs(N, eta, epsi)
    E0_real_sweep.append(E0.real)
    E0_imag_sweep.append(E0.imag)

plt.figure(figsize=(8,5))
plt.plot(xi_vals, E0_real_sweep, 'o-', label='Real(E0)')
plt.plot(xi_vals, E0_imag_sweep, 'o-', label='Imag(E0)')
plt.xlabel(r'$\xi / \eta$')
plt.ylabel('Eigenvalue')
plt.legend()
plt.tight_layout()
plt.show()
