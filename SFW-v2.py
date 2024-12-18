import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
import scipy.special as s
import cmath
import math
import cv2

##################### TRATAMENTO DA IMAGEM #######################################
image = cv2.imread('senoide.png') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
blurred = cv2.GaussianBlur(image, (5, 5), 0)  
(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
threshInv = np.flip(threshInv,0) 
threshInv = cv2.resize(threshInv, (276, 75))
F = threshInv.astype(float)  

###################### PARAMETROS ##########################
lamb = 632.0e-9 #metros
k = 2*np.pi/lamb
Q = 0.9998*k
NN = 15
MM = 300
LPX, HPX = threshInv.shape
L = 0.060
H = L*(HPX/LPX)

rho_min = 2.405/np.sqrt(k**2 - Q**2) 
delta_rho = H/MM 
phi_0 = 0 

### se entendi bem, pxH e pxL não é o mesmo que LPX e HPX?
### se sim, sugiro dar um Ctrl+H e dar um replace usar só um dos pares, para facilitar
pxH = len(threshInv)    
pxL = len(threshInv[0]) 

print("############ PARAMETROS ############")
print("Comprimento de onda no vácuo (m):", lamb)
print("Numero de Onda:", k)
print("2 NN + 1 feixes de Bessel para cada LFW:", NN)
print("Numero de LFWS:", MM)
print("Comprimento L (m):", L)
print("Parametro Q (1/m):", Q)
print("\n############ Espacamento entre as LFWs ############")
print("Largura do maior spot das LFWs:", rho_min)
print("Separacao radial entre as LFWs:", delta_rho)

plt.contourf(threshInv)

# Verificar divisibilidade para indexação
denom = (MM/pxH)
if not denom.is_integer():
    raise ValueError("MM/pxH não é inteiro, o que pode causar problemas de indexação.")
denom = int(denom)

# Definição de arrays para beta e k_rho:
n_array = np.arange(-NN, NN+1)
beta_array = Q + 2*np.pi*n_array/L
k_rho_array = np.sqrt(k**2 - beta_array**2)  # Pode ser complexo se beta>k

# rho_0 array
###rho0_array = np.array([m*delta_rho*(HPX/LPX) for m in range(1, MM+1)])
### Deve haver um erro com essa expressão. Imagine o caso m=MM, o certo seria que
###  rho0_array[MM] = H, mas na sua expressão, ele parece resultar em HPX²/LPX².
### Eu escreveria algo do tipo:
rho0_array = np.array([(m-1)*delta_rho for m in range(1, MM+1)])
### Note que a 1ª FW não estará em ρ=0, e a última estará em ρ=H.
###  Outra forma de definir seria np.array([(m-1)*delta_rho for m in range(1, MM+1)]),
###  assim, nesse caso, a 1ª FW estaria em ρ=0, mas a última em ρ=(MM-1)H. De qqr forma, tanto faz.

print("número de linhas da imagem (pxH):", pxH)
print("número de colunas da imagem (pxL):", pxL)

print("Calculando a matriz A...")
A = np.zeros((2*NN+1, pxH), dtype=complex)
for n_index, n in enumerate(n_array):
    a = 2 * np.pi * n / L
    for pxh_index in range(pxH):
        val = 0+0j
        for il in range(1, pxL):
            F_val = threshInv[pxh_index, il]
            z_start = (il-1)*L/(pxL-1)
            z_end   = il*L/(pxL-1)

            if n == 0:
                integral = (z_end - z_start)
            else:
                integral = (np.exp(1j * a * z_end) - np.exp(1j * a * z_start)) / (1j * a)
            
            val += F_val * integral
        A[n_index, pxh_index] = (1/L)*val

# Pré-calcular A_vals(m,n)
m_array = np.arange(1, MM+1)
col_indices = np.ceil(m_array/denom).astype(int) - 1
A_vals = np.zeros((MM, 2*NN+1), dtype=complex)
for m_i, ci in enumerate(col_indices):
    A_vals[m_i,:] = A[:, ci]

#####################################
# Agora de acordo com o código Mathematica:
# dρ = 3
# dz = 1
dRho = 3
dz = 1

# Construção de List3:
# ρρ = 1 até dρ*pxH = 3*pxH
rho_rho_indices = np.arange(1, dRho*pxH+1)
# zz = 1 até dz*pxL = pxL
zz_indices = np.arange(1, dz*pxL+1)

# ρ = (ρρ/dρ)*Δρ = (ρρ/3)*Δρ
rho_values = (rho_rho_indices/dRho)*delta_rho
# z = (zz/(dz*pxL))*L = (zz/pxL)*L
z_values = (zz_indices/(dz*pxL))*L

# Dimensão final de List3 será (3*pxH, pxL)
# Precisamos vetorizar novamente, desta vez:
# SΨ(ρ,z) = ∑_m ∑_n A_vals(m,n)*j0(k_rho[n]*arg(m,ρ))*exp(-i β[n] z)

# Criar m,ρ mesh
m_mesh, rho_mesh = np.meshgrid(m_array, rho_values, indexing='ij')  # (MM,3*pxH)
rho_mesh_m = rho_mesh # já em metros, pois rho_values está em metros? verifique a unidade.

# Atenção: delta_rho e L estão em metros. rho_values e z_values estão em metros também.
# delta_rho e L foram definidos em metros. Portanto rho_values em metros, z_values em metros.

# arg(m,ρ):
cos_phi_diff = np.cos(0 - phi_0)
rr = rho0_array[m_mesh-1]
arg = np.sqrt(rho_mesh_m**2 + rr**2 - 2*rho_mesh_m*rr*cos_phi_diff)  # (MM, 3*pxH)

# j0(k_rho[n]*arg(m,ρ))
arg_reshaped = arg[:,:,np.newaxis] # (MM,3*pxH,1)
k_rho_reshaped = k_rho_array[np.newaxis,np.newaxis,:] # (1,1,2*NN+1)
bessel_vals = s.jv(0, k_rho_reshaped*arg_reshaped) # (MM,3*pxH,2*NN+1)

# exp_factor(n,z)
exp_factor = np.exp(-1j * np.outer(beta_array, z_values)) # (2*NN+1, pxL)

# combined(m,ρ,n)
A_vals_reshaped = A_vals[:, np.newaxis, :] # (MM,1,2*NN+1)
combined = A_vals_reshaped * bessel_vals   # (MM,3*pxH,2*NN+1)

# Spsi(ρ,z) = ∑_m,n combined(m,ρ,n)*exp_factor(n,z)
Spsi_matrix = np.einsum('mrn,nz->rz', combined, exp_factor)
# Spsi_matrix(3*pxH, pxL) com ρ no primeiro índice e z no segundo índice

List3 = np.abs(Spsi_matrix)**2

#####################################
# Plotar com mesmo estilo do Mathematica:
# Eixo z: de L/pxL até L (já está em metros)
# Eixo ρ: de Δρ/3 até pxH*Δρ
z_min = z_values[0]      # L/pxL
z_max = z_values[-1]     # L
rho_min = rho_values[0]  # Δρ/3
rho_max = rho_values[-1] # pxH*Δρ
pxL_val = pxL
pxH_val = pxH

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='Times', size=20)

fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(List3, extent=(z_min, z_max, rho_min, rho_max),
               origin='lower',aspect=pxH_val/pxL_val,
               cmap='Greens', interpolation='bilinear')

fig.colorbar(im, label=r'$|\Psi_{SFW}(\rho,0,z)|^2$')
ax.set_xlabel(r'$z \,(\text{m})$')
ax.set_ylabel(r'$\rho \,(\text{m})$')
ax.set_title(r'$|\Psi_{SFW}|^2$')

plt.tight_layout()
plt.show()
