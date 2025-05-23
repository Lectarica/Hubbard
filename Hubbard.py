import numpy as np
import matplotlib.pyplot as plt

# User-defined parameters
U_list = [0.0, 0.5, 1.0, 2.0]    # Hubbard U values to plot
n = 0.1                           # Filling per spin (0 < n < 1)
t_hop = 1.0                       # Nearest-neighbor hopping
delta = 1e-1                   # Broadening
Nk = 200                         # k-space sampling per dimension
tol_mu = 1e-4                    # Accuracy for mu search

# k-grid for 2D square lattice
kx = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
# ky = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
KX = np.meshgrid(kx)
# Dispersion ε_k
ek = -2.0 * t_hop * (np.cos(KX))

# Function to compute non-interacting filling per spin for given mu
def filling_noninteracting(mu, ek):
    # Zero-temperature filling: fraction of states with ε_k <= mu
    return np.mean(ek <= mu)

# Find mu that gives desired filling n (non-interacting) via bisection
def find_mu_for_n(n, ek, tol=tol_mu):
    mu_low = ek.min() - 1e-6
    mu_high = ek.max() + 1e-6
    f_low = filling_noninteracting(mu_low, ek)
    f_high = filling_noninteracting(mu_high, ek)
    # Ensure bounds
    if not (f_low <= n <= f_high):
        raise ValueError("Desired filling outside achievable range")
    while mu_high - mu_low > tol:
        mu_mid = 0.5 * (mu_low + mu_high)
        f_mid = filling_noninteracting(mu_mid, ek)
        if f_mid < n:
            mu_low = mu_mid
        else:
            mu_high = mu_mid
    return 0.5 * (mu_low + mu_high)

# Determine chemical potential for given n
mu = find_mu_for_n(n, ek)
print(f"Non-interacting chemical potential mu for filling n={n}: {mu:.4f}")

# Hubbard I self-energy (depends on U and n)
def Sigma_HI(z, U, n):
    return n * U + (n * (1 - n) * U**2) / (z - (1 - n) * U)

# Frequency grid
omegas = np.linspace(-6 * t_hop, 6 * t_hop, 400)

# Compute and plot DOS for each U
plt.figure()
for U in U_list:
    mu = U/2
    DOS = np.zeros_like(omegas)
    for idx, omega in enumerate(omegas):
        z = omega + mu + 1j * delta
        Sigma = Sigma_HI(z, U, n)
        Gk = 1.0 / (z - ek - Sigma)
        DOS[idx] = -np.mean(Gk.imag) / np.pi
    plt.plot(omegas, DOS, label=f'U = {U}')

# Finalize plot
plt.xlabel(r'$\omega$')
plt.ylabel('DOS')
plt.title(f'DOS (Hubbard I, n={n}, mu={mu:.3f})')
plt.legend()
plt.tight_layout()
plt.show()
