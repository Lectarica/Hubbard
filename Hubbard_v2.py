import numpy as np
import matplotlib.pyplot as plt

t = 1.0
D = 2 * t
U = 1.0
mu = U/2
eta = 0.001
n = 0.2

Reomega = np.linspace(-D-U,D+U,1000)
omega = Reomega + eta*1j

def Sigma_Hubbard_I(omega, U, n, mu):
    Gatom = (1-n)/(omega+mu)+n/(omega + mu - U)
    return omega + mu - 1.0/Gatom

Se = Sigma_Hubbard_I(omega, U, n, mu)

z = omega.real + mu - Se.real
z0 = omega.real + mu*0
rho=np.zeros_like(Reomega)
rho0=np.zeros_like(Reomega)
mask = np.abs(z) <= D
rho[mask] = np.sqrt(D**2 - z[mask]**2)/(2*np.pi*t**2)
mask = np.abs(z0) <= D
rho0[mask] = np.sqrt(D**2 - z0[mask]**2)/(2*np.pi*t**2)
plt.figure(figsize=(6, 4))
plt.plot(Reomega, rho, lw=1.5, label='U = 1.0')
plt.plot(Reomega, rho0, lw=1.5, label='U = 0.0')
plt.axvline(-U/2, ls='--', color='gray', alpha=0.7)
plt.axvline( U/2, ls='--', color='gray', alpha=0.7)
plt.xlabel("ω")
plt.ylabel(r"rho($\omega$)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()