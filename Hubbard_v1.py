import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ設定 ---
t = 1.0          # ホッピング
D = 2 * t        # 半帯域幅
U = 0.5          # クーロン相互作用
mu = U/2        # 半分占有 (n=1) の場合
eta = 1e-3j      # 小さい虚数幅（極小であれば帯域エッジの判定に影響しない）

# --- 周波数軸（実軸のみ） ---
omega_real = np.linspace(-D - U, D + U, 1000)
omega = omega_real + eta

# --- Hubbard I のセルフエネルギー ---
def sigma_hubbard_I(omega, U, n, mu):
    Gat = (1 - n) / (omega + mu) + n / (omega + mu - U)
    return omega + mu - 1.0 / Gat

n = 0.5  # 半分占有
Sigma = sigma_hubbard_I(omega, U, n, mu)

# --- 解析的半楕円 DOS 式によるスペクトル関数 ---
z = omega.real + mu - Sigma.real
A = np.zeros_like(omega_real)
mask = np.abs(z) <= D
A[mask] = np.sqrt(D**2 - z[mask]**2) / (2 * np.pi * t**2)

# --- プロット ---
plt.figure(figsize=(6, 4))
plt.plot(omega_real, A, lw=1.5)
plt.axvline(-U/2, ls='--', color='gray', alpha=0.7)
plt.axvline( U/2, ls='--', color='gray', alpha=0.7)
plt.xlabel("ω")
plt.ylabel("A(ω)")
plt.title("Analytic Semi-Elliptic DOS with Hubbard I (U={:.1f})".format(U))
plt.tight_layout()
plt.show()
