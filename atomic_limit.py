#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atomic_limit_dos.py

ハバード模型の原子極限における状態密度をプロットし、
EPS形式で出力するスクリプト。

- Lorentzian broadening η を使って離散ピークを連続関数として近似
- Uの値をリストで指定可能
- 出力ファイル名：dos_atomic_limit.eps
"""

import numpy as np
import matplotlib.pyplot as plt

def atomic_dos(omega, U, ed, n=1.0, eta=0.01):
    """
    原子極限のグリーン関数から状態密度を計算
    G(ω) = (1 - n/2)/(ω + iη) + (n/2)/(ω - U + iη)
    DOS(ω) = -1/π Im G(ω)
    """
    # フィリングn=1で半分ずつ寄与
    w = omega + 1j * eta
    G1 = (1 - n/2) / (w + ed)
    G2 = (n/2) / ((omega + ed) - U + 1j * eta)
    G = G1 + G2
    dos = -1.0 / np.pi * np.imag(G)
    return dos

def main():
    # プロット範囲と解像度
    omega = np.linspace(-3.0, 5.0, 2000)
    # プロットしたいUのリスト（例）
    U_list = [0.0, 0.5, 1.0, 2.0]
    # Lorentzian broadening η
    eta = 0.05
    # フィリング n（半充填: n=1）
    n = 1.0

    ed = 1.0

    plt.figure(figsize=(6, 4))
    for U in U_list:
        dos = atomic_dos(omega, U, ed, n=n, eta=eta)
        plt.plot(omega, dos, lw=1.5, label=f'$U={U}$')

    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$-\frac{1}{\pi}\,\mathrm{Im}\,G(\omega)$')
    plt.title('Atomic limit DOS of Hubbard model')
    plt.legend()
    plt.tight_layout()

    # EPS形式で保存
    plt.savefig('dos_atomic_limit.eps', format='eps')
    plt.savefig('dos_atomic_limit.png', format='png')
    plt.show()
    print('Saved plot to dos_atomic_limit.eps')

if __name__ == '__main__':
    main()
