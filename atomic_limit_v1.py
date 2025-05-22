#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atomic_limit_dos_square.py

ハバード模型の原子極限における状態密度をプロットし、
EPS形式で出力するスクリプト（正方形出力、U=1.0のみ）。

- Lorentzian broadening η を使ってピークを連続関数として近似
- Uの値は1.0のみ
- 軸の目盛り（ticks）は非表示
- 2つのピーク位置に縦の点線を追加
- 出力ファイル名：dos_atomic_limit_square.eps, dos_atomic_limit_square.png
"""
import numpy as np
import matplotlib.pyplot as plt

def atomic_dos(omega, U, ed, n=1.0, eta=0.01):
    """
    原子極限のグリーン関数から状態密度を計算
    G(ω) = (1 - n/2)/(ω + ed + iη) + (n/2)/(ω + ed - U + iη)
    DOS(ω) = -1/π Im G(ω)
    """
    w = omega + 1j * eta
    G1 = (1 - n/2) / (w + ed)
    G2 = (n/2) / ((omega + ed) - U + 1j * eta)
    G = G1 + G2
    return -1.0 / np.pi * np.imag(G)

# メイン処理
def main():
    # プロット領域と解像度
    omega = np.linspace(-2.0, 1.0, 2000)
    U = 1.0
    ed = 1.0
    eta = 0.05
    n = 1.0

    dos = atomic_dos(omega, U, ed, n=n, eta=eta)

    # 正方形の図を作成
    plt.figure(figsize=(6, 6))
    plt.plot(omega, dos, lw=1.5)

    # ピーク位置に縦点線
    peak_positions = [-ed, U - ed]
    for pos in peak_positions:
        plt.axvline(x=pos, linestyle='--', linewidth=1)

    # 軸目盛りを非表示
    plt.xticks([])
    plt.yticks([])

    # タイトルは不要なら削除
    # plt.title('Atomic limit DOS (U=1.0)')

    plt.tight_layout()
    # EPSおよびPNGで保存
    plt.savefig('dos_atomic_limit_square.eps', format='eps', bbox_inches='tight')
    plt.savefig('dos_atomic_limit_square.png', format='png', bbox_inches='tight')
    plt.show()
    print('Saved dos_atomic_limit_square.eps and .png')

if __name__ == '__main__':
    main()