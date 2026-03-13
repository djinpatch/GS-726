import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe, ellipkm1

# constants 
pi = np.pi
mu0 = 4 * pi * 1e-7
const = 1.0 / (4.0 * pi)

def denom(R, Z, Rp, Zp):
    return (R + Rp)**2 + (Z - Zp)**2

def ksquare(R, Z, Rp, Zp):
    return 4.0 * R * Rp / denom(R, Z, Rp, Zp)

def G(R, Z, Rp, Zp, eps=1e-14, switch=1e-8):
    m = ksquare(R, Z, Rp, Zp)

    # keep m in its mathematically valid interval
    m = np.clip(m, 0.0, 1.0 - eps)

    K = np.empty_like(m)
    near1 = (1.0 - m) < switch
    K[near1] = ellipkm1(1.0 - m[near1])
    K[~near1] = ellipk(m[~near1])

    E = ellipe(m)

    return const * np.sqrt(denom(R, Z, Rp, Zp)) * ((2.0 - m) * K - 2.0 * E)

def build_Gfunc(R, Z, Rp, Zp):
    Rb, Zb, Rpb, Zpb = np.meshgrid(R, Z, Rp, Zp, indexing='ij')
    Gfunc = G(Rb, Zb, Rpb, Zpb)
    return Gfunc


def apply_Gfunc(src, R, Z, blockR=25, blockZ=25):
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    NRblocks = len(R) / blockR
    NZblocks = len(Z) / blockZ
    psi = np.zeros_like(src)
    for i in range(NRblocks):
        Rblock = R[i*blockR:(i+1)*blockR]
        for j in range(NZblocks):
            Zblock = Z[j*blockZ:(j+1)*blockZ]
            Gblock = build_Gfunc(R, Z, Rblock, Zblock)
            srcblock = src[i*blockR:(i+1)*blockR,:][:,j*blockZ:(j+1)*blockZ]
            psi += np.einsum('ijkl,kl->ij', Gblock, srcblock) * dR * dZ
    return psi


def prof(a0, psi, nu=2):
    deltapsi = psi.max() - psi.min()
    psihat = (psi - psi.min()) / deltapsi
    return a0 * (1 - psihat**2)**nu


def profprime(a0, psi, nu=2):
    deltapsi = psi.max() - psi.min()
    psihat = (psi - psi.min()) / deltapsi
    return -2 * psihat * nu * a0 * (1 - psihat**2)**(nu-1) / deltapsi


def get_src(p0, F0, Rp, psi, nu=2):
    pprime = profprime(p0, psi, nu=nu)
    F = prof(F0, psi, nu=nu)
    Fprime = profprime(F0, psi, nu=nu)
    src = Rp * pprime + F*Fprime/Rp

    return src

NR, NZ = 400, 400
R = np.linspace(1.0, 2.0, NR)
Z = np.linspace(-0.5, 0.5, NZ)
RR, ZZ = np.meshgrid(R, Z, indexing="ij")

psi_guess = ...
p0 = 3.
F0 = 3.

src = get_src(p0,F0,RR,psi_guess)
psi = apply_Gfunc(src, R, Z, blockR=25, blockZ=25)


