import numpy as np
import torch
from scipy.special import ellipk, ellipkm1, ellipe


# ----------------------------
# device
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)


# ----------------------------
# grid
# ----------------------------
NR, NZ = 40, 40

R_vals_np = np.linspace(1.0, 2.0, NR)
Z_vals_np = np.linspace(-0.5, 0.5, NZ)

R_vals = torch.from_numpy(R_vals_np).to(device=device, dtype=torch.float32)
Z_vals = torch.from_numpy(Z_vals_np).to(device=device, dtype=torch.float32)
RR, ZZ = torch.meshgrid(R_vals, Z_vals, indexing="ij")


# ----------------------------
# stable Green's function
# ----------------------------
pi = np.pi
const = 1.0 / (2.0 * pi)

def denom(R, Z, Rp, Zp):
    return (R + Rp)**2 + (Z - Zp)**2

def norm2(R, Z, Rp, Zp):
    return (R - Rp)**2 + (Z - Zp)**2

def ksquare(R, Z, Rp, Zp):
    return 4.0 * R * Rp / denom(R, Z, Rp, Zp)

def G_stable(R, Z, Rp, Zp, eps=1e-14, switch=1e-8, diag_eps=None):
    m = ksquare(R, Z, Rp, Zp)

    if diag_eps is not None:
        same = norm2(R, Z, Rp, Zp) < diag_eps**2
        m = np.where(same, 1.0 - eps, m)

    m = np.clip(m, eps, 1.0 - eps)

    K = np.empty_like(m)
    near1 = (1.0 - m) < switch
    K[near1] = ellipkm1(1.0 - m[near1])
    K[~near1] = ellipk(m[~near1])

    E = ellipe(m)
    k = np.sqrt(m)

    return const * np.sqrt(Rp / R) * (((2.0 - m) * K - 2.0 * E) / k)


# ----------------------------
# build full 4D Green tensor
# ----------------------------
def build_4D_Greensfunction(R_vals_np, Z_vals_np):
    R4, Z4, Rp4, Zp4 = np.meshgrid(
        R_vals_np, Z_vals_np, R_vals_np, Z_vals_np, indexing="ij"
    )
    dR = R_vals_np[1] - R_vals_np[0]
    dZ = Z_vals_np[1] - Z_vals_np[0]
    diag_eps = 0.5 * np.sqrt(dR**2 + dZ**2)
    G4_np = G_stable(R4, Z4, Rp4, Zp4, diag_eps=diag_eps)
    G4 = torch.from_numpy(G4_np).to(device=device, dtype=torch.float32)
    return G4, dR, dZ

G4, dR, dZ = build_4D_Greensfunction(R_vals_np, Z_vals_np)

# ----------------------------
# apply Green operator to any 2D source
# ----------------------------
def apply_greens(src, G4=G4, dR=dR, dZ=dZ, device=device, dtype=torch.float32):
    """
    Apply the Green's function to a 2D source.

    Parameters
    ----------
    src : torch.Tensor or np.ndarray
        Shape (NR, NZ). Source defined on the (Rp, Zp) grid.

    Returns
    -------
    phi : torch.Tensor
        Shape (NR, NZ). Field defined on the (R, Z) grid.
    """
    if isinstance(src, np.ndarray):
        src = torch.from_numpy(src)

    src = src.to(device=device, dtype=dtype)

    if src.ndim != 2:
        raise ValueError(f"src must be 2D, got shape {tuple(src.shape)}")

    if src.shape != G4.shape[2:]:
        raise ValueError(
            f"src shape must be {G4.shape[2:]}, got {tuple(src.shape)}"
        )

    phi = torch.einsum("ijkl,kl->ij", G4, src) * dR * dZ
    return phi


# ----------------------------
# example 1: Gaussian source
# ----------------------------
src1 = torch.exp(-(((RR - 1.5) / 0.15) ** 2 + (ZZ / 0.12) ** 2))
psi1 = apply_greens(src1)

print("src1 shape:", src1.shape)
print("psi1 shape:", psi1.shape)
print("psi1 device:", psi1.device)


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.contour(R_vals_np, Z_vals_np, psi1.detach().cpu().numpy().T, levels=30)
plt.xlabel("R")
plt.ylabel("Z")
plt.title(r"$\psi(R,Z)$")
plt.tight_layout()
plt.show()

