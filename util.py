from inputs import *

def loop_psi_BR_BZ(R, Z, Rc, Zc, I=1.0, sign_convention="-"):
    """
    Field of one axisymmetric toroidal filament current (circular loop).

    Parameters
    ----------
    R, Z : array-like
        Evaluation points.
    Rc, Zc : float
        Coil location in the poloidal plane.
    I : float
        Coil current.
    sign_convention : str
        "-"  -> formulas for Delta*psi = -mu0 R j_phi  (standard)
        "+"  -> flipped sign for Delta*psi = +mu0 R j_phi

    Returns
    -------
    psi, BR, BZ
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)

    zeta = Z - Zc
    beta2 = (R + Rc)**2 + zeta**2
    beta = np.sqrt(beta2)
    alpha2 = (R - Rc)**2 + zeta**2

    k2 = 4 * R * Rc / beta2

    # Avoid evaluating exactly on the filament singularity
    k2 = np.clip(k2, 0.0, 1.0 - 1e-14)
    alpha2_safe = np.where(alpha2 == 0, np.nan, alpha2)

    K = ellipk(k2)   # scipy uses parameter m = k^2
    E = ellipe(k2)
    k = np.sqrt(k2)

    psi = (mu0 * I / (2 * np.pi)) * (np.sqrt(R * Rc) / k) * ((2 - k2) * K - 2 * E)

    BR = (mu0 * I / (2 * np.pi)) * (
        zeta / (R * beta)
    ) * (
        -K + ((Rc**2 + R**2 + zeta**2) / alpha2_safe) * E
    )

    BZ = (mu0 * I / (2 * np.pi)) * (
        1 / beta
    ) * (
        K + ((Rc**2 - R**2 - zeta**2) / alpha2_safe) * E
    )

    # If user wants Delta*psi = +mu0 R j_phi, flip sign
    if sign_convention == "+":
        psi = -psi
        BR = -BR
        BZ = -BZ

    return psi, BR, BZ


def multi_coil_fields(R, Z, coils, sign_convention="-"):
    """
    Sum fields from many coils.

    coils = [(Rc, Zc, I), ...]
    """
    psi_tot = np.zeros_like(R, dtype=float)
    BR_tot = np.zeros_like(R, dtype=float)
    BZ_tot = np.zeros_like(R, dtype=float)

    for Rc, Zc, I in coils:
        psi, BR, BZ = loop_psi_BR_BZ(R, Z, Rc, Zc, I=I, sign_convention=sign_convention)
        psi_tot += psi
        BR_tot += BR
        BZ_tot += BZ

    return psi_tot, BR_tot, BZ_tot
    

def psi_on_circle(R0,a,coils,N =100):
    '''
    Use to estimate a reasonable psi from external coils
    '''

    # Make a circle of radius a at R0
    theta = np.linspace(0,2*np.pi,N)
    Rs = R0 + a *np.cos(theta)
    Zs = a * np.sin(theta)

    #eval flux 
    psi, BR, BZ = multi_coil_fields(Rs, Zs, coils)

    return psi
    

def initial_psi_plasma(R, Z, psi_coils, R0, a, psi0):
    '''
    This initial guess is circular surfaces. 
    R, Z : 2D arrays from np.meshgrid()   
    psi_coils : psi from coils on RZ grid. This is required so psi_total is as desired
    R0 : plasma center
    a: plasma circle max radius
    '''
    r2 = (R - R0)**2 + Z**2
    rho2 = r2 / a**2

    psi_target = psi0 * rho2

    psi_plasma = np.zeros_like(R)

    inside = rho2 <= 1.0

    psi_plasma[inside] = psi_target[inside] - psi_coils[inside]

    return psi_plasma


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



# const = 1.0 / (4.0 * pi)

# def denom(R, Z, Rp, Zp):
#     return (R + Rp)**2 + (Z - Zp)**2

# def norm2(R, Z, Rp, Zp):
#     return (R - Rp)**2 + (Z - Zp)**2

# def ksquare(R, Z, Rp, Zp):
#     return 4.0 * R * Rp / denom(R, Z, Rp, Zp)

# def G_stable(R, Z, Rp, Zp, eps=1e-14, switch=1e-8, diag_eps=None):
#     m = ksquare(R, Z, Rp, Zp)

#     if diag_eps is not None:
#         same = norm2(R, Z, Rp, Zp) < diag_eps**2
#         m = np.where(same, 1.0 - eps, m)

#     m = np.clip(m, eps, 1.0 - eps)

#     K = np.empty_like(m)
#     near1 = (1.0 - m) < switch
#     K[near1] = ellipkm1(1.0 - m[near1])
#     K[~near1] = ellipk(m[~near1])

#     E = ellipe(m)
#     k = np.sqrt(m)

#     return const * np.sqrt(denom(R, Z, Rp, Zp)) * ((2.0 - m) * K - 2.0 * E)



# def build_4D_Greensfunction(R_vals_np, Z_vals_np, device):
#     R4, Z4, Rp4, Zp4 = np.meshgrid(
#         R_vals_np, Z_vals_np, R_vals_np, Z_vals_np, indexing="ij"
#     )
#     dR = R_vals_np[1] - R_vals_np[0]
#     dZ = Z_vals_np[1] - Z_vals_np[0]
#     diag_eps = 0.5 * np.sqrt(dR**2 + dZ**2)
#     G4_np = G(R4, Z4, Rp4, Zp4, diag_eps=diag_eps)
#     G4 = torch.from_numpy(G4_np).to(device=device, dtype=torch.float32) # returns a torch tensor
#     return G4, dR, dZ

# def apply_greens(src, G4, dR, dZ, device, dtype=torch.float32):
#     """
#     Apply the Green's function to a 2D source.

#     Parameters
#     ----------
#     src : torch.Tensor or np.ndarray
#         Shape (NR, NZ). Source defined on the (Rp, Zp) grid.

#     Returns
#     -------
#     psi : torch.Tensor
#         Shape (NR, NZ). Field defined on the (R, Z) grid.
#     """
#     if isinstance(src, np.ndarray):
#         src = torch.from_numpy(src)

#     src = src.to(device=device, dtype=dtype)

#     if src.ndim != 2:
#         raise ValueError(f"src must be 2D, got shape {tuple(src.shape)}")

#     if src.shape != G4.shape[2:]:
#         raise ValueError(
#             f"src shape must be {G4.shape[2:]}, got {tuple(src.shape)}"
#         )
#     pdb.set_trace()
#     psi = torch.einsum("ijkl,kl->ij", G4, src) * dR * dZ # torch tensor defined on matrix [NR, NZ]

#     return psi