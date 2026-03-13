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

def prof(a0, psi, nu):
    return a0 * (1 - psi**2)**nu
