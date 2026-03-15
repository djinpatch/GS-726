from math import nan

from inputs import *


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

### test precomple + apply_Gfunc_blocks() for speed ups / bugs
def precompile_blocks(R, Z, blockR=25, blockZ=25):
    NR = len(R)
    NZ = len(Z)

    if NR % blockR != 0 or NZ % blockZ != 0:
        raise ValueError("blockR and blockZ must divide len(R) and len(Z) exactly")

    NRblocks = NR // blockR
    NZblocks = NZ // blockZ

    Gblocks = []

    for i in range(NRblocks):
        Rblock = R[i*blockR:(i+1)*blockR]
        for j in range(NZblocks):
            Zblock = Z[j*blockZ:(j+1)*blockZ]
            Gblock = build_Gfunc(R, Z, Rblock, Zblock)
            Gblocks.append(Gblock)

    return np.array(Gblocks)

def apply_Gfunc_blocks(src, R, Z, Gblocks, blockR=25, blockZ=25):
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    NRblocks = len(R) // blockR
    NZblocks = len(Z) // blockZ
    psi = np.zeros_like(src)

    n = 0
    for i in range(NRblocks):
        for j in range(NZblocks):
            Gblock = Gblocks[n]
            srcblock = src[i*blockR:(i+1)*blockR, j*blockZ:(j+1)*blockZ]
            psi += np.einsum('ijkl,kl->ij', Gblock, srcblock) * dR * dZ
            n += 1

    return psi 

####

def prof(a0, psi, psimask, nu=2):
    psivalid = psi[psimask]
    deltapsi = psivalid.max() - psivalid.min()
    psihat = (psi - psi.min()) / deltapsi
    profile = a0 * (1 - psihat**2)**nu
    profile = np.where(psimask, profile, 0.0)
    return profile


def profprime(a0, psi, psimask, nu=2):
    psivalid = psi[psimask]
    deltapsi = psivalid.max() - psivalid.min()
    psihat = (psi - psi.min()) / deltapsi
    profileprime = -2 * psihat * nu * a0 * (1 - psihat**2)**(nu-1) / deltapsi
    profileprime = np.where(psimask, profileprime, 0.0)
    return profileprime

def get_src(p0, F0, Rp, psi, psimask, nu=2):
    pprime = profprime(p0, psi, psimask, nu=nu)
    F = prof(F0, psi, psimask, nu=nu)
    Fprime = profprime(F0, psi, psimask, nu=nu)
    src = Rp * pprime + F*Fprime/Rp

    return src


def multi_coil_fields(R, Z, coils):
    """
    Sum fields from mutiple coils from a list
    coils = [(Rc, Zc, I), ...]
    """
    psi_coils = np.zeros_like(R, dtype=float)
    for Rc, Zc, I in coils:
        psi = mu0 * I * G(R, Z, Rc, Zc)
        psi_coils += psi
    return psi_coils


def psi_on_circle(R0,a,coils,N =100):
    '''
    Use to estimate a reasonable psi from external coils
    '''

    # Make a circle of radius a at R0
    theta = np.linspace(0,2*np.pi,N)
    Rs = R0 + a *np.cos(theta)
    Zs = a * np.sin(theta)

    #eval flux 
    psi = multi_coil_fields(Rs, Zs, coils)

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



########################### THANK YOU CHATGPT ##############################
def polygon_area(vertices):
    """
    Compute polygon area using the shoelace formula.

    Parameters
    ----------
    vertices : ndarray, shape (N, 2)

    Returns
    -------
    area : float
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_closed_vertices(vertices, tol=1e-8):
    """
    Check whether a contour is closed by comparing first and last vertex.

    Parameters
    ----------
    vertices : ndarray, shape (N, 2)
    tol : float

    Returns
    -------
    closed : bool
    """
    return np.allclose(vertices[0], vertices[-1], atol=tol, rtol=0)


def touches_boundary(vertices, Rmin, Rmax, Zmin, Zmax, tol=1e-8):
    """
    Check whether a contour touches the computational boundary.

    Parameters
    ----------
    vertices : ndarray, shape (N, 2)
    Rmin, Rmax, Zmin, Zmax : float
    tol : float

    Returns
    -------
    touches : bool
    """
    Rv = vertices[:, 0]
    Zv = vertices[:, 1]

    return (
        np.any(np.isclose(Rv, Rmin, atol=tol)) or
        np.any(np.isclose(Rv, Rmax, atol=tol)) or
        np.any(np.isclose(Zv, Zmin, atol=tol)) or
        np.any(np.isclose(Zv, Zmax, atol=tol))
    )


def point_in_polygon(point, vertices):
    """
    Check whether a point lies inside a closed polygon.

    Parameters
    ----------
    point : tuple (R, Z)
    vertices : ndarray, shape (N, 2)

    Returns
    -------
    inside : bool
    """
    return Path(vertices).contains_point(point)


def find_magnetic_axis(RR, ZZ, psi, axis="min"):
    """
    Find the magnetic axis approximately as the min or max of psi.

    Parameters
    ----------
    RR, ZZ : ndarray, shape (NR, NZ)
        Meshgrids produced with indexing='ij'
    psi : ndarray, shape (NR, NZ)
    axis : str
        'min' if the magnetic axis is at minimum psi
        'max' if the magnetic axis is at maximum psi

    Returns
    -------
    axis_info : dict
        {
            'R': float,
            'Z': float,
            'psi': float,
            'index': (i, j)
        }
    """
    if axis == "min":
        idx = np.unravel_index(np.argmin(psi), psi.shape)
    elif axis == "max":
        idx = np.unravel_index(np.argmax(psi), psi.shape)
    else:
        raise ValueError("axis must be 'min' or 'max'")

    return {
        "R": RR[idx],
        "Z": ZZ[idx],
        "psi": psi[idx],
        "index": idx,
    }


def get_closed_contours_at_level(RR, ZZ, psi, level, tol=1e-8):
    """
    Extract closed contours at a given psi level that do not touch
    the computational boundary.

    Parameters
    ----------
    RR, ZZ : ndarray, shape (NR, NZ)
    psi : ndarray, shape (NR, NZ)
    level : float
    tol : float

    Returns
    -------
    contours : list of dict
        each dict has keys:
        - 'level'
        - 'vertices'
        - 'area'
    """
    Rmin, Rmax = RR.min(), RR.max()
    Zmin, Zmax = ZZ.min(), ZZ.max()

    fig, ax = plt.subplots()
    cs = ax.contour(RR, ZZ, psi, levels=[level])

    contours = []

    # Works across matplotlib versions by accessing all paths from collections
    for coll in cs.collections:
        for path in coll.get_paths():
            vertices = path.vertices

            if len(vertices) < 3:
                continue

            if not is_closed_vertices(vertices, tol=tol):
                continue

            if touches_boundary(vertices, Rmin, Rmax, Zmin, Zmax, tol=tol):
                continue

            contours.append({
                "level": level,
                "vertices": vertices,
                "area": polygon_area(vertices),
            })

    plt.close(fig)
    return contours


def find_lcfs_from_axis(RR, ZZ, psi, axis="min", nlevels=200, tol=1e-8):
    """
    Find the LCFS as the outermost closed contour that encloses
    the magnetic axis and does not touch the boundary.

    Parameters
    ----------
    RR, ZZ : ndarray, shape (NR, NZ)
    psi : ndarray, shape (NR, NZ)
    axis : str
        'min' or 'max'
    nlevels : int
        Number of candidate contour levels to scan
    tol : float

    Returns
    -------
    lcfs : dict or None
        If found:
        {
            'level': float,
            'vertices': ndarray (N,2),
            'area': float,
            'axis': axis_info_dict
        }
        Otherwise None
    """
    axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis)
    Raxis = axis_info["R"]
    Zaxis = axis_info["Z"]
    psiaxis = axis_info["psi"]

    psimin = psi.min()
    psimax = psi.max()

    if np.isclose(psimax, psimin):
        return None

    if axis == "min":
        levels = np.linspace(psiaxis, psimax, nlevels + 2)[1:-1]
    else:
        levels = np.linspace(psimin, psiaxis, nlevels + 2)[1:-1][::-1]

    last_good = None

    for level in levels:
        contours = get_closed_contours_at_level(RR, ZZ, psi, level, tol=tol)

        enclosing = []
        for c in contours:
            if point_in_polygon((Raxis, Zaxis), c["vertices"]):
                enclosing.append(c)

        if len(enclosing) == 0:
            continue

        # If there are multiple closed contours at this level,
        # take the largest one that still encloses the axis.
        best_here = max(enclosing, key=lambda c: c["area"])
        last_good = best_here

    if last_good is None:
        return None

    last_good["axis"] = axis_info
    return last_good


def make_lcfs_mask(RR, ZZ, lcfs):
    """
    Create a boolean mask for points inside the LCFS.

    Parameters
    ----------
    RR, ZZ : ndarray, shape (NR, NZ)
    lcfs : dict returned by find_lcfs_from_axis

    Returns
    -------
    mask : ndarray, shape (NR, NZ), dtype=bool
    """
    if lcfs is None:
        return np.zeros_like(RR, dtype=bool)

    vertices = lcfs["vertices"]
    path = Path(vertices)

    points = np.column_stack([RR.ravel(), ZZ.ravel()])
    mask = path.contains_points(points).reshape(RR.shape)
    return mask


def plot_psi_axis_lcfs(RR, ZZ, psi, lcfs=None, ncontours=40):
    """
    Plot psi contours, magnetic axis, and LCFS.

    Parameters
    ----------
    RR, ZZ : ndarray, shape (NR, NZ)
    psi : ndarray, shape (NR, NZ)
    lcfs : dict or None
    ncontours : int
    """
    plt.figure(figsize=(6, 5))
    plt.contour(RR, ZZ, psi, levels=ncontours, colors="k", linewidths=0.6)

    if lcfs is not None:
        verts = lcfs["vertices"]
        axis_info = lcfs["axis"]

        plt.plot(verts[:, 0], verts[:, 1], "r-", linewidth=2, label="LCFS")
        plt.plot(axis_info["R"], axis_info["Z"], "bo", markersize=6, label="Magnetic axis")

        plt.title(f"LCFS level = {lcfs['level']:.6g}")
        plt.legend()

    plt.xlabel("R")
    plt.ylabel("Z")
    plt.tight_layout()
    plt.show()