import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.special import ellipk, ellipe, ellipkm1

# ============================================================
# constants
# ============================================================
pi = np.pi
mu0 = 4.0e-7 * pi
const = 1.0 / (4.0 * pi)

# ============================================================
# Green's function pieces
# ============================================================
def denom(R, Z, Rp, Zp):
    return (R + Rp) ** 2 + (Z - Zp) ** 2


def ksquare(R, Z, Rp, Zp):
    return 4.0 * R * Rp / denom(R, Z, Rp, Zp)


def G(R, Z, Rp, Zp, eps=1e-14, switch=1e-8):
    m = ksquare(R, Z, Rp, Zp)
    m = np.clip(m, 0.0, 1.0 - eps)

    K = np.empty_like(m)
    near1 = (1.0 - m) < switch
    K[near1] = ellipkm1(1.0 - m[near1])
    K[~near1] = ellipk(m[~near1])
    E = ellipe(m)

    return const * np.sqrt(denom(R, Z, Rp, Zp)) * ((2.0 - m) * K - 2.0 * E)


def build_Gfunc(R, Z, Rp, Zp):
    Rb, Zb, Rpb, Zpb = np.meshgrid(R, Z, Rp, Zp, indexing="ij")
    return G(Rb, Zb, Rpb, Zpb)


def precompile_blocks(R, Z, blockR=25, blockZ=25):
    NR = len(R)
    NZ = len(Z)

    if NR % blockR != 0 or NZ % blockZ != 0:
        raise ValueError("blockR and blockZ must divide len(R) and len(Z) exactly")

    NRblocks = NR // blockR
    NZblocks = NZ // blockZ

    Gblocks = []
    for i in range(NRblocks):
        Rblock = R[i * blockR : (i + 1) * blockR]
        for j in range(NZblocks):
            Zblock = Z[j * blockZ : (j + 1) * blockZ]
            Gblocks.append(build_Gfunc(R, Z, Rblock, Zblock))

    return np.array(Gblocks)


def apply_Gfunc_blocks(src, R, Z, Gblocks, blockR=25, blockZ=25):
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    NRblocks = len(R) // blockR
    NZblocks = len(Z) // blockZ

    psi = np.zeros_like(src, dtype=float)

    n = 0
    for i in range(NRblocks):
        for j in range(NZblocks):
            Gblock = Gblocks[n]
            srcblock = src[i * blockR : (i + 1) * blockR, j * blockZ : (j + 1) * blockZ]
            psi += np.einsum("ijkl,kl->ij", Gblock, srcblock) * dR * dZ
            n += 1

    return psi


# ============================================================
# coil helpers
# ============================================================
def multi_coil_fields(R, Z, coils):
    """
    coils = [(Rc, Zc, I), ...]
    Returns total psi from axisymmetric toroidal filaments.
    """
    psi_coils = np.zeros_like(R, dtype=float)
    for Rc, Zc, I in coils:
        psi_coils += mu0 * I * G(R, Z, Rc, Zc)
    return psi_coils


def psi_on_circle(R0, a, coils, N=200):
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    Rs = R0 + a * np.cos(theta)
    Zs = a * np.sin(theta)
    return multi_coil_fields(Rs, Zs, coils)


# ============================================================
# plasma mask and profiles
# ============================================================
def circle_mask(R, Z, R0, a):
    return (R - R0) ** 2 + Z**2 <= a**2


def initial_psi_plasma(R, Z, psi_coils, R0, a, psi_edge, psi_center):
    """
    Build an initial plasma-only contribution so that the INITIAL TOTAL psi is
    roughly quadratic in radius inside a circle:
        psi_total = psi_center at axis
        psi_total = psi_edge   at r=a
    """
    r2 = (R - R0) ** 2 + Z**2
    rho2 = r2 / a**2

    psi_target = psi_center + (psi_edge - psi_center) * rho2

    psi_plasma = np.zeros_like(R, dtype=float)
    inside = rho2 <= 1.0
    psi_plasma[inside] = psi_target[inside] - psi_coils[inside]
    return psi_plasma, inside


def prof(a0, psi, psimask, nu=2):
    """
    Profile localized strictly to psimask, normalized using psi values
    inside that mask only.
    """
    profile = np.zeros_like(psi, dtype=float)

    psi_in = psi[psimask]
    if psi_in.size == 0:
        return profile

    psi_min = psi_in.min()
    psi_max = psi_in.max()
    denom = psi_max - psi_min

    if abs(denom) < 1e-14:
        return profile

    s = np.zeros_like(psi, dtype=float)
    s[psimask] = (psi[psimask] - psi_min) / denom
    s_clip = np.clip(s, 0.0, 1.0)

    profile[psimask] = a0 * (1.0 - s_clip[psimask] ** 2) ** nu
    return profile


def profprime(a0, psi, psimask, nu=2):
    """
    d(profile)/dpsi localized strictly to psimask.
    """
    profileprime = np.zeros_like(psi, dtype=float)

    psi_in = psi[psimask]
    if psi_in.size == 0:
        return profileprime

    psi_min = psi_in.min()
    psi_max = psi_in.max()
    denom = psi_max - psi_min

    if abs(denom) < 1e-14:
        return profileprime

    s = np.zeros_like(psi, dtype=float)
    s[psimask] = (psi[psimask] - psi_min) / denom

    active = psimask & (s > 0.0) & (s < 1.0)
    profileprime[active] = (
        -2.0 * nu * a0 * s[active] * (1.0 - s[active] ** 2) ** (nu - 1) / denom
    )
    return profileprime


def get_src(p0, F0, Rp, psi, psimask, nu=2, src_scale=1.0):
    """
    Grad-Shafranov source:
        src = R p'(psi) + F F'(psi) / R
    localized to psimask and optionally amplified by src_scale.
    """
    pprime = profprime(p0, psi, psimask, nu=nu)
    F = prof(F0, psi, psimask, nu=nu)
    Fprime = profprime(F0, psi, psimask, nu=nu)

    src = Rp * pprime + F * Fprime / Rp
    src = src_scale * src
    src = np.where(psimask, src, 0.0)
    return src


# ============================================================
# LCFS / contour helpers
# ============================================================
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_closed_vertices(vertices, tol=1e-6):
    if len(vertices) < 3:
        return False
    return np.linalg.norm(vertices[0] - vertices[-1]) < tol


def touches_boundary(vertices, Rmin, Rmax, Zmin, Zmax, tol=1e-8):
    Rv = vertices[:, 0]
    Zv = vertices[:, 1]
    return (
        np.any(np.isclose(Rv, Rmin, atol=tol))
        or np.any(np.isclose(Rv, Rmax, atol=tol))
        or np.any(np.isclose(Zv, Zmin, atol=tol))
        or np.any(np.isclose(Zv, Zmax, atol=tol))
    )


def point_in_polygon(point, vertices):
    return Path(vertices).contains_point(point)


def find_magnetic_axis(RR, ZZ, psi, axis="min"):
    if axis == "min":
        idx = np.unravel_index(np.argmin(psi), psi.shape)
    elif axis == "max":
        idx = np.unravel_index(np.argmax(psi), psi.shape)
    else:
        raise ValueError("axis must be 'min' or 'max'")

    return {"R": RR[idx], "Z": ZZ[idx], "psi": psi[idx], "index": idx}


def get_closed_contours_at_level(RR, ZZ, psi, level, tol=1e-6):
    Rmin, Rmax = RR.min(), RR.max()
    Zmin, Zmax = ZZ.min(), ZZ.max()

    fig, ax = plt.subplots()
    cs = ax.contour(RR, ZZ, psi, levels=[level])

    contours = []
    if len(cs.allsegs) > 0:
        for vertices in cs.allsegs[0]:
            if len(vertices) < 3:
                continue
            if not is_closed_vertices(vertices, tol=tol):
                continue
            if touches_boundary(vertices, Rmin, Rmax, Zmin, Zmax, tol=tol):
                continue
            contours.append(
                {"level": level, "vertices": vertices, "area": polygon_area(vertices)}
            )

    plt.close(fig)
    return contours


def find_lcfs_from_axis(RR, ZZ, psi, axis="min", nlevels=200, tol=1e-6):
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
        enclosing = [c for c in contours if point_in_polygon((Raxis, Zaxis), c["vertices"])]
        if len(enclosing) == 0:
            continue
        last_good = max(enclosing, key=lambda c: c["area"])

    if last_good is None:
        return None

    last_good["axis"] = axis_info
    return last_good


def plot_psi_axis_lcfs(RR, ZZ, psi, lcfs=None, ncontours=40):
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


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    # ---------------- user parameters ----------------
    figsize = (6, 5)

    NR = 100
    NZ = 100
    R_vals = np.linspace(0.5, 1.5, NR)
    Z_vals = np.linspace(-0.5, 0.5, NZ)

    blockR = 25
    blockZ = 25

    R0 = 1.0
    a = 0.22

    # try increasing these if the plasma contribution is still too weak
    p0 = 1.0
    F0 = 0.2
    nu = 2

    # this is the main strength knob
    src_scale = 30.0

    alpha = 0.2
    Niter = 60

    # ring of current filaments in the RZ plane
    Ncoils = 20
    coil_radius = 0.32
    Icoil = 5.0e5

    # ---------------- grid ----------------
    RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing="ij")
    Rij = RR

    # ---------------- coils ----------------
    theta = np.linspace(0.0, 2.0 * np.pi, Ncoils, endpoint=False)
    Rs = R0 + coil_radius * np.cos(theta)
    Zs = coil_radius * np.sin(theta)
    coils = [(Rc, Zc, Icoil) for Rc, Zc in zip(Rs, Zs)]

    psi_coils = multi_coil_fields(RR, ZZ, coils)

    # ---------------- initial target ----------------
    psi_circle = psi_on_circle(R0, a, coils, N=200)
    psi_edge = float(np.mean(psi_circle))
    psi_center = float(psi_edge - 0.5 * abs(psi_edge) - 0.2)

    psi_plasma0, psimask = initial_psi_plasma(
        RR,
        ZZ,
        psi_coils,
        R0,
        a,
        psi_edge,
        psi_center,
    )

    psi_init = psi_plasma0 + psi_coils
    psi = psi_init.copy()

    Gblocks = precompile_blocks(
        R_vals,
        Z_vals,
        blockR=blockR,
        blockZ=blockZ,
    )

    print("initial setup:")
    print(f"  psi_coils min/max = {psi_coils.min():.6e}, {psi_coils.max():.6e}")
    print(f"  psi_edge          = {psi_edge:.6e}")
    print(f"  psi_center        = {psi_center:.6e}")
    print(f"  initial mask pts  = {np.sum(psimask)} / {psimask.size}")
    print(f"  src_scale         = {src_scale:.6e}")

    # ---------------- iterate ----------------
    for j in range(Niter):
        psi_old = psi.copy()

        src = get_src(
            p0,
            F0,
            Rij,
            psi_old,
            psimask,
            nu=nu,
            src_scale=src_scale,
        )

        psi_plasma = apply_Gfunc_blocks(
            src,
            R_vals,
            Z_vals,
            Gblocks,
            blockR=blockR,
            blockZ=blockZ,
        )

        psi_fixedpoint = psi_plasma + psi_coils

        delta = psi_fixedpoint - psi_old
        residual = psi_old - psi_fixedpoint

        delta_inf = np.max(np.abs(delta))
        delta_l2 = np.sqrt(np.mean(delta**2))

        res_inf = np.max(np.abs(residual))
        res_l2 = np.sqrt(np.mean(residual**2))

        src_inf = np.max(np.abs(src))
        src_l2 = np.sqrt(np.mean(src**2))

        plasma_inf = np.max(np.abs(psi_plasma))
        plasma_l2 = np.sqrt(np.mean(psi_plasma**2))

        coils_inf = np.max(np.abs(psi_coils))
        coils_l2 = np.sqrt(np.mean(psi_coils**2))

        psi_inf = np.max(np.abs(psi_old))
        psi_l2 = np.sqrt(np.mean(psi_old**2))

        ratio = plasma_inf / coils_inf if coils_inf != 0 else np.nan

        print(
            f"iter {j:03d} | "
            f"inside={np.sum(psimask):5d} | "
            f"max|delta|={delta_inf:.3e} | "
            f"rms(delta)={delta_l2:.3e} | "
            f"max|res|={res_inf:.3e} | "
            f"rms(res)={res_l2:.3e} | "
            f"max|src|={src_inf:.3e} | "
            f"rms(src)={src_l2:.3e} | "
            f"max|psi_plasma|={plasma_inf:.3e} | "
            f"rms(psi_plasma)={plasma_l2:.3e} | "
            f"max|psi_coils|={coils_inf:.3e} | "
            f"rms(psi_coils)={coils_l2:.3e} | "
            f"max|psi|={psi_inf:.3e} | "
            f"rms(psi)={psi_l2:.3e} | "
            f"plasma/coils={ratio:.3e}"
        )

        psi = (1.0 - alpha) * psi_old + alpha * psi_fixedpoint

        if j % 10 == 0 or j == Niter - 1:
            fig, ax = plt.subplots(figsize=figsize)
            cf = ax.contourf(RR, ZZ, psi, levels=30)
            fig.colorbar(cf, ax=ax)
            ax.contour(RR, ZZ, psi, levels=30, colors="k", linewidths=0.35)
            ax.plot(RR[psimask], ZZ[psimask], "w.", ms=0.4, alpha=0.15)
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            ax.set_title(f"psi at iter {j}")
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(figsize=figsize)
            cf = ax.contourf(RR, ZZ, src, levels=30)
            fig.colorbar(cf, ax=ax)
            ax.plot(RR[psimask], ZZ[psimask], "w.", ms=0.4, alpha=0.15)
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            ax.set_title(f"src at iter {j}")
            plt.tight_layout()
            plt.show()

    # ---------------- final plots ----------------
    plt.figure(figsize=figsize)
    cf = plt.contourf(RR, ZZ, psi, levels=40)
    plt.colorbar(cf)
    plt.contour(RR, ZZ, psi, levels=40, colors="k", linewidths=0.4)
    plt.plot(RR[psimask], ZZ[psimask], "w.", ms=0.4, alpha=0.15)
    plt.xlabel("R")
    plt.ylabel("Z")
    plt.title("Final psi")
    plt.tight_layout()
    plt.show()

    lcfs = find_lcfs_from_axis(RR, ZZ, psi, axis="min", nlevels=200)
    plot_psi_axis_lcfs(RR, ZZ, psi, lcfs=lcfs, ncontours=40)
