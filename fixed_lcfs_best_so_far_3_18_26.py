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


def prof(a0, psi, psimask, psi_axis, psi_edge, nu=2):
    """
    Profile localized strictly to psimask, using

        s = (psi - psi_axis) / (psi_edge - psi_axis)

    so that ideally:
        s = 0 at magnetic axis
        s = 1 at LCFS
    """
    profile = np.zeros_like(psi, dtype=float)

    denom = psi_edge - psi_axis
    if abs(denom) < 1e-14:
        return profile

    s = np.zeros_like(psi, dtype=float)
    s[psimask] = (psi[psimask] - psi_axis) / denom
    s_clip = np.clip(s, 0.0, 1.0)

    profile[psimask] = a0 * (1.0 - s_clip[psimask] ** 2) ** nu
    return profile


def profprime(a0, psi, psimask, psi_axis, psi_edge, nu=2):
    """
    d(profile)/dpsi localized strictly to psimask, using

        s = (psi - psi_axis) / (psi_edge - psi_axis)
    """
    profileprime = np.zeros_like(psi, dtype=float)

    denom = psi_edge - psi_axis
    if abs(denom) < 1e-14:
        return profileprime

    s = np.zeros_like(psi, dtype=float)
    s[psimask] = (psi[psimask] - psi_axis) / denom

    active = psimask & (s > 0.0) & (s < 1.0)
    profileprime[active] = (
        -2.0 * nu * a0 * s[active] * (1.0 - s[active] ** 2) ** (nu - 1) / denom
    )
    return profileprime


def get_src(p0, F0, Rp, psi, psimask, psi_axis, psi_edge, nu=2, src_scale=1.0):
    """
    Grad-Shafranov source:
        src = R p'(psi) + F F'(psi) / R

    using profiles normalized by psi_axis and psi_edge.
    """
    pprime = profprime(p0, psi, psimask, psi_axis, psi_edge, nu=nu)
    F = prof(F0, psi, psimask, psi_axis, psi_edge, nu=nu)
    Fprime = profprime(F0, psi, psimask, psi_axis, psi_edge, nu=nu)

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


def points_in_polygon(RR, ZZ, vertices):
    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    mask = Path(vertices).contains_points(pts)
    return mask.reshape(RR.shape)


def find_magnetic_axis(RR, ZZ, psi, axis="min", mask=None):
    if mask is None:
        mask = np.ones_like(psi, dtype=bool)

    if not np.any(mask):
        raise ValueError("mask contains no True points")

    psi_masked = np.where(mask, psi, np.nan)

    if axis == "min":
        idx_flat = np.nanargmin(psi_masked)
    elif axis == "max":
        idx_flat = np.nanargmax(psi_masked)
    else:
        raise ValueError("axis must be 'min' or 'max'")

    idx = np.unravel_index(idx_flat, psi.shape)
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


def find_enclosing_contour_at_level(RR, ZZ, psi, level, axis="min", tol=1e-6):
    axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis, mask=psimask)
    Raxis = axis_info["R"]
    Zaxis = axis_info["Z"]

    contours = get_closed_contours_at_level(RR, ZZ, psi, level, tol=tol)
    enclosing = [c for c in contours if point_in_polygon((Raxis, Zaxis), c["vertices"])]

    if len(enclosing) == 0:
        return None

    best = max(enclosing, key=lambda c: c["area"])
    best["axis"] = axis_info
    return best


def recompute_mask_from_target(
    RR,
    ZZ,
    psi,
    psi_edge_target,
    axis="min",
    fallback_mask=None,
    tol=1e-6,
):
    lcfs = find_enclosing_contour_at_level(
        RR,
        ZZ,
        psi,
        level=psi_edge_target,
        axis=axis,
        tol=tol,
    )

    if lcfs is None:
        if fallback_mask is None:
            return np.zeros_like(psi, dtype=bool), None, False
        return fallback_mask.copy(), None, False

    psimask = points_in_polygon(RR, ZZ, lcfs["vertices"])
    return psimask, lcfs, True


def find_lcfs_from_axis(RR, ZZ, psi, axis="min", nlevels=200, tol=1e-6, lcfs_prev=None):
    axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis, mask=psimask)
    Raxis = axis_info["R"]
    Zaxis = axis_info["Z"]
    psiaxis = axis_info["psi"]

    psimin = psi.min()
    psimax = psi.max()

    if axis == "min":
        levels = np.linspace(psiaxis, psimax, nlevels + 2)[1:-1]
    else:
        levels = np.linspace(psimin, psiaxis, nlevels + 2)[1:-1][::-1]

    last_good = None
    best_cost = np.inf

    for level in levels:
        contours = get_closed_contours_at_level(RR, ZZ, psi, level, tol=tol)
        enclosing = [c for c in contours if point_in_polygon((Raxis, Zaxis), c["vertices"])]
        if len(enclosing) == 0:
            continue

        if lcfs_prev is None:
            candidate = max(enclosing, key=lambda c: c["area"])
            last_good = candidate
        else:
            Rc_prev = np.mean(lcfs_prev["vertices"][:, 0])
            Zc_prev = np.mean(lcfs_prev["vertices"][:, 1])
            A_prev = lcfs_prev["area"]

            def cost(c):
                Rc = np.mean(c["vertices"][:, 0])
                Zc = np.mean(c["vertices"][:, 1])
                A = c["area"]
                return (
                    abs(Rc - Rc_prev)
                    + abs(Zc - Zc_prev)
                    + 0.1 * abs(A - A_prev)
                )

            candidate = min(enclosing, key=cost)
            cst = cost(candidate)

            if cst < best_cost:
                best_cost = cst
                last_good = candidate

    if last_good is None:
        raise RuntimeError("ERROR: No closed flux surface around magnetic axis")

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


def psi_edge_from_circle_mask(RR, ZZ, psi, R0, Z0, a_ref, frac=0.5):
    """
    On the current psi(R,Z), build a circular mask and return a psi value
    between min and max inside that circle.

    frac = 0   -> min psi in circle
    frac = 1   -> max psi in circle
    """
    mask_ref = (RR - R0) ** 2 + (ZZ - Z0) ** 2 <= a_ref**2

    psi_in = psi[mask_ref]
    if psi_in.size == 0:
        raise ValueError("reference circle contains no grid points")

    psi_min = float(np.min(psi_in))
    psi_max = float(np.max(psi_in))
    frac = float(np.clip(frac, 0.0, 1.0))

    psi_edge = psi_min + frac * (psi_max - psi_min)
    return psi_edge, mask_ref, psi_min, psi_max


# ============================================================
# equilibrium diagnostics on RZ grid
# ============================================================
def compute_axisymmetric_fields_and_currents(
    RR, ZZ, psi, psimask, p0, F0, nu=2
):
    """
    Returns p, F, B, J, and force-balance residual on the RZ grid.

    Conventions:
      B_R   = -(1/R) dpsi/dZ
      B_Z   =  (1/R) dpsi/dR
      B_phi = F(psi)/R

      J_phi = -(1/mu0 R) * Delta* psi
      J_R   = -(1/mu0 R) dF/dZ
      J_Z   =  (1/mu0 R) dF/dR
    """
    R_vals = RR[:, 0]
    Z_vals = ZZ[0, :]

    dR = R_vals[1] - R_vals[0]
    dZ = Z_vals[1] - Z_vals[0]

    # profiles
    p = prof(p0, psi, psimask,psi_axis, psi_edge_target, nu=nu)
    F = prof(F0, psi, psimask,psi_axis, psi_edge_target, nu=nu)

    # first derivatives of psi
    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)

    # second derivatives of psi
    d2psi_dR2 = np.gradient(dpsi_dR, dR, axis=0, edge_order=2)
    d2psi_dZ2 = np.gradient(dpsi_dZ, dZ, axis=1, edge_order=2)

    # Grad-Shafranov operator
    delta_star_psi = d2psi_dR2 - (1.0 / RR) * dpsi_dR + d2psi_dZ2

    # magnetic field
    B_R = -(1.0 / RR) * dpsi_dZ
    B_Z =  (1.0 / RR) * dpsi_dR
    B_phi = F / RR

    # derivatives of F
    dF_dR = np.gradient(F, dR, axis=0, edge_order=2)
    dF_dZ = np.gradient(F, dZ, axis=1, edge_order=2)

    # current density
    J_R   = -(1.0 / (mu0 * RR)) * dF_dZ
    J_Z   =  (1.0 / (mu0 * RR)) * dF_dR
    J_phi = -(1.0 / (mu0 * RR)) * delta_star_psi

    # pressure gradient
    dp_dR = np.gradient(p, dR, axis=0, edge_order=2)
    dp_dZ = np.gradient(p, dZ, axis=1, edge_order=2)

    # J x B
    # cylindrical components:
    # (J x B)_R = J_phi B_Z - J_Z B_phi
    # (J x B)_Z = J_R B_phi - J_phi B_R
    # (J x B)_phi = J_Z B_R - J_R B_Z
    JxB_R   = J_phi * B_Z - J_Z * B_phi
    JxB_Z   = J_R * B_phi - J_phi * B_R
    JxB_phi = J_Z * B_R - J_R * B_Z

    # force-balance residual
    FB_R   = JxB_R - dp_dR
    FB_Z   = JxB_Z - dp_dZ
    FB_phi = JxB_phi   # grad p has no phi component in axisymmetry

    FB_mag = np.sqrt(FB_R**2 + FB_Z**2 + FB_phi**2)

    return {
        "p": p,
        "F": F,
        "B_R": B_R,
        "B_Z": B_Z,
        "B_phi": B_phi,
        "J_R": J_R,
        "J_Z": J_Z,
        "J_phi": J_phi,
        "dp_dR": dp_dR,
        "dp_dZ": dp_dZ,
        "JxB_R": JxB_R,
        "JxB_Z": JxB_Z,
        "JxB_phi": JxB_phi,
        "FB_R": FB_R,
        "FB_Z": FB_Z,
        "FB_phi": FB_phi,
        "FB_mag": FB_mag,
    }


from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt


def plot_state(
    RR,
    ZZ,
    psi,
    psimask,
    lcfs_target,
    axis_info,
    title,
    coils=None,
    mask=None,
    ax=None,
    figsize=(6, 5),
    show_colorbar=True,
):
    """
    Pure plotting function.
    Does NOT recompute lcfs or axis_info.
    """

    if lcfs_target is None:
        raise ValueError("lcfs_target must be provided")
    if axis_info is None:
        raise ValueError("axis_info must be provided")

    if mask is not None:
        psi_plot = np.ma.masked_where(~mask, psi)
    else:
        psi_plot = psi

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    cf = ax.contourf(RR, ZZ, psi_plot, levels=30)

    if show_colorbar:
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"$\psi(R,Z)$")

    ax.contour(RR, ZZ, psi_plot, levels=30, colors="k", linewidths=0.35)

    verts = lcfs_target["vertices"]
    ax.plot(verts[:, 0], verts[:, 1], "w-", lw=1.8)

    ax.plot(axis_info["R"], axis_info["Z"], "bo", ms=6)

    if coils is not None:
        for c in coils:
            ax.plot(c[0], c[1], "r.", ms=10)

    if psimask is not None:
        ax.plot(RR[psimask], ZZ[psimask], "w.", ms=0.3, alpha=0.1)

    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)

    legend_handles = [
        Line2D([0], [0], color="k", lw=1.0, label=r"$\psi$ contours"),
        Line2D([0], [0], color="w", lw=1.8, label="LCFS"),
        Line2D([0], [0], marker="o", color="b", lw=0, markersize=6, label="Magnetic axis"),
    ]


    if coils is not None:
        legend_handles.append(
            Line2D([0], [0], marker=".", color="r", lw=0, markersize=10, label="Coils")
        )

    ax.legend(handles=legend_handles, loc="best")

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plot_rz_scalar(
    RR,
    ZZ,
    field,
    title,
    cbar_label,
    lcfs=None,
    axis_info=None,
    mask=None,
    coils=None,
    ax=None,
    logscale=False,
    vmin=None,
    vmax=None,
    figsize=(6, 5),
    show_colorbar=True,
):
    """
    Pure plotting function.
    Does NOT recompute lcfs or axis_info.
    """

    if lcfs is None:
        raise ValueError("lcfs must be provided")
    if axis_info is None:
        raise ValueError("axis_info must be provided")

    if mask is not None:
        field_plot = np.ma.masked_where(~mask, field)
    else:
        field_plot = field

    if logscale:
        field_plot = np.ma.masked_where(field_plot <= 0, field_plot)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    if logscale:
        if vmin is None:
            vmin = np.min(field_plot.compressed())
        if vmax is None:
            vmax = np.max(field_plot.compressed())
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    cf = ax.contourf(RR, ZZ, field_plot, levels=40, norm=norm)

    if show_colorbar:
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(cbar_label)

    ax.contour(RR, ZZ, field_plot, levels=20, colors="k", linewidths=0.3)

    verts = lcfs["vertices"]
    ax.plot(verts[:, 0], verts[:, 1], "w-", lw=1.8)

    ax.plot(axis_info["R"], axis_info["Z"], "bo", ms=6)

    if coils is not None:
        for c in coils:
            ax.plot(c[0], c[1], "r.", ms=10)

    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)

    legend_handles = [
        Line2D([0], [0], color="w", lw=1.8, label="LCFS"),
        Line2D([0], [0], marker="o", color="b", lw=0, markersize=6, label="Magnetic axis"),
    ]

    if coils is not None:
        legend_handles.append(
            Line2D([0], [0], marker=".", color="r", lw=0, markersize=10, label="Coils")
        )

    ax.legend(handles=legend_handles, loc="best")

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax




def plot_all_three():

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    diag = compute_axisymmetric_fields_and_currents(
        RR,
        ZZ,
        psi,
        psimask,
        p0,
        F0,
        nu=nu,
    )
        
    
    plot_state(
        RR,
        ZZ,
        psi,
        psimask,
        lcfs_target,
        axis_info,
        title=f"$\\psi$ at iteration {j}",
        coils=coils,
        ax=axes[0],
    )
    
    plot_rz_scalar(
        RR,
        ZZ,
        diag["J_phi"],
        title=rf"$J_\phi$ at iteration {j}",
        cbar_label=r"$J_\phi$",
        lcfs=lcfs_target,
        axis_info=axis_info,
        mask=psimask,
        coils=coils,
        ax=axes[1],
    )
    
    plot_rz_scalar(
        RR,
        ZZ,
        diag["FB_mag"],
        title=rf"$|\mathbf{{J}}\times\mathbf{{B}}-\nabla p|$ at iteration {j}",
        cbar_label=r"$|\mathbf{J}\times\mathbf{B}-\nabla p|$",
        lcfs=lcfs_target,
        axis_info=axis_info,
        mask=psimask,
        coils=coils,
        ax=axes[2],
        logscale=True,
        vmin=1e-8,
    )
    
    plt.tight_layout()
    plt.show()


########## START ###########


figsize = (6, 5)

### Note MST paramaters 
#  VV a = .52m 
#  VV R = 1.5m 
#  p0~480 Pa 
#  <F> = 1.8mt 
#  plasma current = .5MA ~~ conducting shell torodial current 

NR = 100
NZ = 100
R_vals = np.linspace(1, 2, NR)
Z_vals = np.linspace(-0.5, 0.5, NZ)

blockR = 25
blockZ = 25

R0 = 1.5
Z0 = 0.0



p0 =  10 # 300.0 / 100
F0 = .18 # .18
nu = 2

src_scale = 1 ### REMOVE when iteration works, non physical, numerical only 

alpha = 0.075
Niter = 400

axis_type = "min"

# ring of current filaments in the RZ plane
Ncoils = 40
coil_radius = 0.52
I_total = .5e6 
Icoil = I_total/Ncoils

# ---------------- grid ----------------
RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing="ij")
Rij = RR

# ---------------- coils ----------------
theta = np.linspace(0.0, 2.0 * np.pi, Ncoils, endpoint=False)
Rs = R0 + coil_radius * np.cos(theta)
Zs = Z0 + coil_radius * np.sin(theta)
coils = [(Rc, Zc, Icoil) for Rc, Zc in zip(Rs, Zs)]

psi_coils = multi_coil_fields(RR, ZZ, coils)


# ---------------- initial guess ----------------
psi_center = 0.001
psi_edge_init = 0.22
a = 0.5
psi_plasma0, psimask = initial_psi_plasma(
    RR,
    ZZ,
    psi_coils,
    R0,
    a,
    psi_edge_init,
    psi_center,
)

psi = psi_plasma0 + psi_coils

Gblocks = precompile_blocks(
    R_vals,
    Z_vals,
    blockR=blockR,
    blockZ=blockZ,
)

lcfs_target = find_lcfs_from_axis(
    RR,
    ZZ,
    psi,
    axis=axis_type,
    nlevels=200,
    tol=1e-6,
)



psimask = points_in_polygon(RR, ZZ, lcfs_target["vertices"])
psi_edge_target = lcfs_target["level"]
axis_info_init = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)

plot_state(
    RR,
    ZZ,
    psi,
    psimask,
    lcfs_target,
    axis_info_init,
    title="Initial $\\psi$ before iteration",
    figsize=figsize,
)


FREE = False

for j in range(Niter):
    psi_old = psi.copy()

    axis_info = find_magnetic_axis(RR, ZZ, psi_old, axis=axis_type, mask=psimask)
    psi_axis = axis_info["psi"]

    src = get_src(
        p0,
        F0,
        Rij,
        psi_old,
        psimask,
        psi_axis,
        psi_edge_target,
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
    delta_inf = np.max(np.abs(delta))
    plasma_inf = np.max(np.abs(psi_plasma))
    coils_inf = np.max(np.abs(psi_coils))

    print(
        f"iter {j:03d} | "
        f"psi_axis={psi_axis:.6e} | "
        f"psi_edge_target={psi_edge_target:.6e} | "
        f"max|delta|={delta_inf:.3e} | "
        f"max|psi_plasma|={plasma_inf:.3e} | "
        f"max|psi_coils|={coils_inf:.3e}"
    )

    psi = (1.0 - alpha) * psi_old + alpha * psi_fixedpoint

    if j % 50 == 0 and j != 0:
        plot_all_three()
        

    if FREE:
        lcfs_prev = lcfs_target
        lcfs_target = find_lcfs_from_axis(
            RR,
            ZZ,
            psi,
            axis=axis_type,
            nlevels=200,
            tol=1e-6,
            lcfs_prev=lcfs_prev,
        )
        psimask = points_in_polygon(RR, ZZ, lcfs_target["vertices"])
        psi_edge_target = lcfs_target["level"]



psimask_final = points_in_polygon(RR, ZZ, lcfs_target["vertices"])
axis_info_final = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)

diag = compute_axisymmetric_fields_and_currents(
    RR,
    ZZ,
    psi,
    psimask,
    p0,
    F0,
    nu=nu,
)


plot_state(
    RR,
    ZZ,
    psi,
    psimask,
    lcfs_target,
    axis_info_final,
    title=f"$\\psi$ Final",
    coils=coils,
)


plot_rz_scalar(
    RR,
    ZZ,
    diag["J_phi"],
    title=rf"$J_\phi$ Final",
    cbar_label=r"$J_\phi$",
    lcfs=lcfs_target,
    axis_info=axis_info_final,
    mask=psimask,
    coils = coils,
)

plot_rz_scalar(
    RR,
    ZZ,
    diag["FB_mag"],
    title=rf"$|\mathbf{{J}}\times\mathbf{{B}}-\nabla p|$ Final",
    cbar_label=r"$|\mathbf{J}\times\mathbf{B}-\nabla p|$",
    lcfs=lcfs_target,
    axis_info=axis_info_final,
    mask=psimask,
    logscale=True,
    coils = coils,
)

