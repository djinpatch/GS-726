import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
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

    return np.stack(Gblocks, axis=0)


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
# fixed circular shell helpers
# ============================================================
def circle_mask(R, Z, R0, a, Z0=0.0):
    return (R - R0) ** 2 + (Z - Z0) ** 2 <= a**2


def initial_psi_plasma(R, Z, R0, a, psi_edge, psi_center, Z0=0.0):
    """
    Build an initial plasma-only contribution so that the INITIAL TOTAL psi is
    roughly quadratic in radius inside a circle:
        psi_total = psi_center at axis
        psi_total = psi_edge   at r=a

    Since there are no externally imposed coil contributions here, this returns
    the initial total/free flux directly inside the plasma region.
    """
    r2 = (R - R0) ** 2 + (Z - Z0) ** 2
    rho2 = r2 / a**2

    psi_target = psi_center + (psi_edge - psi_center) * rho2

    psi_plasma = np.zeros_like(R, dtype=float)
    inside = rho2 <= 1.0
    psi_plasma[inside] = psi_target[inside]
    return psi_plasma, inside


def psi_on_circle(RR, ZZ, psi, a=0.5, R0=1.5, Z0=0.0, N=200):
    """
    Returns:
    --------
    psi_circle : (N,) values sampled on circle
    psimask    : (NR, NZ) boolean mask for points inside circle
    """
    R_vals = RR[:, 0]
    Z_vals = ZZ[0, :]

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    Rs = R0 + a * np.cos(theta)
    Zs = Z0 + a * np.sin(theta)

    psi_circle = np.empty(N, dtype=float)

    for k in range(N):
        Rq = Rs[k]
        Zq = Zs[k]

        i = np.searchsorted(R_vals, Rq) - 1
        j = np.searchsorted(Z_vals, Zq) - 1

        i = np.clip(i, 0, len(R_vals) - 2)
        j = np.clip(j, 0, len(Z_vals) - 2)

        R1, R2 = R_vals[i], R_vals[i + 1]
        Z1, Z2 = Z_vals[j], Z_vals[j + 1]

        t = (Rq - R1) / (R2 - R1)
        u = (Zq - Z1) / (Z2 - Z1)

        psi11 = psi[i,     j]
        psi21 = psi[i + 1, j]
        psi12 = psi[i,     j + 1]
        psi22 = psi[i + 1, j + 1]

        psi_circle[k] = (
            (1 - t) * (1 - u) * psi11
            + t * (1 - u) * psi21
            + (1 - t) * u * psi12
            + t * u * psi22
        )

    psimask = circle_mask(RR, ZZ, R0, a, Z0=Z0)
    return psi_circle, psimask

# ============================================================
# wall-current fixed-boundary helpers
# ============================================================
def circle_points(R0, a, Z0=0.0, N=64, theta_shift=0.0):
    """
    Return N points on a circle.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False) + theta_shift
    Rpts = R0 + a * np.cos(theta)
    Zpts = Z0 + a * np.sin(theta)
    return Rpts, Zpts


def sample_psi_at_points(RR, ZZ, psi, Rs, Zs):
    """
    Bilinearly interpolate psi(R,Z) at arbitrary points (Rs[k], Zs[k]).

    Parameters
    ----------
    RR, ZZ : meshgrid with indexing='ij'
    psi    : shape (NR, NZ)
    Rs, Zs : shape (N,)

    Returns
    -------
    psi_samples : shape (N,)
    """
    R_vals = RR[:, 0]
    Z_vals = ZZ[0, :]

    psi_samples = np.empty(len(Rs), dtype=float)

    for k in range(len(Rs)):
        Rq = Rs[k]
        Zq = Zs[k]

        i = np.searchsorted(R_vals, Rq) - 1
        j = np.searchsorted(Z_vals, Zq) - 1

        i = np.clip(i, 0, len(R_vals) - 2)
        j = np.clip(j, 0, len(Z_vals) - 2)

        R1, R2 = R_vals[i], R_vals[i + 1]
        Z1, Z2 = Z_vals[j], Z_vals[j + 1]

        t = (Rq - R1) / (R2 - R1)
        u = (Zq - Z1) / (Z2 - Z1)

        psi11 = psi[i,     j]
        psi21 = psi[i + 1, j]
        psi12 = psi[i,     j + 1]
        psi22 = psi[i + 1, j + 1]

        psi_samples[k] = (
            (1 - t) * (1 - u) * psi11
            + t * (1 - u) * psi21
            + (1 - t) * u * psi12
            + t * u * psi22
        )

    return psi_samples


def precompute_wall_system(RR, ZZ, R0, a, Z0=0.0, Nbc=64, Nwall=64, delta_wall=0.02):
    """
    Precompute the linear wall-current response system.

    Boundary condition is enforced on the physical shell of radius a.
    Wall-current basis filaments are placed on a nearby circle of radius a+delta_wall.

    Returns a dict containing:
      Rb, Zb    : collocation points on the shell
      Rw, Zw    : wall-current basis filament locations
      A         : shell response matrix, shape (Nbc, Nwall)
      Wgrid     : flux response on full grid from each wall filament, shape (Nwall, NR, NZ)
    """
    # collocation points ON the shell
    Rb, Zb = circle_points(R0, a, Z0=Z0, N=Nbc, theta_shift=0.0)

    # fictitious wall-current basis just outside the shell
    Rw, Zw = circle_points(
        R0, a + delta_wall, Z0=Z0, N=Nwall, theta_shift=np.pi / max(Nwall, 1)
    )

    # A[m,n] = mu0 * G(boundary point m ; wall filament n)
    A = mu0 * G(
        Rb[:, None],
        Zb[:, None],
        Rw[None, :],
        Zw[None, :],
    )  # shape (Nbc, Nwall)

    # Wgrid[n,:,:] = mu0 * G(grid ; wall filament n)
    Wgrid = mu0 * G(
        RR[None, :, :],
        ZZ[None, :, :],
        Rw[:, None, None],
        Zw[:, None, None],
    )  # shape (Nwall, NR, NZ)

    return {
        "Rb": Rb,
        "Zb": Zb,
        "Rw": Rw,
        "Zw": Zw,
        "A": A,
        "Wgrid": Wgrid,
    }


def solve_wall_currents(psi_free_bdry, wall_system, reg=1e-10):
    """
    Solve for wall-current coefficients Iw such that
        A @ Iw ~= -psi_free_bdry

    using Tikhonov-regularized least squares:
        (A^T A + reg I) Iw = -A^T psi_free_bdry
    """
    A = wall_system["A"]
    ATA = A.T @ A + reg * np.eye(A.shape[1])
    rhs = -A.T @ psi_free_bdry
    Iw = np.linalg.solve(ATA, rhs)
    return Iw


def wall_flux_from_currents(Iw, wall_system):
    """
    Given wall-current coefficients Iw, return psi_wall(R,Z)
    on the full grid.
    """
    Wgrid = wall_system["Wgrid"]  # shape (Nwall, NR, NZ)
    psi_wall = np.tensordot(Iw, Wgrid, axes=(0, 0))
    return psi_wall


def apply_wall_current_correction(RR, ZZ, psi_free, wall_system, reg=1e-10):
    """
    Given free flux psi_free = psi_plasma, solve for wall currents
    and return corrected total flux:

        psi_total = psi_free + psi_wall

    Also returns wall-current coefficients and shell diagnostics.
    """
    Rb = wall_system["Rb"]
    Zb = wall_system["Zb"]

    psi_free_bdry = sample_psi_at_points(RR, ZZ, psi_free, Rb, Zb)
    Iw = solve_wall_currents(psi_free_bdry, wall_system, reg=reg)
    psi_wall = wall_flux_from_currents(Iw, wall_system)
    psi_total = psi_free + psi_wall

    psi_total_bdry = sample_psi_at_points(RR, ZZ, psi_total, Rb, Zb)

    return {
        "psi_total": psi_total,
        "psi_wall": psi_wall,
        "Iw": Iw,
        "psi_free_bdry": psi_free_bdry,
        "psi_total_bdry": psi_total_bdry,
    }

# ============================================================
# magnetic axis
# ============================================================
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

# ============================================================
# plasma profiles and source
# ============================================================
def prof(a0, psi, psimask, psi_axis, psi_edge, nu=2):
    """
    Profile localized strictly to psimask, using
        s = (psi - psi_axis) / (psi_edge - psi_axis)
    so that:
        s = 0 at magnetic axis
        s = 1 at shell boundary
    """
    profile = np.zeros_like(psi, dtype=float)

    denom_val = psi_edge - psi_axis
    if abs(denom_val) < 1e-14:
        return profile

    s = np.zeros_like(psi, dtype=float)
    s[psimask] = (psi[psimask] - psi_axis) / denom_val
    s_clip = np.clip(s, 0.0, 1.0)

    profile[psimask] = a0 * (1.0 - s_clip[psimask] ** 2) ** nu
    return profile


def profprime(a0, psi, psimask, psi_axis, psi_edge, nu=2):
    """
    d(profile)/dpsi localized strictly to psimask.
    """
    profileprime = np.zeros_like(psi, dtype=float)

    denom_val = psi_edge - psi_axis
    if abs(denom_val) < 1e-14:
        return profileprime

    s = np.zeros_like(psi, dtype=float)
    s[psimask] = (psi[psimask] - psi_axis) / denom_val

    active = psimask & (s > 0.0) & (s < 1.0)
    profileprime[active] = (
        -2.0 * nu * a0 * s[active] * (1.0 - s[active] ** 2) ** (nu - 1) / denom_val
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
# equilibrium diagnostics on RZ grid
# ============================================================
def compute_axisymmetric_fields_and_currents(
    RR, ZZ, psi, psimask, psi_axis, psi_edge, p0, F0, nu=2
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

    p = prof(p0, psi, psimask, psi_axis, psi_edge, nu=nu)
    F = prof(F0, psi, psimask, psi_axis, psi_edge, nu=nu)

    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)

    d2psi_dR2 = np.gradient(dpsi_dR, dR, axis=0, edge_order=2)
    d2psi_dZ2 = np.gradient(dpsi_dZ, dZ, axis=1, edge_order=2)

    delta_star_psi = d2psi_dR2 - (1.0 / RR) * dpsi_dR + d2psi_dZ2

    B_R = -(1.0 / RR) * dpsi_dZ
    B_Z =  (1.0 / RR) * dpsi_dR
    B_phi = F / RR

    dF_dR = np.gradient(F, dR, axis=0, edge_order=2)
    dF_dZ = np.gradient(F, dZ, axis=1, edge_order=2)

    J_R   = -(1.0 / (mu0 * RR)) * dF_dZ
    J_Z   =  (1.0 / (mu0 * RR)) * dF_dR
    J_phi = -(1.0 / (mu0 * RR)) * delta_star_psi

    dp_dR = np.gradient(p, dR, axis=0, edge_order=2)
    dp_dZ = np.gradient(p, dZ, axis=1, edge_order=2)

    JxB_R   = J_phi * B_Z - J_Z * B_phi
    JxB_Z   = J_R * B_phi - J_phi * B_R
    JxB_phi = J_Z * B_R - J_R * B_Z

    FB_R   = JxB_R - dp_dR
    FB_Z   = JxB_Z - dp_dZ
    FB_phi = JxB_phi
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

# ============================================================
# plotting
# ============================================================
def plot_state(
    RR,
    ZZ,
    psi,
    psimask,
    axis_info,
    title,
    coils=None,
    mask=None,
    ax=None,
    figsize=(6, 5),
    show_colorbar=True,
):
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

    ax.contour(
        RR,
        ZZ,
        psimask.astype(float),
        levels=[0.5],
        colors="w",
        linewidths=1.8,
    )

    ax.plot(axis_info["R"], axis_info["Z"], "bo", ms=6)

    if coils is not None:
        for c in coils:
            ax.plot(c[0], c[1], "r.", ms=10)

    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)

    legend_handles = [
        Line2D([0], [0], color="k", lw=1.0, label=r"$\psi$ contours"),
        Line2D([0], [0], color="w", lw=1.8, label="Fixed shell boundary"),
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

    if mask is not None:
        ax.contour(
            RR,
            ZZ,
            mask.astype(float),
            levels=[0.5],
            colors="w",
            linewidths=1.8,
        )

    ax.plot(axis_info["R"], axis_info["Z"], "bo", ms=6)

    if coils is not None:
        for c in coils:
            ax.plot(c[0], c[1], "r.", ms=10)

    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)

    legend_handles = [
        Line2D([0], [0], color="w", lw=1.8, label="Fixed shell boundary"),
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


def plot_all_three(RR, ZZ, psi, psimask, psi_axis, psi_edge, axis_info, coils, p0, F0, nu, j):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    diag = compute_axisymmetric_fields_and_currents(
        RR, ZZ, psi, psimask, psi_axis, psi_edge, p0, F0, nu=nu
    )

    plot_state(
        RR,
        ZZ,
        psi,
        psimask,
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
        axis_info=axis_info,
        mask=psimask,
        coils=coils,
        ax=axes[2],
        logscale=True,
        vmin=1e-8,
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# START: MST-style fixed-boundary run with wall-current correction
# No externally imposed coil flux: psi_free = psi_plasma
# ============================================================
figsize = (6, 5)

Plot_Every = 20

NR = 100
NZ = 100
R_vals = np.linspace(1.0, 2.0, NR)
Z_vals = np.linspace(-0.5, 0.5, NZ)

blockR = 25
blockZ = 25

R0 = 1.5
Z0 = 0.0
a = 0.50

p0 = 5.0
F0 = 0.8
nu = 2

src_scale = 1.0
alpha = 0.1
Niter = 280
axis_type = "min"

# wall-current boundary control parameters
Nbc = 128           # number of shell collocation points
Nwall = 128         # number of fictitious wall filaments
delta_wall = 0.01  # radius offset for wall-current basis
wall_reg = 1e-14   # regularization for wall-current solve

# ---------------- grid ----------------
RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing="ij")
Rij = RR

# ---------------- fixed shell mask ----------------
psimask = circle_mask(RR, ZZ, R0, a, Z0=Z0)
psi_edge = 0.0

# ---------------- no externally imposed coil flux ----------------
coils = None

# ---------------- precompute Green blocks ----------------
Gblocks = precompile_blocks(
    R_vals,
    Z_vals,
    blockR=blockR,
    blockZ=blockZ,
)

# ---------------- precompute wall system ----------------
wall_system = precompute_wall_system(
    RR,
    ZZ,
    R0=R0,
    a=a,
    Z0=Z0,
    Nbc=Nbc,
    Nwall=Nwall,
    delta_wall=delta_wall,
)

# ---------------- initial guess ----------------
psi_center = -5.0

psi_plasma0, _ = initial_psi_plasma(
    RR,
    ZZ,
    R0,
    a,
    psi_edge,
    psi_center,
    Z0=Z0,
)

psi_free = psi_plasma0

wall_corr = apply_wall_current_correction(
    RR, ZZ, psi_free, wall_system, reg=wall_reg
)
psi = wall_corr["psi_total"]

# ---------------- initial axis ----------------
axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)
psi_axis = axis_info["psi"]

print(
    "Initial shell flux residual: "
    f"max|psi_shell_resid| = {np.max(np.abs(wall_corr['psi_total_bdry'])):.3e}"
)

plot_state(
    RR,
    ZZ,
    psi,
    psimask,
    axis_info,
    title="Initial $\\psi$ before iteration",
    coils=coils,
    figsize=figsize,
)

for j in range(Niter):
    psi_old = psi.copy()

    src = get_src(
        p0,
        F0,
        Rij,
        psi_old,
        psimask,
        psi_axis,
        psi_edge,
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

    psi_free = psi_plasma

    wall_corr = apply_wall_current_correction(
        RR, ZZ, psi_free, wall_system, reg=wall_reg
    )
    psi_fixedpoint = wall_corr["psi_total"]

    delta = psi_fixedpoint - psi_old
    delta_inf = np.max(np.abs(delta))
    plasma_inf = np.max(np.abs(psi_plasma))
    wall_inf = np.max(np.abs(wall_corr["psi_wall"]))
    shell_resid = np.max(np.abs(wall_corr["psi_total_bdry"]))

    print(
        f"iter {j:03d} | "
        f"psi_axis={psi_axis:.6e} | "
        f"psi_edge={psi_edge:.6e} | "
        f"max|delta|={delta_inf:.3e} | "
        f"max|psi_plasma|={plasma_inf:.3e} | "
        f"max|psi_wall|={wall_inf:.3e} | "
        f"max|psi_shell_resid|={shell_resid:.3e}"
    )

    # relaxed Picard update
    psi = (1.0 - alpha) * psi_old + alpha * psi_fixedpoint

    # fixed shell boundary stays the same
    psi_edge = 0.0

    # update axis only inside the fixed plasma region
    axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)
    psi_axis = axis_info["psi"]

    if j % Plot_Every == 0 and j != 0:
        plot_all_three(
            RR, ZZ, psi, psimask, psi_axis, psi_edge,
            axis_info, coils, p0, F0, nu, j
        )

# ---------------- final diagnostics ----------------
diag = compute_axisymmetric_fields_and_currents(
    RR, ZZ, psi, psimask, psi_axis, psi_edge, p0, F0, nu=nu
)

plot_state(
    RR,
    ZZ,
    psi,
    psimask,
    axis_info,
    title=r"$\psi$ Final",
    coils=coils,
)

plot_rz_scalar(
    RR,
    ZZ,
    diag["J_phi"],
    title=r"$J_\phi$ Final",
    cbar_label=r"$J_\phi$",
    axis_info=axis_info,
    mask=psimask,
    coils=coils,
)

plot_rz_scalar(
    RR,
    ZZ,
    diag["FB_mag"],
    title=r"$|\mathbf{J}\times\mathbf{B}-\nabla p|$ Final",
    cbar_label=r"$|\mathbf{J}\times\mathbf{B}-\nabla p|$",
    axis_info=axis_info,
    mask=psimask,
    logscale=True,
    coils=coils,
)