import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe, ellipkm1

# ============================================================
# constants
# ============================================================
pi  = np.pi
mu0 = 4.0e-7 * pi
const = 1.0 / (2.0 * pi)   # correct prefactor for toroidal Green's function

# ============================================================
# Green's function  (eq. 13.47 from textbook screenshot)
# G(R,Z; R',Z') = (1/pi) * sqrt(RR') * (2-k^2)*K(k) - 2*E(k)) / k
# Rewritten in terms of denom = (R+R')^2 + (Z-Z')^2 and m = k^2:
#   sqrt(RR') = k*sqrt(denom)/2,  so  G = sqrt(denom)/(2*pi) * [(2-m)*K - 2*E]
# ============================================================
def denom(R, Z, Rp, Zp):
    return (R + Rp)**2 + (Z - Zp)**2


def ksquare(R, Z, Rp, Zp):
    return 4.0 * R * Rp / denom(R, Z, Rp, Zp)


def G(R, Z, Rp, Zp, eps=1e-14, switch=1e-8):
    m = ksquare(R, Z, Rp, Zp)
    m = np.clip(m, 0.0, 1.0 - eps)

    K = np.empty_like(m)
    near1 = (1.0 - m) < switch
    K[near1]  = ellipkm1(1.0 - m[near1])
    K[~near1] = ellipk(m[~near1])
    E = ellipe(m)

    return const * np.sqrt(denom(R, Z, Rp, Zp)) * ((2.0 - m) * K - 2.0 * E)


# ============================================================
# fixed circular shell helpers
# ============================================================
def circle_mask(R, Z, R0, a, Z0=0.0):
    return (R - R0)**2 + (Z - Z0)**2 <= a**2


def initial_psi_plasma(R, Z, R0, a, psi_edge, psi_center, Z0=0.0):
    r2   = (R - R0)**2 + (Z - Z0)**2
    rho2 = r2 / a**2
    psi_target = psi_center + (psi_edge - psi_center) * rho2
    psi_plasma = np.zeros_like(R, dtype=float)
    inside = rho2 <= 1.0
    psi_plasma[inside] = psi_target[inside]
    return psi_plasma, inside


def circle_points(R0, a, Z0=0.0, N=64, theta_shift=0.0):
    theta = np.linspace(0.0, 2.0 * pi, N, endpoint=False) + theta_shift
    return R0 + a * np.cos(theta), Z0 + a * np.sin(theta)


def sample_psi_at_points(RR, ZZ, psi, Rs, Zs):
    R_vals = RR[:, 0]
    Z_vals = ZZ[0, :]
    psi_samples = np.empty(len(Rs), dtype=float)
    for k in range(len(Rs)):
        Rq, Zq = Rs[k], Zs[k]
        i = np.clip(np.searchsorted(R_vals, Rq) - 1, 0, len(R_vals) - 2)
        j = np.clip(np.searchsorted(Z_vals, Zq) - 1, 0, len(Z_vals) - 2)
        R1, R2 = R_vals[i], R_vals[i + 1]
        Z1, Z2 = Z_vals[j], Z_vals[j + 1]
        t = (Rq - R1) / (R2 - R1)
        u = (Zq - Z1) / (Z2 - Z1)
        psi_samples[k] = (
            (1-t)*(1-u)*psi[i,   j  ]
          +    t*(1-u)*psi[i+1, j  ]
          + (1-t)*  u *psi[i,   j+1]
          +    t*   u *psi[i+1, j+1]
        )
    return psi_samples


# ============================================================
# Fixed-boundary Green's table  (Anderson et al. eqs. 7-9)
#
# For each plasma grid point k, we find the shell currents Ives0_k
# that zero the boundary flux due to a unit current at k, then store
# the combined (plasma + shell) response as Gfixed.
#
# During iteration:  psi = Gfixed @ src_flat * dR * dZ
# satisfies psi = psi_edge on the shell exactly (to SVD tolerance).
#
# Arguments:
#   RR, ZZ      : 2-D meshgrid of plasma grid  (NR x NZ)
#   R0, a, Z0   : shell geometry (circular, radius a, centre R0,Z0)
#   Nshell      : number of toroidal shell filaments
#   delta_shell : radial offset of shell filaments outside plasma boundary
#   rcond       : SVD truncation threshold (None = machine eps)
# ============================================================
def build_fixed_boundary_Gtable(RR, ZZ, R0, a, Z0=0.0,
                                 Nshell=128, delta_shell=0.05, rcond=None):
    NR, NZ   = RR.shape
    Ngrid    = NR * NZ
    dR       = RR[1, 0] - RR[0, 0]
    dZ       = ZZ[0, 1] - ZZ[0, 0]

    # --- shell filament positions (just outside the boundary) ---
    Rs, Zs = circle_points(R0, a + delta_shell, Z0=Z0, N=Nshell)

    # --- boundary sample points (on the plasma boundary itself) ---
    Rb, Zb = circle_points(R0, a, Z0=Z0, N=Nshell, theta_shift=pi/Nshell)

    # --- Gav: flux at boundary points due to unit shell filament currents ---
    # shape (Nshell_boundary, Nshell)
    Gav =  G(
        Rb[:, None], Zb[:, None],
        Rs[None, :], Zs[None, :],
    )

    # --- Gp_flat: flux at boundary points due to each unit plasma grid current ---
    # Compute column by column would be NR*NZ solves; instead vectorise:
    # shape (Nshell_boundary, Ngrid)
    R_flat = RR.ravel()
    Z_flat = ZZ.ravel()
    Gap = G(
        Rb[:, None], Zb[:, None],
        R_flat[None, :], Z_flat[None, :],
    )   # (Nbdry, Ngrid)

    # --- SVD of Gav to solve  Gav @ Ives0 = -Gap  for all columns at once ---
    # Ives0 shape: (Nshell, Ngrid)
    U, s, Vt = np.linalg.svd(Gav, full_matrices=False)
    if rcond is None:
        rcond = np.finfo(float).eps * max(Gav.shape)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    # Ives0 = V @ diag(1/s) @ U.T @ (-Gap)
    Ives0 = Vt.T @ (s_inv[:, None] * (U.T @ (-Gap)))   # (Nshell, Ngrid)

    print(f"  SVD: {Nshell} singular values, "
          f"smallest kept = {s[s > rcond*s[0]][-1]:.3e}, "
          f"truncated = {np.sum(s <= rcond*s[0])}")

    # --- Gv: flux on full plasma grid due to unit shell filament currents ---
    # shape (Nshell, NR, NZ)
    Gv = G(
        RR[None, :, :], ZZ[None, :, :],
        Rs[:, None, None], Zs[:, None, None],
    )

    # --- Gp: free-space flux on plasma grid due to unit plasma grid currents ---
    # shape (NR, NZ, NR, NZ)  — build in blocks to avoid OOM
    # We store Gfixed as (Ngrid, Ngrid) = Gp_flat + (Gv reshaped) @ Ives0
    print("  Building Gp (free-space plasma table)...")
    Gp_flat =  G(
        RR.ravel()[:, None], ZZ.ravel()[:, None],
        R_flat[None, :],     Z_flat[None, :],
    )   # (Ngrid, Ngrid)

    # --- fixed-boundary table  (eq. 9) ---
    # Gfixed = Gp + Gv @ Ives0
    # Gv reshaped: (Nshell, Ngrid) -> need (Ngrid_obs, Nshell) @ (Nshell, Ngrid_src)
    Gv_mat  = Gv.reshape(Nshell, Ngrid)          # (Nshell, Ngrid_obs) — note transposed
    # psi_obs = Gp_flat @ src + Gv_mat.T @ (Ives0 @ src)
    #         = (Gp_flat + Gv_mat.T @ Ives0) @ src
    Gfixed  = Gp_flat + Gv_mat.T @ Ives0         # (Ngrid_obs, Ngrid_src)

    # Scale by dR*dZ so apply is just  psi_flat = -Gfixed @ src_flat
    Gfixed *= dR * dZ

    # --- verification: check boundary residual on a uniform source ---
    test_src = np.ones(Ngrid)
    psi_test = -Gfixed @ test_src
    psi_test_2d = psi_test.reshape(NR, NZ)
    bdry_resid = sample_psi_at_points(RR, ZZ, psi_test_2d, Rb, Zb)
    print(f"  Boundary verification (uniform src): "
          f"max|psi_shell| = {np.max(np.abs(bdry_resid)):.3e}  "
          f"(should be ~0)")

    return Gfixed, Rs, Zs, Rb, Zb


def apply_fixed_boundary_Gtable(src, Gfixed):
    """
    psi = -Gfixed @ src_flat
    The negative sign comes from  psi = -∫ G * S dR'dZ'
    where S = -mu0*R^2*p' - F*F' is the G-S source (eq. 13.51).
    dR*dZ scaling is already baked into Gfixed at build time.
    """
    psi_flat = -Gfixed @ src.ravel()
    return psi_flat.reshape(src.shape)


# ============================================================
# magnetic axis
# ============================================================
def find_magnetic_axis(RR, ZZ, psi, axis="min", mask=None):
    if mask is None:
        mask = np.ones_like(psi, dtype=bool)
    psi_masked = np.where(mask, psi, np.nan)
    idx_flat = np.nanargmin(psi_masked) if axis == "min" else np.nanargmax(psi_masked)
    idx = np.unravel_index(idx_flat, psi.shape)
    return {"R": RR[idx], "Z": ZZ[idx], "psi": psi[idx], "index": idx}


# ============================================================
# polynomial profiles
# ============================================================
def psi_to_s(psi, psi_axis, psi_edge):
    return (psi - psi_axis) / (psi_edge - psi_axis)


def poly_eval(x, coeffs):
    out = np.zeros_like(x, dtype=float)
    for k, c in enumerate(coeffs):
        out += c * x**k
    return out


def poly_eval_prime(x, coeffs):
    out = np.zeros_like(x, dtype=float)
    for k, c in enumerate(coeffs[1:], start=1):
        out += k * c * x**(k - 1)
    return out


def plasma_mask_from_s(s):
    return (s >= 0.0) & (s <= 1.0)


def poly_profile(psi, coeffs, psi_axis, psi_edge):
    denom_val = psi_edge - psi_axis
    out = np.zeros_like(psi, dtype=float)
    if abs(denom_val) < 1e-14:
        return out
    s = psi_to_s(psi, psi_axis, psi_edge)
    inside = plasma_mask_from_s(s)
    out[inside] = poly_eval(s[inside], coeffs)
    return out


def poly_profile_prime(psi, coeffs, psi_axis, psi_edge):
    denom_val = psi_edge - psi_axis
    out = np.zeros_like(psi, dtype=float)
    if abs(denom_val) < 1e-14:
        return out
    s = psi_to_s(psi, psi_axis, psi_edge)
    inside = plasma_mask_from_s(s)
    out[inside] = poly_eval_prime(s[inside], coeffs) / denom_val
    return out


def get_src_poly(pcoeffs, Fcoeffs, Rp, psi, psimask, psi_axis, psi_edge, src_scale=1.0):
    """
    G-S source S (eq. 13.51):
        S = -mu0 * R^2 * p'(psi) - F(psi) * F'(psi)
    """
    pprime = poly_profile_prime(psi, pcoeffs, psi_axis, psi_edge)
    F      = poly_profile(psi, Fcoeffs, psi_axis, psi_edge)
    Fprime = poly_profile_prime(psi, Fcoeffs, psi_axis, psi_edge)

    src = -mu0 * Rp * pprime - F * Fprime/Rp
    src = src_scale * src
    src = np.where(psimask, src, 0.0)
    return src


# ============================================================
# toroidal current diagnostic
# S = -mu0 * R * Jphi  =>  Jphi = -S / (mu0 * R)
# Ip = ∫∫ Jphi dR dZ
# ============================================================
def total_toroidal_current(src, R_vals, Z_vals):
    dR = R_vals[1] - R_vals[0]
    dZ = Z_vals[1] - Z_vals[0]
    RR, _ = np.meshgrid(R_vals, Z_vals, indexing="ij")
    Jtor = -src / (mu0 * RR)
    return np.sum(Jtor) * dR * dZ


# ============================================================
# plotting
# ============================================================
def plot_psi_contours(RR, ZZ, psi, psimask, axis_info, title, levels=40, savepath=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    psi_plot = np.where(psimask, psi, np.nan)
    cf = ax.contourf(RR, ZZ, psi_plot, levels=levels, cmap="viridis")
    ax.contour(RR, ZZ, psi_plot, levels=levels, colors="k", linewidths=0.6)
    ax.contour(RR, ZZ, psimask.astype(float), levels=[0.5], colors="r", linewidths=1.5)
    ax.plot(axis_info["R"], axis_info["Z"], "bo", ms=5)
    fig.colorbar(cf, ax=ax).set_label(r"$\psi$")
    ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title(title); ax.set_aspect("equal")
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200); plt.close(fig)
    else:
        plt.show()


def plot_initial_overview(RR, ZZ, psi, psimask, axis_info,
                          p_coeffs, F_coeffs, savepath=None):
    s       = np.linspace(0.0, 1.0, 300)
    p_vals  = poly_eval(s, p_coeffs)
    dp_vals = poly_eval_prime(s, p_coeffs)
    F_vals  = poly_eval(s, F_coeffs)
    dF_vals = poly_eval_prime(s, F_coeffs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    psi_plot = np.where(psimask, psi, np.nan)
    cf = ax.contourf(RR, ZZ, psi_plot, levels=40, cmap="viridis")
    ax.contour(RR, ZZ, psi_plot, levels=40, colors="k", linewidths=0.5)
    ax.contour(RR, ZZ, psimask.astype(float), levels=[0.5], colors="r", linewidths=1.5)
    ax.plot(axis_info["R"], axis_info["Z"], "bo", ms=5)
    fig.colorbar(cf, ax=ax).set_label(r"$\psi$")
    ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title(r"Initial $\psi$"); ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(s, p_vals,  "C0",   lw=2,   label=r"$p(s)$")
    ax.plot(s, dp_vals, "C0--", lw=1.5, label=r"$p'(s)$")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axvline(1, color="gray", lw=0.6, ls="--", label="edge")
    ax.set_xlabel(r"$s$"); ax.set_title("Pressure profile")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(s, F_vals,  "C1",   lw=2,   label=r"$F(s)$")
    ax.plot(s, dF_vals, "C1--", lw=1.5, label=r"$F'(s)$")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axvline(1, color="gray", lw=0.6, ls="--", label="edge")
    ax.set_xlabel(r"$s$"); ax.set_title(r"Toroidal field $F(\psi)$")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Initial state and profiles", fontsize=13)
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200); plt.close(fig)
    else:
        plt.show()


def plot_Jphi_contours(RR, ZZ, src, psimask, axis_info, title="J_phi", savepath=None):
    """Plot toroidal current density contours inside the plasma."""
    Jtor = -src / (mu0 * RR)
    Jtor_plot = np.where(psimask, Jtor, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))

    import matplotlib.colors as mcolors
    cf = ax.contourf(RR, ZZ, Jtor_plot, levels=40, cmap="RdBu_r",
                     norm=mcolors.CenteredNorm(vcenter=0))

    ax.contour(RR, ZZ, Jtor_plot, levels=40, colors="k", linewidths=0.4)
    ax.contour(RR, ZZ, psimask.astype(float), levels=[0.5], colors="r", linewidths=1.5)
    ax.plot(axis_info["R"], axis_info["Z"], "bo", ms=5)
    fig.colorbar(cf, ax=ax).set_label(r"$J_\phi$ (A/m²)")
    ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title(title); ax.set_aspect("equal")
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200); plt.close(fig)
    else:
        plt.show()



# ============================================================
# profiles
# ============================================================

## test 

A = 0.6
B = 1.4
F_coeffs = np.array([A, -A, -A*B, A*B, 0.0, 0.0], dtype=float)

# p(s) = p0*(1-s)^2   => p(0)=p0, p(1)=0, smooth decay
# = p0*(1 - 2s + s^2)
# coefficients [p0, -2*p0, p0, 0, 0, 0]
p0 = 1200.0
p_coeffs = np.array([p0, -2*p0, p0, 0.0, 0.0, 0.0], dtype=float)

# ============================================================
# constraint targets  (set any to None to disable)
# ============================================================
Ip_target    = None    # total toroidal plasma current [A], e.g. 100e3
Bphi_target  = None    # toroidal field at magnetic axis [T], e.g. 0.25
beta0_target = None    # central beta, e.g. 0.05

# ============================================================
# run parameters
# ============================================================
Plot_Every = 1000
save_dir   = "pngs_here"
os.makedirs(save_dir, exist_ok=True)

NR = 100
NZ = 100
R_vals = np.linspace(1.0, 2.0, NR)
Z_vals = np.linspace(-0.5, 0.5, NZ)

R0 = 1.5
Z0 = 0.0
a  = 0.50

src_scale = 1.0
alpha     = 0.1
alpha_min = 0.00005
alpha_max = 0.3
Niter     = 1000
axis_type = "min"

# fixed-boundary table parameters
Nshell      = 128    # number of shell filaments (and boundary sample points)
delta_shell = 0.05   # radial offset of shell filaments outside boundary [m]

psi_edge   = 0.0
psi_center = -5.0

# ============================================================
# setup
# ============================================================
RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing="ij")
Rij    = RR
psimask = circle_mask(RR, ZZ, R0, a, Z0=Z0)

# Build fixed-boundary Green's table (done once, replaces Gblocks + wall system)
print("Building fixed-boundary Green's table...")
Gfixed, Rs, Zs, Rb, Zb = build_fixed_boundary_Gtable(
    RR, ZZ, R0, a, Z0=Z0,
    Nshell=Nshell, delta_shell=delta_shell,
)
print("Done.\n")

# Initial psi: project the guess onto the fixed-boundary space
psi_plasma0, _ = initial_psi_plasma(RR, ZZ, R0, a, psi_edge, psi_center, Z0=Z0)
# For the initial guess we just use psi_plasma0 directly —
# it already satisfies psi~psi_edge outside, and the first G-S
# iteration will produce a proper fixed-boundary psi.
psi = psi_plasma0.copy()

axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)
psi_axis  = axis_info["psi"]

delta_inf_prev = np.inf

# Initial boundary residual diagnostic
bdry_resid_init = sample_psi_at_points(RR, ZZ, psi, Rb, Zb)
print(f"Initial | max|psi_boundary| = {np.max(np.abs(bdry_resid_init)):.3e}")

plot_initial_overview(
    RR, ZZ, psi, psimask, axis_info,
    p_coeffs, F_coeffs,
)

# ============================================================
# Picard iteration
# ============================================================
for j in range(Niter):
    psi_old      = psi.copy()
    psi_axis_old = psi_axis

    # G-S source  S = -mu0*R^2*p' - F*F'
    src = get_src_poly(
        p_coeffs, F_coeffs, Rij,
        psi_old, psimask, psi_axis_old, psi_edge,
        src_scale=src_scale,
    )

    # Fixed-boundary Green's table apply:  psi = -Gfixed @ src
    # Boundary condition psi=psi_edge is satisfied by construction.
    psi_fixedpoint = apply_fixed_boundary_Gtable(src, Gfixed)

    # Adaptive under-relaxation
    delta_trial     = psi_fixedpoint - psi_old
    delta_inf_trial = np.max(np.abs(delta_trial))

    if delta_inf_trial > 1.02 * delta_inf_prev:
        alpha = max(0.7 * alpha, alpha_min)
    elif delta_inf_trial < 0.98 * delta_inf_prev:
        alpha = min(1.02 * alpha, alpha_max)

    psi = (1.0 - alpha) * psi_old + alpha * psi_fixedpoint

    # Update magnetic axis every iteration
    axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)
    psi_axis  = axis_info["psi"]

    delta_inf = np.max(np.abs(psi - psi_old))
    Ip        = total_toroidal_current(src, R_vals, Z_vals)

    # Boundary residual diagnostic (cheap — sample on boundary points)
    bdry_resid = sample_psi_at_points(RR, ZZ, psi, Rb, Zb)
    shell_resid = np.max(np.abs(bdry_resid - psi_edge))

    # --------------------------------------------------------
    # optional constraint rescaling
    # --------------------------------------------------------
    if Bphi_target is not None:
        Bphi_current = F_coeffs[0] / axis_info["R"]
        if abs(Bphi_current) > 1e-14:
            F_coeffs = F_coeffs * (Bphi_target / Bphi_current)

    if beta0_target is not None:
        Bphi_axis = F_coeffs[0] / axis_info["R"]
        p0_target = beta0_target * Bphi_axis**2 / (2.0 * mu0)
        if abs(p_coeffs[0]) > 1e-14:
            p_coeffs = p_coeffs * (p0_target / p_coeffs[0])

    if Ip_target is not None and abs(Ip) > 1e-6:
        src_scale = src_scale * (Ip_target / Ip)

    print(
        f"iter {j:04d} | "
        f"alpha={alpha:.3e} | "
        f"psi_axis={psi_axis:.6e} | "
        f"Ip={Ip:.6e} A | "
        f"max|delta|={delta_inf:.3e} | "
        f"max|psi_shell|={shell_resid:.3e}"
    )

    delta_inf_prev = delta_inf_trial

    if j % Plot_Every == 0 and j != 0:
        plot_psi_contours(
            RR, ZZ, psi, psimask, axis_info,
            title=f"$\\psi$ at iteration {j}",
            savepath=f"{save_dir}/psi_{j:04d}.png",
        )

    if delta_inf < 1e-14:
        print(f"Converged at iteration {j}")
        break

plot_psi_contours(
    RR, ZZ, psi, psimask, axis_info,
    title=r"$\psi$ Final",
)

# Recompute src from final converged psi for consistent diagnostics
axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)
src_final = get_src_poly(
    p_coeffs, F_coeffs, Rij,
    psi, psimask, axis_info["psi"], psi_edge,
    src_scale=src_scale,
)

# Now plot Jphi  using src_final
plot_Jphi_contours(RR, ZZ, src_final, psimask, axis_info, title=r"$J_\phi$ Final")