import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe, ellipkm1
from scipy.optimize import minimize

# ============================================================
# constants
# ============================================================
pi  = np.pi
mu0 = 4.0e-7 * pi
const = 1.0 / (2.0 * pi)

# ============================================================
# Green's function  — no mu0 factor
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
# grid helpers
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
# No mu0 in G — consistent with get_src_poly which carries mu0
# ============================================================
def build_fixed_boundary_Gtable(RR, ZZ, R0, a, Z0=0.0,
                                 Nshell=128, delta_shell=0.05, rcond=None):
    NR, NZ = RR.shape
    Ngrid  = NR * NZ
    dR     = RR[1, 0] - RR[0, 0]
    dZ     = ZZ[0, 1] - ZZ[0, 0]

    Rs, Zs = circle_points(R0, a + delta_shell, Z0=Z0, N=Nshell)
    Rb, Zb = circle_points(R0, a, Z0=Z0, N=Nshell, theta_shift=pi/Nshell)

    Gav = G(Rb[:, None], Zb[:, None], Rs[None, :], Zs[None, :])
    R_flat = RR.ravel()
    Z_flat = ZZ.ravel()
    Gap = G(Rb[:, None], Zb[:, None], R_flat[None, :], Z_flat[None, :])

    U, s, Vt = np.linalg.svd(Gav, full_matrices=False)
    if rcond is None:
        rcond = np.finfo(float).eps * max(Gav.shape)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    Ives0 = Vt.T @ (s_inv[:, None] * (U.T @ (-Gap)))

    print(f"  SVD: {Nshell} singular values, "
          f"smallest kept = {s[s > rcond*s[0]][-1]:.3e}, "
          f"truncated = {np.sum(s <= rcond*s[0])}")

    Gv = G(RR[None, :, :], ZZ[None, :, :],
           Rs[:, None, None], Zs[:, None, None])

    print("  Building Gp (free-space plasma table)...")
    Gp_flat = G(RR.ravel()[:, None], ZZ.ravel()[:, None],
                R_flat[None, :],     Z_flat[None, :])

    Gv_mat = Gv.reshape(Nshell, Ngrid)
    Gfixed = Gp_flat + Gv_mat.T @ Ives0
    Gfixed *= dR * dZ

    test_src = np.ones(Ngrid)
    psi_test = -Gfixed @ test_src
    bdry_res = sample_psi_at_points(RR, ZZ, psi_test.reshape(NR, NZ), Rb, Zb)
    print(f"  Boundary verification: max|psi_shell| = {np.max(np.abs(bdry_res)):.3e}  (should be ~0)")

    return Gfixed, Rb, Zb

def apply_fixed_boundary_Gtable(src, Gfixed):
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
# 5th degree polynomial profiles
# F(1) = 0  =>  F0 = -(F1+F2+F3+F4+F5)
# p(1) = 0  =>  p0 = -(p1+p2+p3+p4+p5)
# Free parameters: [F1,F2,F3,F4,F5, p1,p2,p3,p4,p5]  (10 total)
# ============================================================
def make_quintic_coeffs(free_params):
    F1, F2, F3, F4, F5 = free_params[0:5]
    p1, p2, p3, p4, p5 = free_params[5:10]
    F_coeffs = np.array([-(F1+F2+F3+F4+F5), F1, F2, F3, F4, F5], dtype=float)
    p_coeffs = np.array([-(p1+p2+p3+p4+p5), p1, p2, p3, p4, p5], dtype=float)
    return F_coeffs, p_coeffs

def pressure_penalty(p_coeffs, ns=300, penalty_weight=100.0):
    s = np.linspace(0.0, 1.0, ns)
    p_vals = poly_eval(s, p_coeffs)
    violations = np.maximum(0.0, -p_vals)
    return penalty_weight * np.trapezoid(violations**2, s)

# ============================================================
# polynomial evaluation
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

def get_src_poly(pcoeffs, Fcoeffs, Rp, psi, psimask, psi_axis, psi_edge):
    """S = -mu0*R^2*p'(psi) - F(psi)*F'(psi)"""
    pprime = poly_profile_prime(psi, pcoeffs, psi_axis, psi_edge)
    F      = poly_profile(psi, Fcoeffs, psi_axis, psi_edge)
    Fprime = poly_profile_prime(psi, Fcoeffs, psi_axis, psi_edge)
    src = -mu0 * Rp * pprime - F * Fprime/Rp
    src = np.where(psimask, src, 0.0)
    return src

# ============================================================
# toroidal current   S = -mu0*R*Jphi  =>  Jphi = -S/(mu0*R)
# ============================================================
def total_toroidal_current(src, R_vals, Z_vals):
    dR = R_vals[1] - R_vals[0]
    dZ = Z_vals[1] - Z_vals[0]
    RR, _ = np.meshgrid(R_vals, Z_vals, indexing="ij")
    Jtor = -src / (mu0 * RR)
    return np.sum(Jtor) * dR * dZ

# ============================================================
# inner G-S Picard loop
# ============================================================
def run_gs_inner(F_coeffs, p_coeffs,
                 Gfixed, RR, ZZ, Rij, R_vals, Z_vals,
                 psimask, psi_edge, axis_type,
                 psi_init,
                 alpha=0.1, alpha_min=0.00005, alpha_max=0.3,
                 Niter=2000, tol=1e-5, verbose=False):

    psi       = psi_init.copy()
    axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)
    psi_axis  = axis_info["psi"]
    delta_inf_prev = np.inf

    for j in range(Niter):
        psi_old      = psi.copy()
        psi_axis_old = psi_axis

        src = get_src_poly(
            p_coeffs, F_coeffs, Rij,
            psi_old, psimask, psi_axis_old, psi_edge,
        )

        psi_fixedpoint = apply_fixed_boundary_Gtable(src, Gfixed)

        delta_trial     = psi_fixedpoint - psi_old
        delta_inf_trial = np.max(np.abs(delta_trial))

        if delta_inf_trial > 1.02 * delta_inf_prev:
            alpha = max(0.7 * alpha, alpha_min)
        elif delta_inf_trial < 0.98 * delta_inf_prev:
            alpha = min(1.02 * alpha, alpha_max)

        psi = (1.0 - alpha) * psi_old + alpha * psi_fixedpoint

        axis_info = find_magnetic_axis(RR, ZZ, psi, axis=axis_type, mask=psimask)
        psi_axis  = axis_info["psi"]

        delta_inf      = np.max(np.abs(psi - psi_old))
        delta_inf_prev = delta_inf_trial

        if verbose:
            Ip = total_toroidal_current(src, R_vals, Z_vals)
            print(f"  inner {j:04d} | alpha={alpha:.3e} | "
                  f"psi_axis={psi_axis:.4e} | Ip={Ip:.4e} | delta={delta_inf:.3e}")

        if delta_inf < tol:
            if verbose:
                print(f"  Converged at inner iteration {j}")
            break

    Ip = total_toroidal_current(src, R_vals, Z_vals)
    return psi, axis_info, Ip, src

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

def plot_profiles(F_coeffs, p_coeffs, title="Profiles", savepath=None):
    s = np.linspace(0.0, 1.0, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(s, poly_eval(s, F_coeffs),       "C1",   lw=2,   label=r"$F(s)$")
    ax.plot(s, poly_eval_prime(s, F_coeffs), "C1--", lw=1.5, label=r"$F'(s)$")
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.axvline(1, color="gray", lw=0.6, ls="--")
    ax.set_xlabel(r"$s$"); ax.set_title(r"$F(\psi)$")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(s, poly_eval(s, p_coeffs),       "C0",   lw=2,   label=r"$p(s)$")
    ax.plot(s, poly_eval_prime(s, p_coeffs), "C0--", lw=1.5, label=r"$p'(s)$")
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.axvline(1, color="gray", lw=0.6, ls="--")
    ax.set_xlabel(r"$s$"); ax.set_title("Pressure")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(title); plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200); plt.close(fig)
    else:
        plt.show()


def plot_chi2():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(calls, chi2_h, "C0-", lw=1.2)
    ax.axvline(best_call, color="C1", ls="--", lw=1, label=f"best (call {best_call})")
    ax.scatter([best_call], [chi2_h[best_call-1]], color="C1", zorder=5, s=40)
    ax.set_xlabel("Call number"); ax.set_ylabel(r"$\chi^2$")
    ax.set_title(r"Optimization trajectory: $\chi^2$")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout(); plt.show()

def plot_diagnostics():
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax = axes[0]
    ax.plot(calls, np.abs(Ip_h)/1e3, "C0-", lw=1.2, label=r"$|I_p|$")
    ax.axhline(Ip_target/1e3, color="C3", ls="--", lw=1.5,
               label=f"target = {Ip_target/1e3:.0f} kA")
    ax.axvline(best_call, color="C1", ls=":", lw=1)
    ax.set_ylabel(r"$|I_p|$ (kA)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("Diagnostic trajectories")

    ax = axes[1]
    ax.plot(calls, Bphi_h, "C2-", lw=1.2, label=r"$B_\phi$ axis")
    ax.axhline(Bphi_target, color="C3", ls="--", lw=1.5,
               label=f"target = {Bphi_target} T")
    ax.axvline(best_call, color="C1", ls=":", lw=1)
    ax.set_ylabel(r"$B_\phi$ (T)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(calls, beta0_h, "C4-", lw=1.2, label=r"$\beta_0$")
    ax.axhline(beta0_target, color="C3", ls="--", lw=1.5,
               label=f"target = {beta0_target}")
    ax.axvline(best_call, color="C1", ls=":", lw=1)
    ax.set_ylabel(r"$\beta_0$"); ax.set_xlabel("Call number")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()

def plot_call(call_number):
    idx = call_number - 1
    if idx < 0 or idx >= Ncalls:
        print(f"Call {call_number} out of range (1 to {Ncalls})")
        return

    F_c    = F_coeffs_h[idx]
    p_c    = p_coeffs_h[idx]
    psi    = psi_h[idx]
    chi2_c = chi2_h[idx]

    s       = np.linspace(0.0, 1.0, 300)
    F_vals  = poly_eval(s, F_c)
    dF_vals = poly_eval_prime(s, F_c)
    p_vals  = poly_eval(s, p_c)
    dp_vals = poly_eval_prime(s, p_c)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Call {call_number} | $\\chi^2$={chi2_c:.3e} | "
        f"$B_\\phi$={Bphi_h[idx]:.3f} T | "
        f"$\\beta_0$={beta0_h[idx]:.4f} | "
        f"$I_p$={Ip_h[idx]/1e3:.1f} kA",
        fontsize=11,
    )

    ax = axes[0, 0]
    psi_plot = np.where(psimask, psi, np.nan)
    cf = ax.contourf(RR, ZZ, psi_plot, levels=40, cmap="viridis")
    ax.contour(RR, ZZ, psi_plot, levels=40, colors="k", linewidths=0.5)
    ax.contour(RR, ZZ, psimask.astype(float), levels=[0.5], colors="r", linewidths=1.5)
    fig.colorbar(cf, ax=ax).set_label(r"$\psi$")
    ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title(r"$\psi$ contours"); ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.semilogy(calls, chi2_h, "C0-", lw=1.0, alpha=0.7)
    ax.scatter([call_number], [chi2_c], color="C1", zorder=5, s=60,
               label=f"call {call_number}")
    ax.scatter([best_call], [chi2_h[best_call-1]], color="C3", zorder=5,
               s=60, marker="*", label=f"best (call {best_call})")
    ax.set_xlabel("Call number"); ax.set_ylabel(r"$\chi^2$")
    ax.set_title(r"$\chi^2$ trajectory")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 0]
    ax.plot(s, F_vals,  "C1",   lw=2,   label=r"$F(s)$")
    ax.plot(s, dF_vals, "C1--", lw=1.5, label=r"$F'(s)$")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axvline(1, color="gray", lw=0.6, ls="--", label="edge")
    ax.set_xlabel(r"$s$"); ax.set_title(r"$F(\psi)$ profile")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(s, p_vals,  "C0",   lw=2,   label=r"$p(s)$")
    ax.plot(s, dp_vals, "C0--", lw=1.5, label=r"$p'(s)$")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axvline(1, color="gray", lw=0.6, ls="--", label="edge")
    ax.set_xlabel(r"$s$"); ax.set_title("Pressure profile")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()



# ============================================================
# setup — same parameters as original working GS run
# ============================================================
save_dir = "pngs_here"
os.makedirs(save_dir, exist_ok=True)

NR = 100
NZ = 100
R_vals = np.linspace(1.0, 2.0, NR)
Z_vals = np.linspace(-0.5, 0.5, NZ)

R0 = 1.5
Z0 = 0.0
a  = 0.50

axis_type   = "min"
Nshell      = 128
delta_shell = 0.05
psi_edge    = 0.0
psi_center  = -5.0

RR, ZZ  = np.meshgrid(R_vals, Z_vals, indexing="ij")
Rij     = RR
psimask = circle_mask(RR, ZZ, R0, a, Z0=Z0)

print("Building fixed-boundary Green's table...")
Gfixed, Rb, Zb = build_fixed_boundary_Gtable(
    RR, ZZ, R0, a, Z0=Z0,
    Nshell=Nshell, delta_shell=delta_shell,
)
print("Done.\n")

# ============================================================
# initial profiles
# free_params = [F1,F2,F3,F4,F5, p1,p2,p3,p4,p5]
# ============================================================


A = 0.6
B = 1.4
F_coeffs_new = np.array([A, -A, -A*B, A*B, 0.0, 0.0], dtype=float)
print(f"F(1) = {np.sum(F_coeffs_new):.6f}  (should be 0)")

# p(s) = p0*(1-s)^2   => p(0)=p0, p(1)=0, smooth decay
# = p0*(1 - 2s + s^2)
# coefficients [p0, -2*p0, p0, 0, 0, 0]
p0 = 1200.0
p_coeffs_new = np.array([p0, -2*p0, p0, 0.0, 0.0, 0.0], dtype=float)
print(f"p(1) = {np.sum(p_coeffs_new):.6f}  (should be 0)")

free_params_0 = np.array([
    F_coeffs_new[1], F_coeffs_new[2], F_coeffs_new[3], F_coeffs_new[4], F_coeffs_new[5],
    p_coeffs_new[1], p_coeffs_new[2], p_coeffs_new[3], p_coeffs_new[4], p_coeffs_new[5],
], dtype=float)
# verify F0 and p0 are correctly derived
F_check, p_check = make_quintic_coeffs(free_params_0)
print(f"F0={F_check[0]:.4f}  (should be {A})")
print(f"p0={p_check[0]:.4f}  (should be {p0})")



F_coeffs_0, p_coeffs_0 = make_quintic_coeffs(free_params_0)
print(f"Initial F_coeffs: {F_coeffs_0}")
print(f"Initial p_coeffs: {p_coeffs_0}")
print(f"F(1) = {np.sum(F_coeffs_0):.2e}  (should be 0)")
print(f"p(1) = {np.sum(p_coeffs_0):.2e}  (should be 0)\n")

# ============================================================
# initial inner solve
# ============================================================
psi_plasma0, _ = initial_psi_plasma(RR, ZZ, R0, a, psi_edge, psi_center, Z0=Z0)

print("Running initial inner solve...")
psi_current, axis_info_0, Ip_current, src_current = run_gs_inner(
    F_coeffs_0, p_coeffs_0,
    Gfixed=Gfixed, RR=RR, ZZ=ZZ, Rij=Rij,
    R_vals=R_vals, Z_vals=Z_vals,
    psimask=psimask, psi_edge=psi_edge, axis_type=axis_type,
    psi_init=psi_plasma0,
    verbose=True,
)
Bphi_0  = F_coeffs_0[0] / axis_info_0["R"]
beta0_0 = 2.0 * mu0 * p_coeffs_0[0] / Bphi_0**2
print(f"Initial: psi_axis={axis_info_0['psi']:.4e} | Ip={Ip_current:.4e} A | "
      f"Bphi={Bphi_0:.4f} T | beta0={beta0_0:.4f}\n")

# ============================================================
# constraint targets — Ip, Bphi, beta0
# From Anderson et al. 2004, 380 kA standard MST discharge
# ============================================================
Ip_target      = 380e3    # A
Bphi_target    = 0.37     # T  (axis toroidal field, fig 7f)
beta0_target   = 0.025    # dimensionless  (2*mu0*p0/Bphi^2)
penalty_weight = 100.0
conv_tol       = 1e-6

Ip_target    = 100e3   # A
Bphi_target  = 0.25    # T  — from fig 5(a) axis value
beta0_target = 0.05

# ============================================================
# history — saved at every cost function call
# ============================================================
history = {
    "chi2":     [],
    "F_coeffs": [],
    "p_coeffs": [],
    "Ip":       [],
    "Bphi":     [],
    "beta0":    [],
    "psi":      [],
    "psi_axis": [],
}

# ============================================================
# state
# ============================================================
state = {
    "psi":         psi_current,
    "n_calls":     0,
    "best_chi2":   np.inf,
    "best_params": free_params_0.copy(),
    "best_F":      F_coeffs_0.copy(),
    "best_p":      p_coeffs_0.copy(),
    "best_psi":    psi_current.copy(),
    "calls_since_improvement": 0,

}

# ============================================================
# cost function — Ip, Bphi, beta0
# All three from self-consistent converged equilibrium.
# No src_scale — profiles alone control everything.
# ============================================================
class ConvergedException(Exception):
    pass

def cost(free_params):
    F_coeffs, p_coeffs = make_quintic_coeffs(free_params)

    psi_conv, axis_info, Ip, src = run_gs_inner(
        F_coeffs, p_coeffs,
        Gfixed=Gfixed, RR=RR, ZZ=ZZ, Rij=Rij,
        R_vals=R_vals, Z_vals=Z_vals,
        psimask=psimask, psi_edge=psi_edge, axis_type=axis_type,
        psi_init=state["psi"],
        verbose=False,
    )

    Bphi_current  = F_coeffs[0] / axis_info["R"]
    B0sq          = Bphi_current**2
    beta0_current = 2.0 * mu0 * p_coeffs[0] / B0sq if B0sq > 1e-20 else 0.0

    # chi2 — all three targets
    chi2 = (
        ((abs(Ip)       - Ip_target)    / Ip_target)**2
      + ((Bphi_current  - Bphi_target)  / Bphi_target)**2
      + ((beta0_current - beta0_target) / beta0_target)**2
    )
    chi2 += pressure_penalty(p_coeffs, penalty_weight=penalty_weight)

    # save to history
    history["chi2"].append(chi2)
    history["F_coeffs"].append(F_coeffs.copy())
    history["p_coeffs"].append(p_coeffs.copy())
    history["Ip"].append(float(Ip))
    history["Bphi"].append(float(Bphi_current))
    history["beta0"].append(float(beta0_current))
    history["psi"].append(psi_conv.copy())
    history["psi_axis"].append(float(axis_info["psi"]))

    

    # warm-start from this call
    state["psi"] = psi_conv
    '''

    # track best
    if chi2 < state["best_chi2"]:
        state["best_chi2"]   = chi2
        state["best_params"] = free_params.copy()
        state["best_F"]      = F_coeffs.copy()
        state["best_p"]      = p_coeffs.copy()
        state["best_psi"]    = psi_conv.copy()

    state["n_calls"] += 1

    print(
        f"call {state['n_calls']:04d} | "
        f"chi2={chi2:.4e} | "
        f"Ip={Ip:.3e} A | "
        f"Bphi={Bphi_current:.3f} T | "
        f"beta0={beta0_current:.4f} | "
        f"F0={F_coeffs[0]:.3f} | p0={p_coeffs[0]:.1f} | "
        f"best={state['best_chi2']:.4e}"
    )

    if state["best_chi2"] < conv_tol:
        raise ConvergedException(
            f"Converged at call {state['n_calls']}, chi2={state['best_chi2']:.3e}"
        )
    '''
    if chi2 < state["best_chi2"]:
        state["best_chi2"]             = chi2
        state["best_params"]           = free_params.copy()
        state["best_F"]                = F_coeffs.copy()
        state["best_p"]                = p_coeffs.copy()
        state["best_psi"]              = psi_conv.copy()
        state["calls_since_improvement"] = 0
    else:
        state["calls_since_improvement"] += 1
    
    state["n_calls"] += 1
    
    print(
        f"call {state['n_calls']:04d} | "
        f"chi2={chi2:.4e} | "
        f"Ip={Ip:.3e} A | "
        f"Bphi={Bphi_current:.3f} T | "
        f"beta0={beta0_current:.4f} | "
        f"F0={F_coeffs[0]:.3f} | p0={p_coeffs[0]:.1f} | "
        f"best={state['best_chi2']:.4e} | "
        f"no_improve={state['calls_since_improvement']}"
    )
    
    if state["best_chi2"] < conv_tol:
        raise ConvergedException(
            f"Converged at call {state['n_calls']}, chi2={state['best_chi2']:.3e}"
        )
    
    if state["calls_since_improvement"] >= 50:
        raise ConvergedException(
            f"No improvement for 50 calls, stopping at call {state['n_calls']}, "
            f"best chi2={state['best_chi2']:.3e}"
        )

    return chi2

# ============================================================
# Nelder-Mead optimization
# ============================================================
print("Starting Nelder-Mead optimization...")
print(f"Targets: Ip={Ip_target:.0f} A | Bphi={Bphi_target} T | beta0={beta0_target}\n")

try:
    result = minimize(
        cost,
        x0=free_params_0,
        method="Nelder-Mead",
        options={
            "maxiter": 2000,
            "xatol":   1e-4,
            "fatol":   1e-6,
            "disp":    True,
        },
    )
except ConvergedException as e:
    print(f"\nEarly exit: {e}")

# ============================================================
# final equilibrium — best solution, already self-consistent
# ============================================================
print("\nOptimization complete.")
print(f"  Best chi2 = {state['best_chi2']:.4e}")
print(f"  F_coeffs  = {state['best_F']}")
print(f"  p_coeffs  = {state['best_p']}")
print(f"  F(1) = {np.sum(state['best_F']):.2e}  (should be ~0)")
print(f"  p(1) = {np.sum(state['best_p']):.2e}  (should be ~0)")

psi_final   = state["best_psi"]
axis_final  = find_magnetic_axis(RR, ZZ, psi_final, axis=axis_type, mask=psimask)
src_final   = get_src_poly(
    state["best_p"], state["best_F"], Rij,
    psi_final, psimask, axis_final["psi"], psi_edge,
)
Ip_final    = total_toroidal_current(src_final, R_vals, Z_vals)
Bphi_final  = state["best_F"][0] / axis_final["R"]
beta0_final = 2.0 * mu0 * state["best_p"][0] / Bphi_final**2

print(f"\nFinal equilibrium:")
print(f"  psi_axis = {axis_final['psi']:.4e} Wb")
print(f"  Ip       = {Ip_final:.4e} A   (target {Ip_target:.0f} A)")
print(f"  Bphi     = {Bphi_final:.4f} T   (target {Bphi_target} T)")
print(f"  beta0    = {beta0_final:.4f}     (target {beta0_target})")

# ============================================================
# save history
# ============================================================
hist_path = f"{save_dir}/optimization_history.npz"
np.savez(
    hist_path,
    chi2     = np.array(history["chi2"]),
    F_coeffs = np.array(history["F_coeffs"]),
    p_coeffs = np.array(history["p_coeffs"]),
    Ip       = np.array(history["Ip"]),
    Bphi     = np.array(history["Bphi"]),
    beta0    = np.array(history["beta0"]),
    psi      = np.array(history["psi"]),
    psi_axis = np.array(history["psi_axis"]),
)
print(f"\nHistory saved to {hist_path}  ({len(history['chi2'])} calls)")

# ============================================================
# plots
# ============================================================
plot_psi_contours(
    RR, ZZ, psi_final, psimask, axis_final,
    title=r"$\psi$ Final (optimized)",
    savepath=f"{save_dir}/psi_final_opt.png",
)

plot_profiles(
    state["best_F"], state["best_p"],
    title=(f"Optimized profiles | "
           f"Ip={Ip_final:.2e} A | "
           f"Bphi={Bphi_final:.3f} T | "
           f"beta0={beta0_final:.4f}"),
    savepath=f"{save_dir}/profiles_opt.png",
)

# ============================================================
# history plotting — run this cell after optimization completes
# ============================================================
hist = np.load(f"{save_dir}/optimization_history.npz")

chi2_h     = hist["chi2"]
F_coeffs_h = hist["F_coeffs"]
p_coeffs_h = hist["p_coeffs"]
Ip_h       = hist["Ip"]
Bphi_h     = hist["Bphi"]
beta0_h    = hist["beta0"]
psi_h      = hist["psi"]

Ncalls = len(chi2_h)
calls  = np.arange(1, Ncalls + 1)
best_call = np.argmin(chi2_h) + 1



# --- call these ---
plot_chi2()
plot_diagnostics()

plot_call(best_call)