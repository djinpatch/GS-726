from inputs import *

def plot_psi_total_contours(
    R,
    Z,
    psi_coils,
    psi_plasma,
    R0=None,
    a=None,
    nlevels=30,
    levels=None,
    figsize=(7, 8),
    show_boundary=True,
    show_axis_guess=True,
):
    """
    Plot contours of psi_total = psi_coils + psi_plasma.

    Parameters
    ----------
    R, Z : 2D arrays
        Meshgrid arrays.
    psi_coils : 2D array
        Flux from external coils on the same grid.
    psi_plasma : 2D array
        Initial plasma flux guess on the same grid.
    R0 : float, optional
        Initial magnetic axis guess.
    a : float, optional
        Trial minor radius for boundary circle.
    nlevels : int
        Number of contour levels if `levels` is None.
    levels : array-like or None
        Explicit contour levels.
    figsize : tuple
        Figure size.
    show_boundary : bool
        Whether to draw the trial circular plasma boundary.
    show_axis_guess : bool
        Whether to mark (R0, 0).
    """
    psi_total = psi_coils + psi_plasma

    fig, ax = plt.subplots(figsize=figsize)

    # filled contours of total psi
    cf = ax.contourf(R, Z, psi_total, levels=100)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\psi_{\mathrm{total}}$")

    # line contours of total psi
    if levels is None:
        cs = ax.contour(R, Z, psi_total, levels=nlevels, colors="k", linewidths=0.8)
    else:
        cs = ax.contour(R, Z, psi_total, levels=levels, colors="k", linewidths=0.8)

    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2e")

    # optional trial boundary
    if show_boundary and (R0 is not None) and (a is not None):
        theta = np.linspace(0, 2 * np.pi, 400)
        Rb = R0 + a * np.cos(theta)
        Zb = a * np.sin(theta)
        ax.plot(Rb, Zb, "r--", lw=2, label="initial boundary")
        ax.legend(loc="best")

    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(r"Contours of $\psi_{\mathrm{coils}} + \psi_{\mathrm{plasma}}$")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()