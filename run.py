from inputs import *
from util import *
from plotting import *

if run_plasma_guess_example:

    print('Running plasma guess example...')
    psi_target = np.min(psi_on_circle(R0,a,Coils))
    psi_coils_0 = multi_coil_fields(RR, ZZ, Coils)
    print('Calculating psi from coils...')
    psi_plasma_0 = initial_psi_plasma(RR, ZZ,psi_coils_0, R0, a, psi_target)
    print('Generating guess...')
   

    plot_psi_total_contours(
        RR,
        ZZ,
        psi_coils_0,
        psi_plasma_0*1)



else:
    print('No Flag Specified for Type of Run')





# print(f"device: {device}")

# R_vals_torch = torch.from_numpy(R_vals).to(device=device, dtype=torch.float32)
# Z_vals_torch = torch.from_numpy(Z_vals).to(device=device, dtype=torch.float32)
# R, Z = torch.meshgrid(R_vals_torch, Z_vals_torch, indexing="ij")

# G4, dR, dZ = build_4D_Greensfunction(R_vals, Z_vals, device=device)

# psi_total = torch.from_numpy(psi_plasma_0+psi_coils_0).to(device=device, dtype=torch.float32)
# src = get_src(p0, F0, R, psi_total)

# psi = apply_greens(src, G4, dR, dZ, device=device)
# pdb.set_trace()
# import matplotlib.pyplot as plt

# plt.figure(figsize=(6, 5))
# plt.contour(R_vals, Z_vals, psi.detach().cpu().numpy().T, levels=30)
# plt.xlabel("R")
# plt.ylabel("Z")
# plt.title(r"$\psi(R,Z)$")
# plt.tight_layout()
# plt.show()

# src = get_src(p0, F0, R, psi)