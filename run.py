from inputs import *
from util import *
from plotting import *

if run_plasma_guess_example:

    print('Running plasma guess example...')
    psi_target = np.max(psi_on_circle(R0,a,Coils))
    psi_coils_0 , BR, BZ = multi_coil_fields(RR, ZZ, Coils)
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



R_vals_torch = torch.from_numpy(R_vals).to(device=device, dtype=torch.float32)
Z_vals_torch = torch.from_numpy(Z_vals).to(device=device, dtype=torch.float32)
R, Z = torch.meshgrid(R_vals_torch, Z_vals_torch, indexing="xy")

G4, dR, dZ = build_4D_Greensfunction(R_vals, Z_vals, device=device)
src = get_src(p0, F0, R, psi_target)


for i in iterationloop:
    psi = apply_greens(src, G4, dR, dZ, device=device)
    src = get_src(p0, F0, R, psi)