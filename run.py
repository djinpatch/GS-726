from inputs import *
from util import *
from plotting import *

if run_plasma_guess_example:

    print('Running plasma guess example...')
    psi_target = np.min(psi_on_circle(R0,a,Coils))
    psi_coils = multi_coil_fields(RR, ZZ, Coils)
    print('Calculating psi from coils...')
    psi_plasma_0 = initial_psi_plasma(RR, ZZ,psi_coils, R0, a, psi_target)
    print('Generating guess...')
   

    # plot_psi_total_contours(
    #     RR,
    #     ZZ,
    #     psi_coils_0,
    #     psi_plasma_0*1)



else:
    print('No Flag Specified for Type of Run')

Rij, Zij = np.meshgrid(R_vals, Z_vals, indexing='ij')
Gblocks = precompile_blocks(R_vals, Z_vals)
psi = (psi_plasma_0 + psi_coils).T

lcfs = find_lcfs_from_axis(Rij, Zij, psi, axis="min", nlevels=200)
psimask = make_lcfs_mask(RR, ZZ, lcfs)
src = get_src(p0, F0, Rij, psi, psimask)
pdb.set_trace()
for j in range(Niter):
    psi = apply_Gfunc_blocks(src, R_vals, Z_vals, Gblocks, blockR=25, blockZ=25) + psi_coils.T

    lcfs = find_lcfs_from_axis(Rij, Zij, psi, axis="min", nlevels=200)
    psimask = make_lcfs_mask(RR, ZZ, lcfs)
    src = get_src(p0, F0, Rij, psi, psimask)


