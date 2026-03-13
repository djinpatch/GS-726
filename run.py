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
