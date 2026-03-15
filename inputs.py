import numpy as np
import pdb
# import torch
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe, ellipkm1

# constants 
pi = np.pi
mu0 = 4 * pi * 1e-7

# ------------------------------------------------------------
# external coil filiments 
## Format : R_location, Z_location, Current

I = 100

# octapole case
Coils = [
        (1.2,  0.3, I),
        (1.2, -0.3, I),
        (1.8,  0.3, I),
        (1.8, -0.3, I),
]


# ------------------------------------------------------------
# Grid
## Make sure this matches coil domain 

NR, NZ = 100, 100
R_vals = np.linspace(1.0, 2.0, NR)
Z_vals = np.linspace(-0.5, 0.5, NZ)
RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing="xy")

# ------------------------------------------------------------
# Initial plasma inputs 
## This sets the plasma center and radius
R0 = 1.5
a = .2
p0 = 3.
F0 = 3.


# ------------------------------------------------------------
# Flags? 
run_plasma_guess_example = True 
Niter = 100
