import numpy as np 
import matplotlib.pyplot as plt

def prof(a0, psi, nu):
    return a0 * (1 - psi**2)**nu


fig, ax = 