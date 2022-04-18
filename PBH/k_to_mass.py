import numpy as np 


def k_to_m(cosmo,k,fm):

    a_fct = cosmo.eof

    rho = ((cosmo.Om0/a_fct**3) + (cosmo.Or0/a_fct**4))*cosmo.rhoc*fm

    R = (2*np.pi/k) # Comoving Radius

    return (4./3)*np.pi * (a_fct*R)**3 * rho 