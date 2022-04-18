import numpy as np 
from scipy.integrate import quad


def Cz(MF, z):

    '''
    It computes the correction C_z as Eq. (74). 

    It receives as imput the mass function from class MassFunction and the redshift of Cz
    and the redshift z where this correction will be computed.

    '''

    if z == 0.:

        # If the objective redshifts is z=0, then the correction is exactly 1.

        return 1. 

    Mev_z0 = np.log10(MF.Mev(0))
    Mev_z = np.log10(MF.Mev(z))

    if MF.M1ph is None:
        
        MF.compute_M1ph()
    
    M1ph_z0 = MF.M1ph[0]
    
    try:

        M1ph_z = MF.M1ph[z]

    except:

        print('Invalid redshift value. Need to compute manually.')

        raise(ValueError)


    # We check here if these masses were given during the function call.
    # If not, then it computes them here.
    
    if M1ph_z0 < Mev_z0: # num = 0

        # If we do not have objects within the horizon that have not evaporated by today.

        num = 0
            
    else:

        num = MF.mass_density(lo = Mev_z0,hi = M1ph_z0)

    if M1ph_z < Mev_z: #den = 0

        # If we do not have objects within the horizon that have not evaporated at redshift z.
        
        den = 0
        
    else:

        den = MF.mass_density(lo = Mev_z,hi = M1ph_z)
            
    try:

        result = num/den
        
        if np.isinf(result):

            # This could happen if the denominator is very small compared to the numerator.
            # Then, we choose an arbitrary large number.
            
            #print('Is infinity')
            
            result = 1e100
        
        elif num == 0. or result < 1e-200:

            # If this is too small, we fix a small number that can be plotted in a log scale plot
        
            result = 1e-200
        
    except ZeroDivisionError:
        
        if num == 0:

            # If both, numerator and denominator, are zero, then we do not have PBHs
            # to constraint, hence, this does not make sense. 
            
            #print('ZeroDivError num = 0')
            
            result = np.nan # 0/0
        
        else:

            # If the denominator is zero, then it is equivalent to have Cz = infty.
            
            #print('ZeroDiv infty')
            
            result = 1e100
    
    return result

def CM(MF, g):

    '''
    It computes the correction C_M as Eq. (73). 

    It receives as imput the mass function from class MassFunction and the corresponding g function,
    related to the physical constraint.
    
    '''

    z = g.z
    M1ph = MF.M1ph[z] # log10(M1ph)
    Mev = np.log10(g.Mev) # log10(Mev)

    if M1ph < Mev: # num = 0

        # If we do not have objects within the horizon that have not evaporated by today.

        num = 0
            
    else:

        num = MF.mass_density(lo = Mev,hi = M1ph)

    num = MF.mass_density(Mev,M1ph)

    den_lo = np.nanmax([np.log10(g.mmin),Mev])
    den_hi = np.nanmin([np.log10(g.mmax),M1ph])

    def integrand(logM):

        f1 = (g(10**logM)/g.gmax) * (10**logM)

        f2 = MF.dndlogM(logM)

        return f1*f2

    
    if den_hi < den_lo:

        den = 0.

    else:

        den = quad(integrand,den_lo,den_hi,epsrel=1e-8,limit=50)[0]

    try:

        result = num/den
        
        if np.isinf(result):

            # This could happen if the denominator is very small compared to the numerator.
            # Then, we choose an arbitrary large number.
            
            #print('Is infinity')
            
            result = 1e100
        
        elif num == 0. or result < 1e-200:

            # If this is too small, we fix a small number that can be plotted in a log scale plot
        
            result = 1e-200
        
    except ZeroDivisionError:
        
        if num == 0:

            # If both, numerator and denominator, are zero, then we do not have PBHs
            # to constraint, hence, this does not make sense. 
            
            #print('ZeroDivError num = 0')
            
            result = np.nan # 0/0
        
        else:

            # If the denominator is zero, then it is equivalent to have CM = infty.
            
            #print('ZeroDiv infty')
            
            result = 1e100

    return result
    


