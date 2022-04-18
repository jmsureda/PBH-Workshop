import numpy as np 
from scipy.optimize import root
from scipy.integrate import quad
from .constants import c_cgs,G_cgs,hbar_cgs,g_to_Ms,km_to_Mpc

def t_ev(M):

    '''
    
    Evaporation time in seconds of a BH. 

    Input: Mass of the BH in Solar Masses

    Returns: Evaporation time of the BH in seconds
    
    '''


    # Some Constants 
    # ----------------
    c = c_cgs # speed of light in cm per second

    hbar = hbar_cgs * g_to_Ms

    G = G_cgs / g_to_Ms 

    # ----------------
    
    evap = ((5120*np.pi*G**2)/(hbar*c**4))*M**3 # evaporation time in seconds

    return evap

def M_ev(t_ev, Solar = True):

    '''
    
    Computes the mass that will take a time t_ev to evaporate completely. By default in Solar Masses

    Input:  t_ev (Expected in seconds)
            Solar. To define if the result is in solar masses or in grams

    Returns: Mass (In Solar Masses by default)

    
    '''
    
    hbar = hbar_cgs # in cgs
    G = G_cgs # in cgs
    c = c_cgs # in cm per second

    if Solar == True:

        hbar = hbar * g_to_Ms

        G = G / g_to_Ms 
    
    Coef = np.power((hbar*c**4)/(5120*np.pi*G**2),1./3.)
    
    return Coef*np.power(t_ev,(1/3))


def a_ev(u,a_fct,cosmo):
    
    '''
    Recieves u = Log10(M) ---> M = 10**u
    
    Returns the scale factor on which a PBH of mass M, born in a scale factor a_fct, will evaporate all of its mass.
    
    '''
    
   # Some Constants 
    # ----------------
    c = c_cgs # speed of light in cm per second

    hbar = hbar_cgs * g_to_Ms

    G = G_cgs / g_to_Ms 

    # ----------------
    
    M = 10 ** u

    t_fct = cosmo.age(a_fct) # Initial time in seconds
    
    t_ev = ((5120*np.pi*G**2)/(hbar*c**4))*M**3 # evaporation time in seconds
    
    t_age = cosmo.age(1.)
    
    def eq_a(a,t1,t2,Omega_r0,Omega_m0,h0): # To solve numerically a(t_ev)
        
        def integrand(v, omega_r0, omega_m0):

            return 1./np.sqrt(omega_r0/v**2 + omega_m0/v + (1. - omega_r0 - omega_m0)*v**2)

        Units = (1/(100*h0*km_to_Mpc)) # Age of the Universe in Seconds

        integral = quad(integrand,0.,a,args=(Omega_r0,Omega_m0))[0]
        
        #print('int = ',integral)
        
        #print(a,Units*integral, t1+t2)
        
        return Units * integral - (t1+t2)
    
    
    if t_fct + t_ev  > t_age:
        
        print('Not Evaporated Yet')
        
        return np.nan
    
    else:
    
        a_f = root(eq_a,1e-5,args=(t_ev,t_fct,cosmo.Or0,cosmo.Om0,cosmo.h),method='hybr')

        return a_f['x'][0]