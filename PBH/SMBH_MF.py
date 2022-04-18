import numpy as np
from os.path import dirname, realpath
from scipy.interpolate import interp1d
from scipy.integrate import quad

def PS3(M, ns, A,cosmo):
    Mstar_ps=mmax #0.8
    alpha=(ns+3.)/3.0
    beta=1.+(ns/3)
    rho =  cosmo.rhoc * cosmo.Odm0
    amp=(beta*rho)*np.sqrt(1.0/np.pi)
    x=M/Mstar_ps
    yx0=np.power(x,alpha/2.)
    yx1=np.power(x,alpha)
    f1=yx0/(M*M)
    f2=np.exp(-yx1)
    return A*amp*f1*f2

def dMdHps(M):
    if M <mmin or M>1e20:
        return 0
    else:
        return dndm_interpolps(M)
    

def N_SMBH(lo=7,hi=16):

    '''
    Cumulative number of SMBH from lo to hi
    '''
    
    def g_m_integrand(u):
        M = np.power(10.,u) 
        f1 = dMdHps(M)/0.01
        result = f1
        return result
        
    numerator=quad(g_m_integrand,lo,hi,limit=200)[0]
                
    return numerator

def gen_interp(cosmo):

    '''
    Generates the interpolated function for the SMBH mass function from the PS fit.
    Needed in order to compute the Cumulative number of SMBH.

    '''

    global dndm_interpolps, mmin, mmax

    mh=[]
    dndmsmbh=[]

    filepath = dirname(realpath(__file__)) + '/data/halo_data/smbh.dat'

    with open(filepath) as fdat:
        for line in fdat:
            cols = [float(x) for x in line.split()]
            mf,ff = cols[0], cols[1]
            mh.append(np.power(10,mf))
            
            dndmsmbh.append(np.power(10,ff))
            


    mmin=min(mh)
    mmax=max(mh)

    ajuste=[]

    mx=np.logspace(np.log10(mmin),20)

    nsps=0.2 # Spectral index for the PS fit
    Aps=4.0 # Amplitude for the PS fit

    for i in range(len(mx)):
        ajuste.append(PS3(mx[i],nsps, Aps,cosmo))

    dndm_interpolps = interp1d(mx, ajuste)

    return dndm_interpolps

