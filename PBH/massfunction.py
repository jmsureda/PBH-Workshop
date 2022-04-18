import numpy as np
from numpy import float_power as power
import warnings
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from .misc import log_interp1d
from .constraints import f, g, list_g, Dict
from .corrections import Cz, CM
from . import SMBH_f

class Massfunction:

    '''
    To be written...
    '''

    def __init__(self, func, cosmo, name = '', **kwargs):
        
        self.cosmo = cosmo
        self.parameters = kwargs
        self.name = name
        self.__func = lambda logM : func(logM, self.cosmo, **self.parameters)

        self.An = 1.

        self.Mev = self.cosmo.Mev 
        self.Mmax = None

        self.f = None

        self.__init_integrals()
        
        self.M1ph = {}
        self.__compute_M1ph(1.) # M1ph at redshift 0

        self.__init_normalization()

    def __repr__(self):
        
        return "Mass Function '%s'"%self.name

    def __call__(self,logM,dndlogM = False):

        if dndlogM == True:

            return self.dndlogM(logM)

        else:
            
            return self.dndM(logM)

    def dndM(self,logM):

        return self.An * self.__func(logM)

    def dndlogM(self,logM):

        return power(10.,logM) * np.log(10) * self.dndM(logM)

    def plot(self,savefig = ''):

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})

        x = np.linspace(np.log10(self.Mev(0)), np.log10(self.Mmax),200)
        y = self.dndM(x)

        fig = plt.figure(dpi=120)
        ax1 = fig.add_subplot(111)
        ax1.semilogy(x, y)
        ax1.set_xlabel(r"$\log_{10}(M/M_\odot)$")
        ax1.set_ylabel(r"$\frac{dn}{dM}$")
        ax1.set_xlim(np.log10(self.Mev(0)),np.log10(self.Mmax)+1)
        ax1.set_ylim(np.max(y)/1e50,np.max(y))

        ax1.tick_params(axis = 'both', which ='both', direction = 'in',right = True, top = True)
        if savefig != '':
            plt.savefig(savefig)
        plt.show()

    def __init_Nden(self):

        '''
        This computes a cumulative number density from the most massive PBHs to the lighest ones.
        
        '''

        x = np.linspace(30,-60,1000)
        y = self.dndlogM(x)

        integral = -1. * cumtrapz(y, x, initial = 0.)

        nonzeros_id = np.where(integral != 0)

        # Use only non zero values of the integral
        x = x[nonzeros_id]
        integral = integral[nonzeros_id]

        # The first non-zero value in x is defined as the maximum mass (no more PBHs above that value).
        if self.Mmax is None:

            self.Mmax = power(10,x[0])

        self.Nden = log_interp1d(x, integral, which = 'y')

        self.__M_of_Nden = log_interp1d(integral, x, which = 'x') # logM as a function of log10(Number density)

    def __init_Mden(self):

        x = np.linspace(-60,30,1000)
        y = power(10.,x) * self.dndlogM(x)

        initial = power(10.,-60) * self.dndlogM(-60)

        integral = cumtrapz(y, x, initial = initial)

        self.Mden = log_interp1d(x, integral, which = 'y')
    
    def __init_integrals(self):

        self.__init_Nden()
        self.__init_Mden()
    
    def number_density(self, lo, hi):

        return self.An * (self.Nden(lo) - self.Nden(hi))
    

    def mass_density(self, lo, hi):

        return self.An * (self.Mden(hi) - self.Mden(lo))

    def mean_mass(self, lo = -60, hi = 20):

        '''
        Returns the Mean Mass for the PBH's distribution. This is the Mass density divided
        by the number density.
        '''

        numerator = self.mass_density(lo,hi)
        denominator = self.number_density(lo,hi)

        try:

            result = numerator/denominator

        except ZeroDivisionError:

            result = np.nan

        return result 

    def __init_M1ph(self):

        @np.vectorize
        def Full_M1ph(a):

            arg = np.log10((1./self.cosmo.V_h(a)) / self.An)
    
            result = self.__M_of_Nden(arg)

            return result

        a = np.logspace(-20,0,100)
        log_a = np.log10(a)

        log_M1ph = np.log10(Full_M1ph(a))

        #print(log_a,log_M1ph)

        interpolation = interp1d(log_a,log_M1ph)

        @np.vectorize
        def M1ph(z):

            a = 1 / (1 + z)

            return power(10., interpolation(np.log10(a)))

        self.M1ph = M1ph


        
    def __compute_M1ph(self, a):

        '''
        Computes the mass where the cumulative number density of the mass function equals 
        one per horizon volume.

        Compares the inverse of the comoving volume with the cumulative number density from a comoving mass function.
        '''

        z = 1/a - 1

        arg = np.log10((1./self.cosmo.V_h(a)) / self.An)

        self.M1ph[z] = self.__M_of_Nden(arg)

        return self.M1ph[z]

    def __generate_M1ph(self):

        ''' 
        This function computes the log10(M1ph) value for the relevant redshifts presented in 
        the paper z = {0 , 1 , 450 , 1e10}.

        Returns a dictionary.
        '''
        for z in [1., 450., 1160.72, 1e10]:

            if z not in self.M1ph.keys():

                self.__compute_M1ph(1. / (1. + z))

    def __compute_An(self):

        '''
        Calculates and update the required amplitude for the mass function such that the mass density,
        integrated between Mev(z=0) and M1ph(z=0) results in the DM density today.
        '''

        logMev = np.log10(self.Mev(0.))
        logM1ph = np.log10(self.M1ph[0])

        if logM1ph < logMev:

            self.An = np.nan
        
            return np.nan
    
        num = self.mass_density(lo = -60, hi = 20)
        den = self.mass_density(lo = logMev, hi = logM1ph)
    
        try:
            res = num/den
        
        except ZeroDivisionError:
            
            res = np.nan
        
        self.An = res

    def __init_normalization(self, steps = 20):

        '''
        Computes the corresponding An and M1ph iterating up to 20 times by default. If An converges within a 1% difference 
        from it last iterated value it will stop computing.
        
        '''

        for i in range(steps):

            An_old = self.An

            if i != 0:

                self.__compute_M1ph(1.)# Redshift 0

            self.__compute_An()

            error = np.abs(self.An - An_old) / ( (An_old + self.An) / 2)

            if (i != 0 and error < 0.01) or np.isnan(self.An):

                break

            elif i == steps - 1:

                print("An has not converged within %d steps"%steps)

    def Cz(self,z):

        return Cz(self,z)

    def CM(self,gM):

        return CM(self,gM)

    def compute_f(self, *args, SMBH = True ,usedisputed=False):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__generate_M1ph() # calcula M1ph a el resto de los z

        #if self.f is not None:

        #    return self.f

        if self.M1ph is None:

            self.__compute_M1ph()
        
        # If list is given
        if len(args) == 1 and isinstance(args[0],list):
            
            keys = args[0]
        
        else:
        
            keys = list(args)
        
        
        # If no argument is given
        if len(keys) == 0:
            
            filenames = list_g(usedisputed = usedisputed)
            
        else:
            
            names = []
            
            for key in keys:
                
                name = Dict[key]
                
                if isinstance(name,tuple):
                    for i in name:
                    
                        names.append('gm_' + i)
                        
                else:
                    
                    names.append('gm_' + name)
                
            filenames = list_g(*names, usedisputed = usedisputed)
        
        f_values = []

        for File in filenames:

            gM = g(File,self.cosmo)

            with warnings.catch_warnings():
                
                warnings.simplefilter("ignore")
                f_values.append(f(self,gM).value)

        ## Here the constraint of SMBH is calculated

        if SMBH:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                SMBH_f.init_dMdHI(self.cosmo)
                SMBH_f.init_AGN_MF(self.cosmo)
                f_smbh = SMBH_f.f(self)

            f_values.append(f_smbh)

        allnan = all(i != i for i in f_values)

        if allnan:

            self.f = np.nan
        
        else:
            
            f_values.append(10.)
            self.f = np.nanmin(f_values)

        return self.f
