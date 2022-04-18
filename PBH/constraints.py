import numpy as np
from os import listdir, getcwd
from os.path import isfile, join, dirname, realpath
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from .corrections import CM,Cz

Dict = {"Accretion" : ("cmb_acc.dat"),
              "CMB-Dip" : ("cmb_dipole.dat"),
              "BBN" : ("dhbbn.txt", "he.txt", "li.txt", "Y.txt"),
              "Disk Heating" : "dheating.dat",
              "Dynamical" : "dynamical.dat",
              "EGB" : ("egb.dat", "egb2.dat"),
              "EROS" : "eros.dat",
              "Galaxy Disruption" : "gal.dat",
              "GC Disruption" : "gc.dat",
               "GRB" : "grb.dat",
               "INTEGRAL" : "integral.dat",
               "GW" : "ligovirgo.dat",
               "LSS" : "lss.dat",
               "MACHOS" : "machos.dat",
               "Neutron Star" : "ns.txt",
               "OGLE" : "ogle.dat",
               "Radio Sources" : "radio.dat",
               "SUBARU" : "subaru.dat",
               "Wide Binaries" : "wbinaries.dat",
               "White Dwarfs" : "wdwarfs.dat",
               "X-ray Binaries" : "xrb.dat"
                }


class g:

    '''
    This class generates an object with the charasteristics of a particular g(M) at a particular redshift.
    Defining its particular function for the constrained mass for each formation scenario FCT and HC.
    '''
    
    def __init__(self,File,cosmo):

        self.path = dirname(realpath(__file__)) + '/data/gM/%s'%File

        self.z = self.__read_z__(self.path)
        
        self.DATA = np.loadtxt(self.path).transpose()
        
        self.g_int = interp1d(self.DATA[0],self.DATA[1],bounds_error=False,fill_value=0.)
        
        self.mmin = np.nanmin(self.DATA[0])
        
        self.mmax = np.nanmax(self.DATA[0])
        
        self.gmax = np.nanmax(self.DATA[1])
        
        self.cosmo = cosmo
        
        self.a = 1/(1+self.z)
        
        self.Mev = cosmo.Mev(self.a)

    def __read_z__(self,File):

        with open(File) as f:
    
            z_const = float(f.readline()[4:])

        return z_const

    def __repr__(self):

        string = 'g defined at z = %r'%self.z
        return string
        
    def __call__(self,M):
            
        return self.g_int(M)
    
    def plot(self):
        
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        
        ax.loglog(self.DATA[0],self.DATA[1])
        
        ax.set_xlim((self.mmin,self.mmax))
        
        ax.set_xlabel(r'$M$ [$M_\odot$]')
        ax.set_ylabel(r'$g(M)$')
        
        ax.grid(alpha=0.2)
        
        plt.show()
    
    def avg(self,MF):

        '''
        Computes the average output function <g(M)> (Eq. 71) weighted by the mass function. 
        '''

        M1ph = MF.M1ph[self.z]
        Mev = self.Mev

        lo = np.nanmax([np.log10(self.mmin),np.log10(Mev)])
        hi = np.nanmin([np.log10(self.mmax),M1ph])

        if hi < lo:

            # If both, numerator and denominator, are zero, then we do not have PBHs
            # to constraint, hence, this does not make sense. 

            return np.nan

        def integrand(logM):

            f1 = (self(10**logM)) # g(M)

            f2 = MF.dndlogM(logM) # dn/dlog(log(M))

            return f1*f2

        num = quad(integrand,lo,hi)[0]

        den = MF.number_density(lo,hi)

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

                # If the denominator is zero, then it is equivalent to have <g(M)> = infty.
                
                #print('ZeroDiv infty')
                
                result = 1e100

        return result


class f:

    def __init__(self, MF, g):

        self.g = g
        self.MF = MF
        self.z = g.z

        self.CM = CM(self.MF,self.g)
        self.Cz = Cz(self.MF,self.z)
        self.gavg = g.avg(self.MF)
        self.value = self.compute()

    def compute(self):

        '''
        It computes the value of the allowed fraction f for a particular mass function
        and observation given by g. This follows the expression given by Eq. (75).
        '''
        
        return ( self.CM * self.Cz ) / self.gavg




def list_g(*args,usedisputed=False):

    path = dirname(realpath(__file__)) + '/data/gM/'
    
    #path = 'PBH/data/gM'
    
    filenames = list(args)
    
    if len(filenames) == 0:
    
        filenames = [f for f in listdir(path) if isfile(join(path, f))]
        
    disputed = ["grb","wdwarfs","ns","subaru","cmb_acc","ligovirgo"]
    
    if usedisputed == False:
        
        for fname in filenames:
        
            name = fname[3:-4]
            
            if name in disputed:
                
                filenames.remove(fname)
                
    return filenames
