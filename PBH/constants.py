'''
This script contains the values for different physical constant and their convertion 
factors to the desired units

'''

G_cgs = 6.674*10**(-8) # Gravitational Constant [ cm^3 / (g*s^2) ]
c_kms = 299792 # Speed of Light [ km/s ]
hbar_cgs = 1.0546e-27 # Reduced Planck Constant in cgs
c_cgs = 3e10 # in cm per second

# Useful Unit Conversions

km_to_Mpc = 3.241*10**(-20) # Kilometers to MegaParsec
cm_to_Mpc = 3.241*10**(-25) # Centimeters to MegaParsec
g_to_Ms = 5.03e-34 # Grams to Solar Mass
s_to_yr = 1/(3.154e7) # Seconds to Years

G = G_cgs*(cm_to_Mpc**3)/(g_to_Ms) # Gracitational Constant [ Mpc^3 M_s^-1 s^-2 ]
c = c_kms * km_to_Mpc # Speed of Light [ Mpc / s]


# Parameters of the Power Spectrum as measured by Planck 2018

A0 = 2.10521e-9
k0 = 0.05 # Mpc^-1
ns = 0.9649 # Spectral Index
