# -*- coding: utf-8 -*-




import numpy as np
import pandas as pd
import warnings
import time
import threading
import logging

from bisect import bisect_left
from synphot import SpectralElement
from scipy.interpolate import interp1d
from decimal import Decimal

import astropy.units as u

# Get around logging warnings of missing packages
logging.getLogger('isochrones').setLevel(logging.ERROR)

from isochrones.interp import DFInterpolator





# convert apparent to absolute mag
def abs_mag(app_mag, plx):
    return app_mag + (5 * (1 + np.log10(plx)))




# convert apparent mag error to absolute mag error
def abs_mag_error(e_app_mag, plx, e_plx):
    return np.sqrt(((5 * e_plx) / (plx * np.log(10)))**2 + e_app_mag**2)




# convert absolute to apparent mag
def app_mag(abs_mag, plx):
    return abs_mag - (5 * (1 + np.log10(plx)))




# convert absolute mag error to apparent mag error
def app_mag_error(e_abs_mag, plx, e_plx):
    return np.sqrt(((5 * e_plx) / (plx * np.log(10)))**2 + e_abs_mag**2)




# grab filter profiles and setup for synphot extinction calculation
def filter_profiles_setup(dirpath):
    filter_profiles = dict()
            
    # 2MASS
    # https://old.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html
    filter_profiles.update({'2mass_jmag' :  SpectralElement.from_file(dirpath + r'/2MASS.J.dat', wave_unit=u.um)})
    filter_profiles.update({'2mass_hmag' :  SpectralElement.from_file(dirpath + r'/2MASS.H.dat', wave_unit=u.um)})
    filter_profiles.update({'2mass_kmag' :  SpectralElement.from_file(dirpath + r'/2MASS.Ks.dat', wave_unit=u.um)})
    
    # Gaia 
    # https://www.cosmos.esa.int/web/gaia/edr3-passbands
    filter_profiles.update({'gaia_gmag' :  SpectralElement.from_file(dirpath + r'/GAIA_EDR3.G.dat', wave_unit=u.nm)})
    filter_profiles.update({'gaia_bpmag' :  SpectralElement.from_file(dirpath + r'/GAIA_EDR3.Gbp.dat', wave_unit=u.nm)})
    filter_profiles.update({'gaia_rpmag' :  SpectralElement.from_file(dirpath + r'/GAIA_EDR3.Grp.dat', wave_unit=u.nm)})
    
    # SDSS
    # https://www.sdss.org/instruments/camera/#Filters
    filter_profiles.update({'sdss_gmag' :  SpectralElement.from_file(dirpath + r'/SDSS.g.dat', wave_unit=u.AA)})
    filter_profiles.update({'sdss_rmag' :  SpectralElement.from_file(dirpath + r'/SDSS.r.dat', wave_unit=u.AA)})
    filter_profiles.update({'sdss_imag' :  SpectralElement.from_file(dirpath + r'/SDSS.i.dat', wave_unit=u.AA)})
    filter_profiles.update({'sdss_zmag' :  SpectralElement.from_file(dirpath + r'/SDSS.z.dat', wave_unit=u.AA)})
    
    # Johnson
    # http://svo2.cab.inta-csic.es/svo/theory/
    filter_profiles.update({'johnson_bmag' :  SpectralElement.from_file(dirpath + r'/Generic_Johnson.B.dat', wave_unit=u.AA)})
    filter_profiles.update({'johnson_vmag' :  SpectralElement.from_file(dirpath + r'/Generic_Johnson.V.dat', wave_unit=u.AA)})
    
    # Cousins
    # http://svo2.cab.inta-csic.es/svo/theory/
    filter_profiles.update({'cousins_rmag' :  SpectralElement.from_file(dirpath + r'/Generic_Cousins.R.dat', wave_unit=u.AA)})
    filter_profiles.update({'cousins_imag' :  SpectralElement.from_file(dirpath + r'/Generic_Cousins.I.dat', wave_unit=u.AA)})
    
    # TYCHO
    # http://svo2.cab.inta-csic.es/svo/theory/
    filter_profiles.update({'tycho_bmag' :  SpectralElement.from_file(dirpath + r'/TYCHO_TYCHO.B.dat', wave_unit=u.AA)})
    filter_profiles.update({'tycho_vmag' :  SpectralElement.from_file(dirpath + r'/TYCHO_TYCHO.V.dat', wave_unit=u.AA)})
    
    # Hipparcos
    # http://svo2.cab.inta-csic.es/svo/theory/
    filter_profiles.update({'hipparcos_hpmag' :  SpectralElement.from_file(dirpath + r'/Hipparcos_Hipparcos.Hp.dat', wave_unit=u.AA)})
    
    # PS1
    # https://ipp.ifa.hawaii.edu/ps1.filters/
    filter_profiles.update({'ps_gmag' :  SpectralElement.from_file(dirpath + r'/PS1.g.dat', wave_unit=u.nm)})
    filter_profiles.update({'ps_rmag' :  SpectralElement.from_file(dirpath + r'/PS1.r.dat', wave_unit=u.nm)})
    filter_profiles.update({'ps_imag' :  SpectralElement.from_file(dirpath + r'/PS1.i.dat', wave_unit=u.nm)})
    filter_profiles.update({'ps_zmag' :  SpectralElement.from_file(dirpath + r'/PS1.z.dat', wave_unit=u.nm)})
    filter_profiles.update({'ps_ymag' :  SpectralElement.from_file(dirpath + r'/PS1.y.dat', wave_unit=u.nm)})
    
    
    return filter_profiles




# DataFrame holding the wavelengths and zeropoints for all bands
def flux_meta():
    
    bands = [
            'johnson_bmag', 
            'johnson_vmag', 
            'cousins_rmag', 
            'cousins_imag',
            '2mass_jmag',
            '2mass_hmag',
            '2mass_kmag',
            'gaia_gmag',
            'gaia_bpmag',
            'gaia_rpmag',
            'sdss_gmag',
            'sdss_rmag',
            'sdss_imag',
            'sdss_zmag',
            'tycho_bmag',
            'tycho_vmag',
            'hipparcos_hpmag',
            'ps1_gmag',
            'ps1_imag',
            'ps1_omag',
            'ps1_rmag',
            'ps1_wmag',
            'ps1_ymag',
            'ps1_zmag',
            ]
    
    bands.sort()
    
    df = pd.DataFrame(index=bands)
    
    
    # Using effective wavelengths
    
    # http://svo2.cab.inta-csic.es/theory/fps/
    df.loc['johnson_bmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 4378.1, 6.293e-9, 5.679e-9
    df.loc['johnson_vmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 5466.1, 3.575e-9, 3.64326e-9
    # http://svo2.cab.inta-csic.es/theory/fps/
    df.loc['cousins_rmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 6357.96, 2.24563e-9, 2.69285e-9
    df.loc['cousins_imag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 7829.17, 1.20234e-9, 1.77589e-9
    # http://svo2.cab.inta-csic.es/theory/fps/
    df.loc['2mass_jmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 12350, 3.143e-10, 7.21192e-10
    df.loc['2mass_hmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 16620, 1.144e-10, 4.05446e-10
    df.loc['2mass_kmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 21590, 4.306e-11, 2.35016e-10
    # http://svo2.cab.inta-csic.es/theory/fps/ # they have GAIA DR3
    df.loc['gaia_gmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 5822.39, 2.50386e-9, 3.17259e-9
    df.loc['gaia_bpmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 5035.75, 4.07852e-9, 4.27793e-9
    df.loc['gaia_rpmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 7619.96, 1.26902e-9, 1.83971e-9
    # http://svo2.cab.inta-csic.es/theory/fps/ # no effective wavelength from sloan
    df.loc['sdss_gmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 4671.78, 5.45476e-9, 4.98749e-9
    df.loc['sdss_rmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 6141.12, 2.49767e-9, 2.88637e-9
    df.loc['sdss_imag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 7457.89, 1.38589e-9, 1.95711e-9
    df.loc['sdss_zmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 8922.78, 8.38585e-10, 1.36725e-9
    # http://svo2.cab.inta-csic.es/theory/fps/
    df.loc['tycho_bmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 4280.0, 6.5091e-9, 6.14866e-9
    df.loc['tycho_vmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 5340.0, 3.98353e-9, 3.95858e-9
    # http://svo2.cab.inta-csic.es/theory/fps/
    df.loc['hipparcos_hpmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 4897.85, 4.39107e-9, 4.5377e-9
    # http://svo2.cab.inta-csic.es/theory/fps/
    df.loc['ps1_gmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 4810.88, 5.04261e-9, 4.70324e-9
    df.loc['ps1_imag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 7503.68, 1.37212e-9, 1.9333e-9
    df.loc['ps1_omag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 6439.35, 1.88196e-9, 2.6252e-9
    df.loc['ps1_rmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 6156.36, 2.48016e-9, 2.8721e-9
    df.loc['ps1_wmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 5985.87, 2.45753e-9, 3.03803e-9
    df.loc['ps1_ymag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 9613.45, 7.14837e-10, 1.17784e-9
    df.loc['ps1_zmag', ['wavelength', 'VEGA_zeropoint', 'AB_zeropoint']] = 8668.56, 9.06849e-10, 1.44861e-9
    
    
    return df




# convert apparent mag to flux and optionally calculate error
def mag_to_flux(app_mag, zp, e_app_mag=None):
    f = zp * 10**(-app_mag/2.5)
    
    if e_app_mag is not None:
        error = np.abs(-np.log(10) * f * e_app_mag / 2.5)
        
        return f, error
    
    return f




# calculate the log probability of a point on a normal distribution
def log_normal(x, mu, sigma):
    
    if np.isnan(mu) or np.isnan(sigma):
        return 0.0
    
    return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma)) - 0.5 * (x - mu)**2 / sigma**2




# DataFrame of parameter labels used for plotting
# if `zero_extinction=True` (Av=0), drops Av from index
def plot_labels(zero_extinction=False):
    
    params = ['age',
              'mass',
              'Av',
              'f',
              'radius',
              'Teff',
              'density'
              ]
    
    df = pd.DataFrame(index=params)
    
    df.loc['age', ['label', 'fancy_label', 'fancy_label_unitless']] = 'age (Myr)', '$\\tau_{\mathsf{\\star}} \, \mathrm{(Myr)}$', '$\\tau_{\mathsf{\\star}}$'
    df.loc['mass', ['label', 'fancy_label', 'fancy_label_unitless']] = 'mass (M_Sun)', '$M_{\mathsf{\\star}} \, (\mathrm{M}_{\\odot})$', '$M_{\mathsf{\\star}}$'
    df.loc['Av', ['label', 'fancy_label', 'fancy_label_unitless']] = 'Av [mag]', '$A_V \ \mathrm{[mag]}$', '$A_V$'
    df.loc['f', ['label', 'fancy_label', 'fancy_label_unitless']] = 'f [mag]', '$f \ \mathrm{[mag]}$', '$f$'
    df.loc['radius', ['label', 'fancy_label', 'fancy_label_unitless']] = 'radius (R_Sun)', '$R_{\mathsf{\\star}} \, (\mathrm{R}_{\\odot})$', '$R_{\mathsf{\\star}}$'
    df.loc['Teff', ['label', 'fancy_label', 'fancy_label_unitless']] = 'Teff (K)', '$T_{\mathrm{eff}} \, \mathrm{(K)}$', '$T_{\mathrm{eff}}$'
    df.loc['density', ['label', 'fancy_label', 'fancy_label_unitless']] = 'density (M_Sun/R_Sun^3)', '$\\rho_{\mathsf{\\star}} \, (\mathrm{M}_{\\odot}/{R_{\\odot}}^3)$', '$\\rho_{\mathsf{\\star}}$)'
    
    if zero_extinction:
        df.drop(index='Av', inplace=True)
    
    return df




def phot_plot_labels():
    
    return {
        '2mass_jmag' : 'J',
        '2mass_hmag' : 'H',
        '2mass_kmag' : 'K$_s$',
        'gaia_gmag' : 'G',
        'gaia_bpmag' : 'G$_{BP}$',
        'gaia_rpmag' : 'G$_{RP}$',
        'sdss_gmag' : 'g${\prime}$',
        'sdss_rmag' : 'r${\prime}$',
        'sdss_imag' : 'i${\prime}$',
        'sdss_zmag' : 'z${\prime}$',
        'johnson_bmag' : 'B',
        'johnson_vmag' : 'V',
        'cousins_rmag' : 'R',
        'cousins_imag' : 'I',
        'tycho_bmag' : 'B$_T$',
        'tycho_vmag' : 'V$_T$',
        'hipparcos_hpmag' : 'H$_P$',
        'ps1_gmag' : 'g',
        'ps1_rmag' : 'r',
        'ps1_imag' : 'i',
        'ps1_zmag' : 'z',
        'ps1_ymag' : 'y',
        'ps1_omag' : 'o',
        'ps1_wmag' : 'w',
        }




# calculate the standard deviation of the residual
def residualSE(x, x_meas, N, ndim):
    
    return np.sqrt(
        np.sum((x_meas - x)**2) / (N - ndim)
        )




# residual
def residual(a, b):
    
    return a - b




# calculate the square root of sigma**2 given two different errors
def sigma(sigma_a, sigma_b):
    
    return np.sqrt(sigma_a**2 + sigma_b**2)




# fractional residual
def frac_res(a, b):
    
    return (a - b) / b




# fractional residual in units of sigma
def frac_res_sigma(a, sigma_a, b, sigma_b):
    
    return (a - b) / sigma(sigma_a, sigma_b)




# fractional difference
def frac_diff(a, b):
    
    return (a - b) / a




# fractional residual error
def frac_res_error(a, sigma_a, b, sigma_b):
    
    return np.sqrt((sigma_a / b)**2 + (a * sigma_b / b**2)**2)




# fractional difference error
def frac_diff_error(a, sigma_a, b, sigma_b):
    
    return np.sqrt(((b * sigma_a)/a**2)**2 + (sigma_b / a)**2)




# load isochrone table
def load_isochrone(filename):
    
    # suppress FutureWarning about element-wise comparison
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
        return pd.read_csv(filename, index_col=('age','mass'), dtype=np.float32, converters={'age':Decimal, 'mass':Decimal})
    
    
    
    
# use DFInterpolator to interpolate the isochrone model grid
def interpolate_true(idx, grid, agelist=None, masslist=None):
    
    age, mass = idx
    
    if type(grid) is str:
        grid = pd.read_pickle(grid)
    
    interpolator = DFInterpolator(grid)
    

    df = pd.DataFrame(columns=grid.columns, index=pd.MultiIndex.from_product([[age], [mass]], names=('age', 'mass')), dtype=float)
    
    df[df.columns] = interpolator([age, mass])
    
    del grid
    
    return df




# getting list index out of range for nearest masses
# "interpolate" by finding the nearest point in the isochrone model grid
def interpolate_nearest(idx, grid, agelist=None, masslist=None):
    
    age, mass = idx
    
    if type(grid) is str:
        grid = pd.read_pickle(grid)
    
    df = pd.DataFrame(columns=grid.columns, index=pd.MultiIndex.from_product([[age], [mass]], names=('age', 'mass')), dtype=float)
    
    
    if agelist is None:
        agelist = grid.index.get_level_values('age').drop_duplicates()
        
        
    try:
        nearest_ages = [agelist[bisect_left(agelist, age) - 1], agelist[bisect_left(agelist, age)]]
    except IndexError:
        df[df.columns] = np.nan
        if 'grid' in locals():
            del grid
        return df
    else:
        closest_age = nearest_ages[np.argmin(np.diff([nearest_ages[0], age, nearest_ages[1]]))]
        
        
    if masslist is None:
        masslist = grid.loc[closest_age].index.get_level_values('mass') # won't be duplicates in mass for a given age
        
        
    try:
        nearest_masses = [masslist[bisect_left(masslist, mass) - 1], masslist[bisect_left(masslist, mass)]]
    except IndexError:
        df[df.columns] = np.nan
    else:
        closest_mass = nearest_masses[np.argmin(np.diff([nearest_masses[0], mass, nearest_masses[1]]))]
       
    
    try:
        df[df.columns] = grid.loc[(closest_age, closest_mass)].values
    except KeyError:
        df[df.columns] = np.nan
    finally:
        if 'grid' in locals():
            del grid
            
        return df




# use nearest neighbor in age, then use `scipy.interpolate.interp1d` to interpolate mass
def interpolate_hybrid(idx, grid, agelist=None, masslist=None):
    
    age, mass = idx
    
    if type(grid) is str:
        grid = pd.read_pickle(grid)
        
    df = pd.DataFrame(columns=grid.columns, index=pd.MultiIndex.from_product([[age], [mass]], names=('age', 'mass')), dtype=float)
    
    
    if agelist is None:
        agelist = grid.index.get_level_values('age').drop_duplicates()
        
    
    try:
        nearest_ages = [agelist[bisect_left(agelist, age) - 1], agelist[bisect_left(agelist, age)]]
    except IndexError:
        df[df.columns] = np.nan
        if 'grid' in locals():
            del grid
        return df
    else:
        closest_age = nearest_ages[np.argmin(np.diff([nearest_ages[0], age, nearest_ages[1]]))]
    
    
    mass_df = grid.loc[closest_age]
    
    f = interp1d(mass_df.index.values, mass_df.values, kind='linear', axis=0, bounds_error=False, fill_value=np.nan, assume_sorted=False)
    
    
    try:
        df[df.columns] = f(mass)
    except KeyError:
        df[df.columns] = np.nan
    finally:
        if 'grid' in locals():
            del grid
            
        return df




# ``with WaitingAnimation(): ...`` to print a dynamic loading "..."
# combination of stackoverflow comments from
# https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor
# https://stackoverflow.com/questions/44606005/print-text-loading-with-dots-going-forward-and-backward-in-python-shell
class WaitingAnimation():
    
    def __init__(self, text='', n=3, delay=0.5):
        
        self.text = text
        
        self.delay = delay
        self.busy = False
        
        self.n = n
        
    def task(self):
        
        ndots = 1
        print('')
        while self.busy:
            print('\r' + self.text, ndots*'.', end="", flush=True)
            
            time.sleep(self.delay)
            
            if ndots == self.n:
                print('\b \b'*ndots, end="")
                
                ndots = 0
                
            else:
                ndots += 1
                
    def __enter__(self):
        
        self.busy = True
        threading.Thread(target=self.task).start()
        
    def __exit__(self, exception, value, traceback):
        
        self.busy = False
        time.sleep(self.delay)
        
        if exception is not None:
            return False
    
        
        
        


    












