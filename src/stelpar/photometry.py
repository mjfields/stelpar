# -*- coding: utf-8 -*-




from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import astropy.units as u

from synphot import SourceSpectrum, Observation, ExtinctionModel1D
from synphot.models import BlackBodyNorm1D
from synphot.reddening import ExtinctionCurve
from synphot import units

from dust_extinction.parameter_averages import CCM89

import numpy as np
import pandas as pd
from numba import njit

from .config import FILTERPROFILESPATH
from .metadata import PhotometryMetadata
from .utils import (abs_mag, 
                    abs_mag_error, 
                    log_normal, 
                    filter_profiles_setup, 
                    flux_meta, mag_to_flux, 
                    interpolate_true, 
                    interpolate_nearest, 
                    interpolate_hybrid
                    )

import warnings

FILTER_PROFILES = filter_profiles_setup(FILTERPROFILESPATH) # For whatever reason, synphot only works properly if this is a global variable (otherwise flux is nan)




__all__ = ['MeasuredPhotometry', 'SyntheticPhotometry']




class MeasuredPhotometry(object):
    
    """
    Handles all the photometry-getting from online databases for a single target.
    
    Parameters
    ----------
    name : str
        The name of the target as would be accepted by Vizier or Simbad.
    coords : SkyCoord, optional
        The coordinates (astropy SkyCoord object) of the target. If `None` (default),
        the name is used to query the object, otherwise, the coordinates are used.
    photometry_meta : DataFrame, optional
        The metadata used to access Vizier data and index DataFrames like `photometry`.
    search_radius : float or `astropy.units.quantity.Quantity`, optional
        The initial search radius for the Vizier query. Default is 5 arcseconds.
    radius_step : float or `astropy.units.quantity.Quantity`, optional
        How much is the search radius decreased every query if it needs to be iterated?
        Default is 1 arcsecond.
    tol : int, optional
        Minimum number of magnitudes needed to run the simulation. Default is 0.
    lazy_tol : bool, optional
        If `True` (default), will find the search radius where all queried
        tables contain 1 target. If `False`, will save each table successively
        once they contain only 1 target (trying to maximize photometry amount).
    vizier_kwargs : dict, optional
        Keyword arguments passable to `astroquery.vizier.Vizier` or related methods. See
        https://astroquery.readthedocs.io/en/latest/api/astroquery.vizier.VizierClass.html#astroquery.vizier.VizierClass
        for available attributes.
    isochrone_cols : list, optional
        If the found photometry is not in the list of columns for the working isochrone model grid,
        they are dropped from the photometry DataFrame. The default is `None`.
        
    """
    
    def __init__(self, name, coords=None, photometry_meta=None, search_radius=5*u.arcsec, radius_step=1*u.arcsec, tol=0, lazy_tol=False, vizier_kwargs=None, isochrone_cols=None):
        
        self.name = name
        self.coords = coords
        
        if photometry_meta is not None:
            self._photometry_meta = photometry_meta.copy()
        else:
            self._photometry_meta = PhotometryMetadata().photometry.copy()
        
        if isinstance(radius_step, u.quantity.Quantity):
            self._radius_step = radius_step
        else:
            self._radius_step = float(radius_step) * u.arcsec
            
        if isinstance(search_radius, u.quantity.Quantity):
            self._search_radius = search_radius
        else:
            self._search_radius = float(search_radius) * u.arcsec
            
        self._tol = tol
        self._lazy_tol = lazy_tol
        
        if vizier_kwargs is None:
            self._vizier_kwargs = dict()
        else:
            self._vizier_kwargs = vizier_kwargs
        
        self._isochrone_cols = isochrone_cols
        
        self._catalogs = self._photometry_meta.index.get_level_values('catalog').drop_duplicates().tolist()
        
            
            
            
    def simbad_plx(self):
        
        """
        Grabs the target parallax and error from Simbad.
        
        Returns
        -------
        plx, plx_err : float, float
            The target parallax and error in milliarcseconds.
            
        """
        
        # adds parallax and error to the queried table
        Simbad.add_votable_fields('plx', 'plx_error')
        
        if self.coords is None:
            query = Simbad.query_object(self.name)
            
        else:
            query = Simbad.query_region(self.coords, radius=self._search_radius)
        
        plx = query['PLX_VALUE']
        plx_err = query['PLX_ERROR']
        
        # if parallax or error is masked, assume it doesn't exist and return False
        if np.ma.is_masked(plx) or np.ma.is_masked(plx_err):
            return False
        
        
        return plx, plx_err ## in mas
            
            
    
    def get_data(self):
        
        """
        Grabs the target photometry from Vizier.
        
        Returns
        -------
        photometry : DataFrame
            Contains the measured photometry found for each filter band, the parallax and error, 
            and the magnitude system for each band.
        termination_message : str
            Indicates whether the data-grab was unsuccessful and why, or whether it was successful.
            
        """
        
        # forces Vizier to return all columns -- mostly to reveal TYCHO errors
        v = Vizier(columns=['**'], catalog=self._catalogs, **self._vizier_kwargs)
        
        if self.coords is None:
            query = v.query_object(self.name, radius=self._search_radius)
            
        else:
            query = v.query_region(self.coords, radius=self._search_radius)
            
        
        # got TableParseError around this line when running the full sim (not when running this class on it's own)
        # fixed by deleting all files within ~/.astropy/cache/astroquery/Vizier/
        # if problem persists, may need to add a line here to automatically clear the cache
        
        if len(query.keys()) == 0:
            termination_message = f"Failed for {self.name}: could not find photometry."
            
            return False, termination_message
        
        
        table_lengths = [len(query[table_name][query[table_name]['mode']==1]) if 'sdss' in table_name else len(query[table_name]) for table_name in query.keys()]
                
        
        ncatalogs = len(table_lengths)
        
        
        query_results = dict()
        
        
        if self._lazy_tol:
            
            if np.mean(table_lengths) > 1:
            
                while True:
                    
                    self._search_radius = self._search_radius - self._radius_step
                    
                    
                    if self._search_radius <= 0*u.arcsec:
                        termination_message = f"Failed for {self.name}: could not resolve a single object."
                    
                        return False, termination_message
                    
                    if self.coords is None:
                        new_query = v.query_object(self.name, radius=self._search_radius)
                        
                    else:
                        new_query = v.query_region(self.coords, radius=self._search_radius)
                    
                    
                    if len(new_query.keys()) == 0:
                        termination_message = f"Failed for {self.name}: could not find photometry."
                        
                        return False, termination_message
                    
                    
                    # in SDSS DR16, `mode=1` indicates a primary image
                    table_lengths = [len(new_query[table_name][new_query[table_name]['mode']==1]) if 'sdss' in table_name else len(new_query[table_name]) for table_name in new_query.keys()]
                    
                    
                    if np.mean(table_lengths) > 1:
                        continue
                    
                    else:
                        query = new_query
                        break
                    
        else:
            
            if 1 in table_lengths:
                ctlg = np.array(query.keys())[np.where(np.array(table_lengths) == 1)]
                
                query_results.update({c : (query[c][query[c]['mode']==1] if 'sdss' in c else query[c]) for c in ctlg.tolist()})
            
            if np.mean(table_lengths) > 1:
            
                while True:
                    
                    self._search_radius = self._search_radius - self._radius_step
                    
                    
                    if self._search_radius <= 0*u.arcsec:
                        if len(query_results.keys()) == 0:
                            termination_message = f"Failed for {self.name}: could not resolve a single object."
                        
                            return False, termination_message
                        
                        else:
                            query = query_results
                            break
                    
                    if self.coords is None:
                        new_query = v.query_object(self.name, radius=self._search_radius)
                        
                    else:
                        new_query = v.query_region(self.coords, radius=self._search_radius)
                    
                    
                    if len(new_query.keys()) == 0:
                        if len(query_results.keys()) == 0:
                            termination_message = f"Failed for {self.name}: could not find photometry."
                        
                            return False, termination_message
                        
                        else:
                            query = query_results
                            break
                    
                    table_lengths = [len(new_query[table_name][new_query[table_name]['mode']==1]) if 'sdss' in table_name else len(new_query[table_name]) for table_name in new_query.keys()]
                    
                    
                    if 1 in table_lengths:
                        ctlg = np.array(new_query.keys())[np.where(np.array(table_lengths) == 1)]
                        
                        query_results.update({c : (new_query[c][new_query[c]['mode']==1] if 'sdss' in c else new_query[c]) for c in ctlg.tolist()})
                        
                    
                    if len(query_results.keys()) == ncatalogs:
                        query = query_results
                        break
                    
                    else:
                        continue
                
            
        
        photometry = pd.DataFrame()
        
        # suppress `UserWarning` when masked element from table is converted to NaN 
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            meta = self._photometry_meta
            
            plx_mas, plxe_mas, ruwe = np.nan, np.nan, np.nan
            
            for catalog in self._catalogs:
                
                if catalog in list(query.keys()):
                    
                    if 'sdss' in catalog.lower():
                        table = query[catalog][query[catalog]['mode']==1]
                    else:
                        table = query[catalog]
                    
                    for band in meta.loc[catalog].index:
                        
                        mag, error, system, analog = meta.loc[(catalog, band)]
                        
                        photometry.loc[band, 'apparent_magnitude'], photometry.loc[band, 'apparent_magnitude_error'] = table[mag], table[error]
                        photometry.loc[band, 'system'] = system.upper()
                        photometry.loc[band, 'isochrone_analog'] = analog.lower()
                        
                    if 'gaia' in catalog.lower():
                        
                        plx_mas, plxe_mas = table['Plx'], table['e_Plx']
                        ruwe = table['RUWE'][0]
                        
                        
                if catalog.lower() == 'local':
                    
                    for band in meta.loc[catalog].index:
                        
                        mag, error, system, analog = meta.loc[(catalog, band)]
                        
                        photometry.loc[band, 'apparent_magnitude'], photometry.loc[band, 'apparent_magnitude_error'] = mag, error
                        photometry.loc[band, 'system'] = system.upper()
                        photometry.loc[band, 'isochrone_analog'] = analog.lower()
                
                
            if self._isochrone_cols is not None:
                for i in photometry.index:
                    
                    if i not in self._isochrone_cols:
                        photometry.drop(index=i, inplace=True)
            
            
            # if Vizier didn't have a parallax or parallax error, try Simbad        
            if (plx_mas is np.nan or plxe_mas is np.nan) or (np.ma.is_masked(plx_mas) or np.ma.is_masked(plxe_mas)):
                plx_solution = self.simbad_plx()
                
                if plx_solution is False:
                    term_message = f"Failed for {self.name}: could not find a viable parallax and/or parallax error."
                    
                    return False, term_message
                
                else:
                    plx_mas, plxe_mas = plx_solution
                
        
        if len(photometry.index) < self._tol:
            term_message = f"Failed for {self.name}: could not find enough magnitudes ({self._tol} needed). Try setting `tol` to a lower value."
            
            return False, term_message
        
        
        # parallaxes are converted from milliarcsec to arcsec
        parallax = plx_mas[0] / 1000
        parallax_error = plxe_mas[0] / 1000
        
        
        # Luri et al. 2018 suggest a full Bayesian approach for dealing with negitive parallaxes
        # I'm ignoring this and cutting them out b/c I can't be bothered right now
        if parallax < 0 or parallax_error < 0:
            term_message = f"Failed for {self.name}: parallax or error is negative. Ignoring Luri+2018 and removing the target."
            
            return False, term_message
        
        
        photometry['ABSOLUTE_MAGNITUDE'] = photometry.loc[:, 'apparent_magnitude'].apply(abs_mag, args=([parallax]))
        photometry['ABSOLUTE_MAGNITUDE_ERROR'] = photometry.loc[:, 'apparent_magnitude_error'].apply(abs_mag_error, args=([parallax, parallax_error]))
        
        
        photometry['parallax'] = parallax
        photometry['parallax_error'] = parallax_error
        
        photometry = photometry.dropna()
        
        photometry['RUWE'] = ruwe
        
        
        term_message = f"Success for {self.name}: photometry found."
        
        
        # calculate measured flux and error here
        for band in photometry.index:
            
            analog = photometry.loc[band, 'isochrone_analog']
            
            photometry.loc[band, 'wavelength'] = flux_meta().loc[analog, 'wavelength']
            
            system = photometry.loc[band, 'system'].upper()
            photometry.loc[band, 'zeropoint_flux'] = flux_meta().loc[analog, f'{system}_zeropoint']
            
            photometry.loc[band, ['flux', 'flux_error']] = mag_to_flux(*photometry.loc[band, ['apparent_magnitude', 'zeropoint_flux', 'apparent_magnitude_error']])
        
        
        
        # put 'system' and 'isochrone_analog' columns last
        photometry = photometry[[
            'apparent_magnitude',
            'apparent_magnitude_error',
            'ABSOLUTE_MAGNITUDE',
            'ABSOLUTE_MAGNITUDE_ERROR',
            'flux',
            'flux_error',
            'parallax',
            'parallax_error',
            'RUWE',
            'zeropoint_flux',
            'wavelength',
            'system',
            'isochrone_analog'
            ]]
                
        
        return photometry, term_message




class SyntheticPhotometry(object):
    
    """
    Uses `synphot` to calculate extinction and add it to the estimated magnitudes.
    See http://learn.astropy.org/rst-tutorials/color-excess.html?highlight=filtertutorials#example-3-calculate-color-excess-with-synphot
    for mor information
    
    Parameters
    ----------
    photometry_df : DataFrame
        The DataFrame containing the target's measured photometry.
    model_grid : DataFrame or str, optional
        The full isochrone grid DataFrame or the file location (string) of
        the cached, serialized grid. The latter is preferred when using the pre-interpolated grid 
        because the memory usage of that DataFrame is higher than the uninterpolated frame.
        The default is `None` which is acceptable if only using extinction methods.
    interp_method : str, optional
        If 'true', uses the standard interpolation method of DFInterpolator.
        If 'nearest' uses nearest-neighbor interpolation. If 'hybrid' uses
        nearest-neighbor interpolation for age and DFInterpolator for mass.
        The default is 'true'.
    extinction_kwargs : dict, optional
        Can pass additional keyword arguments to be used by the extinction functions.
    interp_kwargs : dict, optional
        Can pass additional keyword arguments to be used by the interpolator function.
    
    """
    
    def __init__(self, photometry_df, model_grid=None, interp_method='true', extinction_kwargs=None, interp_kwargs=None):
        
        self.model_grid = model_grid
        
        self._photometry = photometry_df.copy()
        
        self._extinction = pd.DataFrame(index=self._photometry.index)
        
        self._vega = SourceSpectrum.from_file(FILTERPROFILESPATH + r'/alpha_lyr_stis_010.fits')
        
        self._wav = np.arange(1000, 30000, 10) * u.angstrom
        
        self._band_array = np.zeros((self._photometry.shape[0], self.wav_array.shape[0]))
        
        for ii, b in enumerate(self._photometry.index):
            
            unit = self._photometry.loc[b, 'system'] + 'mag'
            self._extinction.loc[b, 'flux_unit'] = unit.upper()
            
            analog = self._photometry.loc[b, 'isochrone_analog']
            self._band_array[ii] = np.array(FILTER_PROFILES[analog]._get_arrays(self.wav_array)[1])
        
        
        if interp_method.lower() == 'true':
            self._interp_func = interpolate_true
            
        elif interp_method.lower() == 'nearest':
            self._interp_func = interpolate_nearest
            
        elif interp_method.lower() == 'hybrid':
            self._interp_func = interpolate_hybrid
            
        else:
            raise ValueError(
                f"Invalid interpolation method: {interp_method}"
                )
            
        if extinction_kwargs is None:
            self._extinction_kwargs = dict()
        else:
            self._extinction_kwargs = extinction_kwargs
        
        if interp_kwargs is None:
            self._interp_kwargs = dict()
        else:
            self._interp_kwargs = interp_kwargs
        
        
        
        
    @property
    def wav_array(self):
        
        return self._wav.value # Angstroms, synphot default
    
    
    
    
    @property
    def band_array(self):
        
        return self._band_array
    
    
    
    
    @staticmethod
    @njit
    def _effective_stimulus(wavelength, flux_array, band_array):
        """
        A pared-down version of `:func: synphot.Observation.effstim` using numpy
        arrays. Calculates the radiation that would be observed given some 
        incoming flux through a bandpass filter over some wavelength.
        Compiled using `numba` just-in-time (JIT) compilation.

        Parameters
        ----------
        wavelength : numpy.ndarray
            1-dimensional array of wavelength in angstroms.
        flux_array : numpy.ndarray
            2-dimensional array of flux values for each bandpass filter
            (actually flux * transmission frac) in FLAM units (i.e. flux 
            density with respect to wavelength).
        band_array : numpy.ndarray
            2-dimensional array of transmission fractions for each bandpass 
            filter at each wavelength.

        Returns
        -------
        res_array : numpy.ndarray
            1-dimensional array of effective stimulus values in each bandpass
            filter.

        """
        
        res_array = np.zeros(len(flux_array))
    
        for ii in range(len(flux_array)):
            
            flux = flux_array[ii]
            band = band_array[ii]
            
            num = abs(np.trapz(wavelength * flux, x=wavelength))
            den = abs(np.trapz(wavelength * band, x=wavelength))
            
            res_array[ii] = num / den
        
        return res_array
    
    
    
    
    @staticmethod
    @njit
    def _pivot(wavelength, band_array, quantum_eff=False):
        """
        Calculates the pivot wavelength used to transform flux density from
        wavelength to frequency units. 
        Compiled using `numba` just-in-time (JIT) compilation.

        Parameters
        ----------
        wavelength : numpy.ndarray
            1-dimensional array of wavelength in angstroms.
        band_array : numpy.ndarray
            2-dimensional array of transmission fractions for each bandpass 
            filter at each wavelength.
        quantum_eff : bool, optional
            Should the 'quantum-efficiency' convention be used to calculate 
            the pivot wavelength (as opposed to the equal-energy convention). 
            The default is False.

        Returns
        -------
        piv : numpy.ndarray
            1-dimensional array of pivot wavelengths in angstroms for each
            bandpass filter.

        """
        
        piv = np.zeros(band_array.shape[0])
        
        for ii in range(band_array.shape[0]):
            
            band = band_array[ii]
            
            if quantum_eff: # quantum-efficiency response function
                
                num = np.trapz(wavelength * band, x=wavelength)
                den = np.trapz(band / wavelength, x=wavelength)
        
                piv[ii] = np.sqrt(abs(num / den))
            
            else: # equal-energy response function # SVO2 filter profiles
                
                num = np.trapz(band, x=wavelength)
                den = np.trapz(band / wavelength**2, x=wavelength)
        
                piv[ii] = np.sqrt(abs(num / den))
                
        
        return piv # Angstroms
    
    
    
    
    def _numpy_extinction(self, Av, Teff):
        """
        Calculate pre- and post-extinction flux values to calculate extinction.

        Parameters
        ----------
        Av : float
            The V-band extinction in mag from which the extinction for other bands is calculated.
        Teff : float
            The effective temperature of the target in K.

        Returns
        -------
        val_before : numpy.ndarray
            1-dimensional array of flux values in FLAM for every bandpass filter
            that have not been synthetically extincted. Essentially the flux from
            the source blackbody.
        val_after : numpy.ndarray
            1-dimensional array of flux values in FLAM for every bandpass filter
            after the source flux has been synthetically extincted. The source
            flux * the extinction transmission function
        ex_array : numpy.ndarray
            2-dimensional array of fractional values that tell whether how much
            flux will be extincted in each bandpass filter at every wavelength.

        """
        
        sp = SourceSpectrum(BlackBodyNorm1D, temperature=Teff)
        sp_array = sp._get_arrays(self._wav)[1].value # PHOTLAM, default
        before_flam = units.convert_flux(self.wav_array, sp_array, units.FLAM).value
        
        ext = CCM89(Rv=3.1).extinguish(self._wav, Av=Av)
        ex = ExtinctionCurve(ExtinctionModel1D, points=self._wav, lookup_table=ext)
        ex_array = ex._get_arrays(self._wav)[1].value # transmission frac from 0 - 1
        
        sp_ext_array = sp_array*ex_array
        after_flam = units.convert_flux(self.wav_array, sp_ext_array, units.FLAM).value
        
        
        before_flux = before_flam * self.band_array
        after_flux = after_flam * self.band_array
        
        
        val_before = self._effective_stimulus(self.wav_array, before_flux, self.band_array) * units.FLAM
        val_after = self._effective_stimulus(self.wav_array, after_flux, self.band_array) * units.FLAM
        
        
        return val_before, val_after, ex_array
    
    
    
    
    # calculate the effective stimulus for vega to convert other calculated fluxes to VEGAMAG
    def _vega_effstim(self):
    
        vega_array = self._vega._get_arrays(self._wav)[1].value # PHOTLAM, default
        flam = units.convert_flux(self.wav_array, vega_array, units.FLAM).value
        
        before_flux = flam * self.band_array
        
        val = self._effective_stimulus(self.wav_array, before_flux, self.band_array) * units.FLAM
        
        
        return val
        
        
        
        
    def _synphot_extinction(self, Av, Teff):
        """
        Calculates extinction for every band using `synphot`.

        Parameters
        ----------
        Av : float
            The V-band extinction in mag from which the extinction for other bands is calculated.
        Teff : float
            The effective temperature of the target in K.

        Returns
        -------
        extinction : DataFrame
            Contains the calculated extinction and flux unit for each band.

        """
        
        sp = SourceSpectrum(BlackBodyNorm1D, temperature=Teff)
        
        ext = CCM89(Rv=3.1).extinguish(self._wav, Av=Av)    
        ex = ExtinctionCurve(ExtinctionModel1D, points=self._wav, lookup_table=ext)
        
        sp_ext = sp*ex
        
        
        for band in self._extinction.index:
            
            unit = self._extinction.loc[band, 'flux_unit']
            analog = self._photometry.loc[band, 'isochrone_analog']
            
            sp_obs = Observation(sp_ext, FILTER_PROFILES[analog], force='extrap')
            sp_obs_before = Observation(sp, FILTER_PROFILES[analog], force='extrap')
    
            sp_stim_before = sp_obs_before.effstim(flux_unit=unit, vegaspec=self._vega)
            sp_stim = sp_obs.effstim(flux_unit=unit, vegaspec=self._vega)
        
            A_calc = sp_stim - sp_stim_before
            
            
            self._extinction.loc[band, 'extinction'] = A_calc.value
            
        return self._extinction
    
    
    
    
    def calculate_extinction(self, Av, Teff, use_synphot=False, pivot_quantum_eff=False):
        """
        Calculates extinction using the results from `self._synphot_extinction`
        or `self._numpy_extinction`. If the latter (i.e., `use_synphot=False`)
        the values are converted to the desired flux units within this function
        (`synphot` does it internally).

        Parameters
        ----------
        Av : float
            The V-band extinction in mag from which the extinction for other bands is calculated.
        Teff : float
            The effective temperature of the target in K.
        use_synphot : bool, optional
            Use the built-in synphot calculation or calculate extinction with
            `numpy` arrays which is faster. The default is False.
        pivot_quantum_eff : bool, optional
            Should the 'quantum-efficiency' convention be used to calculate 
            the pivot wavelength (as opposed to the equal-energy convention). 
            The default is False.

        Returns
        -------
        self._extinction, DataFrame
            Contains the calculated extinction and the extinction-corrected
            photometry for the target.

        """
        
        if use_synphot:
        
            return self._synphot_extinction(Av, Teff)
        
        
        val_before, val_after, ex_array = self._numpy_extinction(Av, Teff) # FLAM
        
        
        if 'VEGAMAG' in self._extinction['flux_unit'].values:
            
            vega_val = self._vega_effstim() # FLAM
            
        
        if 'ABMAG' in self._extinction['flux_unit'].values:
            
            wav_pivot = self._pivot(self.wav_array, self.band_array, quantum_eff=pivot_quantum_eff)
            fnu_before = units.convert_flux(wav_pivot, val_before, units.FNU)
            fnu_after = units.convert_flux(wav_pivot, val_after, units.FNU)
            
        
        for ii, b in enumerate(self._extinction.index):
            
            unit = self._extinction.loc[b, 'flux_unit']
            
            if unit.upper() == 'VEGAMAG': # if this takes forever, just call the vega func once outside the loop
            
                res_before = -2.5 * np.log10(val_before.value[ii] / vega_val.value[ii])
                res_after = -2.5 * np.log10(val_after.value[ii] / vega_val.value[ii])
                
                
            if unit.upper() == 'ABMAG':
                
                res_before = -2.5 * np.log10(fnu_before.value[ii]) - 48.60
                res_after = -2.5 * np.log10(fnu_after.value[ii]) - 48.60
                
            
            res = res_after - res_before
                
            self._extinction.loc[b, 'extinction'] = res
                
                
        
        return self._extinction
    
    
    
    
    def interpolate_isochrone(self, idx):
        """
        Wrapper for interpolation functions so it can be pickled 
        properly with `multiprocessing.Pool`.
        
        Parameters
        ----------
        idx : list of floats
            The [age, mass] index to interpolate the model grid.
        
        Returns
        -------
        DataFrame
            The interpolated data point of the isochrone with `idx` as its index.
        
        """
        
        if self.model_grid is None:
            raise ValueError(
                "model grid is missing from class initialization"
                )
        
        return self._interp_func(idx, self.model_grid, **self._interp_kwargs)

    
    
    
    def photometry_model(self, age, mass, Av, Teff_prior=None, zero_extinction=False):
        """
        Combines the estimated photometry from the isochrone with the calculated extinction
        to form a model which can be used as a fit against the measured photometry.

        Parameters
        ----------
        age : float
            The age in Myr of the target.
        mass : float
            The mass in M_Sun of the target.
        Av : float
            The V-band extinction in mag.
        Teff_prior : tuple, optional
            The (mean, std) pair used to calculate the logarithm of the effective temperature 
            prior using a normal distribution. The default is None.
        zero_extinction : bool, optional
            If `True`, set extinction to zero (Av=0). The default is `False`.

        Returns
        -------
        model : DataFrame
            Contains the interpolated magnitudes, the calculated extinction (if not set to zero),
            and the sum of these for each band.
        log_teff_prior : float
            The effective temperature prior to be added to the probability functions.
            If `Teff_prior` is `None` or contains NaN values, `log_teff_prior` will be 0.

        """
    
        model = pd.DataFrame(index=self._photometry.index.values)
        
        interpolated_data = self.interpolate_isochrone([age, mass])
        
        Teff = interpolated_data.loc[(age, mass), 'Teff']
        
        if np.isnan(Teff):
            return False, Teff_prior
        
        if Teff_prior:
            if not np.nan in Teff_prior:
                log_teff_prior = log_normal(Teff, Teff_prior[0], Teff_prior[1])
            else:
                log_teff_prior = 0
        else:
            log_teff_prior = 0
        
        
        interpolated_abs_magnitudes = interpolated_data.loc[(age, mass), self._photometry['isochrone_analog']]
        
        model.loc[:, 'ABSOLUTE_MAGNITUDE'] = interpolated_abs_magnitudes.values
        
        
        # Av = 0;
        # magnitudes aren't actually 'corrected', but leaving this column name as is for continuity
        if zero_extinction:
            
            model['CORRECTED_MAGNITUDE'] = model['ABSOLUTE_MAGNITUDE']
            
            return model, log_teff_prior
        
        
        extinction = self.calculate_extinction(Av, Teff, **self._extinction_kwargs)
        
        model.loc[:, ['extinction', 'flux_unit']] = extinction
        
        model['CORRECTED_MAGNITUDE'] = model['ABSOLUTE_MAGNITUDE'] + model['extinction']
    
        return model, log_teff_prior
    
    
    
    
    
    
    
    
    

    










    
    
    
