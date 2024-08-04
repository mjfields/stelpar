# -*- coding: utf-8 -*-




import pandas as pd
import numpy as np

import warnings

from emcee.moves import StretchMove, DEMove, DESnookerMove, KDEMove
from astropy.table import Table




__all__ = ['InitialConditions', 'Moves', 'PhotometryMetadata', 'MetaDataFrame']




class InitialConditions(object):
    
    """
    The backend framework to access and manipulate the initial conditions and relevent
    metadata for the simulation.
    
    """
    
    def __init__(self):
        
        self._default_initial_conditions = pd.DataFrame(
            index=['age', 'mass', 'Av', 'f', 'Teff'],
            columns=['bounds', 'prior', 'position', 'perturbation'],
            dtype=object
            )
        
        self._default_initial_conditions.loc['age', 'bounds'] = (1.0, 10000.0)
        self._default_initial_conditions.loc['mass', 'bounds'] = (0.09, 2.45)
        self._default_initial_conditions.loc['Av', 'bounds'] = (0.0, 3.0)
        self._default_initial_conditions.loc['f', 'bounds'] = (0.0, 2.0)
        self._default_initial_conditions.loc['Teff', 'bounds'] = (np.nan, np.nan)
        
        self._default_initial_conditions.loc[:, 'prior'] = [(np.nan, np.nan)]*5
        
        self._default_initial_conditions.loc['age', 'position'] = 700.0
        self._default_initial_conditions.loc['mass', 'position'] = 1.0
        self._default_initial_conditions.loc['Av', 'position'] = 0.0
        self._default_initial_conditions.loc['f', 'position'] = 0.0
        
        self._default_initial_conditions.loc['age', 'perturbation'] = 35.0
        self._default_initial_conditions.loc['mass', 'perturbation'] = 0.2
        self._default_initial_conditions.loc['Av', 'perturbation'] = 0.2
        self._default_initial_conditions.loc['f', 'perturbation'] = 0.2
        
        self.initial_conditions = self._default_initial_conditions.copy()
        
        
        
        
    @property
    def bounds(self):
        
        return self.initial_conditions['bounds']
    
    
    
    
    @bounds.setter
    def bounds(self, bounds):
        
        if bounds is None:
            return
        
        # put the tuple into a list so pandas keeps it as 1 element
        for key in bounds:
            bounds.update({key:[bounds[key]]})
            
        bounds_df = pd.DataFrame.from_dict(bounds, orient='index', columns=['bounds'], dtype=object)
        
        self.initial_conditions.update(bounds_df)
           
        
        
        
    @property
    def perturbation(self):
        
        return self.initial_conditions['perturbation']
    
    
    
    
    @perturbation.setter
    def perturbation(self, perturbation):
        
        if perturbation is None:
            return
        
        # put the value into a list so pandas keeps it as 1 element
        for key in perturbation:
            perturbation.update({key:[perturbation[key]]})
            
        perturbation_df = pd.DataFrame.from_dict(perturbation, orient='index', columns=['perturbation'], dtype=object)
        
        self.initial_conditions.update(perturbation_df)
        
    
    
    
    @property
    def prior(self):
        
        return self.initial_conditions['prior']
    
    
    
    
    @prior.setter
    def prior(self, prior):
        
        if prior is None:
            return
        
        # put the tuple into a list so pandas keeps it as 1 element
        for key in prior:
            prior.update({key:[prior[key]]})
            
        prior_df = pd.DataFrame.from_dict(prior, orient='index', columns=['prior'], dtype=object)
        
        self.initial_conditions.update(prior_df)
        
            
            
            
    @property
    def position(self):
        
        return self.initial_conditions['position']
    
    
    
    
    @position.setter
    def position(self, position):
        
        if position is None:
            return
        
        # put the value into a list so pandas keeps it as 1 element
        for key in position:
            position.update({key:[position[key]]})
            
        position_df = pd.DataFrame.from_dict(position, orient='index', columns=['position'], dtype=object)
        
        self.initial_conditions.update(position_df)
        
        
        
        
        
    def reset(self, params=None, conds=None):
        """
        Resets the specified initial conditions to default.

        Parameters
        ----------
        params : list of str, optional
            The list of parameters whose initial conditions are to be reset to default. 
            If None, all parameters will be reset. The default is None.
        conds : list of str, optional
            The list of conditions which are to be reset to default. 
            If None, all conditions will be reset. The default is None.

        """
        
        if params is None:
            
            params = self._default_initial_conditions.index.values
            
        if conds is None:
            
            conds = self._default_initial_conditions.columns.values
        
        
        new_conds = self._default_initial_conditions.loc[params, conds].copy()
        
        self.initial_conditions.update(new_conds)
            
            
            
            
class Moves(object):
    
    """
    The backend framework to access and manipulate the ensemble moves for the simulation. 
    See https://emcee.readthedocs.io/en/stable/user/moves/ for more information.
    
    """
    
    def __init__(self):
        
        self._default_moves = [(StretchMove(), 1.0)]
        self._moves = self._default_moves
        
        
        
        
    @property
    def moves(self):
        
        return self._moves
    
    
    
    
    @moves.setter
    def moves(self, moves):
        
        weights = [move[1] for move in moves]
        
        if np.sum(weights) != 1.0:
            raise ValueError(
                "The sum of the weights must equal 1.0."
                )
            
        move_list = []
        for move in moves:
            
            m, w, *args = move
            
            
            if m.upper() == 'STRETCH':
                
                move_list.append((StretchMove(*args), w))
                
            if m.upper() == 'DE':
                
                move_list.append((DEMove(*args), w))
                
            if m.upper() == 'SNOOKER':
                
                move_list.append((DESnookerMove(*args), w))
                
            if m.upper() == 'KDE':
                
                move_list.append((KDEMove(*args), w))
                
                
        self._moves = move_list
        
        
        
        
    def reset(self):
        
        self._moves = self._default_moves
        
        
        
        
class PhotometryMetadata(object):
    
    """
    The backend framework to access and manipulate the photometry used to estimate
    the stellar parameters.
    
    """
    
    def __init__(self):
        
        self._default_photometry = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [
                    ('II/246/out', '2mass_jmag'),
                    ('II/246/out', '2mass_hmag'),
                    ('II/246/out', '2mass_kmag'),
                    
                    ('I/355/gaiadr3', 'gaia_gmag'),
                    ('I/355/gaiadr3', 'gaia_bpmag'),
                    ('I/355/gaiadr3', 'gaia_rpmag'),
                    
                    ('V/154/sdss16', 'sdss_gmag'),
                    ('V/154/sdss16', 'sdss_rmag'),
                    ('V/154/sdss16', 'sdss_imag'),
                    ('V/154/sdss16', 'sdss_zmag'),
                    
                    ('II/336/apass9', 'johnson_bmag'),
                    ('II/336/apass9', 'johnson_vmag'),
                    
                    ('I/259/tyc2', 'tycho_bmag'),
                    ('I/259/tyc2', 'tycho_vmag'),
                    
                    ('I/239/hip_main', 'hipparcos_hpmag'),
                    ],
                names=[
                    'catalog',
                    'band'
                    ]
                ),
            columns=[
                'magnitude',
                'error',
                'system',
                'isochrone_analog'
                ]
            )
        
        self._default_photometry.loc[('II/246/out', '2mass_jmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'Jmag', 'e_Jmag', 'VEGA', '2mass_jmag'
        self._default_photometry.loc[('II/246/out', '2mass_hmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'Hmag', 'e_Hmag', 'VEGA', '2mass_hmag'
        self._default_photometry.loc[('II/246/out', '2mass_kmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'Kmag', 'e_Kmag', 'VEGA', '2mass_kmag'
        
        self._default_photometry.loc[('I/355/gaiadr3', 'gaia_gmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'Gmag', 'e_Gmag', 'VEGA', 'gaia_gmag'
        self._default_photometry.loc[('I/355/gaiadr3', 'gaia_bpmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'BPmag', 'e_BPmag', 'VEGA', 'gaia_bpmag'
        self._default_photometry.loc[('I/355/gaiadr3', 'gaia_rpmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'RPmag', 'e_RPmag', 'VEGA', 'gaia_rpmag'
        
        self._default_photometry.loc[('V/154/sdss16', 'sdss_gmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'gmag', 'e_gmag', 'AB', 'sdss_gmag'
        self._default_photometry.loc[('V/154/sdss16', 'sdss_rmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'rmag', 'e_rmag', 'AB', 'sdss_rmag'
        self._default_photometry.loc[('V/154/sdss16', 'sdss_imag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'imag', 'e_imag', 'AB', 'sdss_imag'
        self._default_photometry.loc[('V/154/sdss16', 'sdss_zmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'zmag', 'e_zmag', 'AB', 'sdss_zmag'
        
        self._default_photometry.loc[('II/336/apass9', 'johnson_bmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'Bmag', 'e_Bmag', 'VEGA', 'johnson_bmag'
        self._default_photometry.loc[('II/336/apass9', 'johnson_vmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'Vmag', 'e_Vmag', 'VEGA', 'johnson_vmag'
        
        self._default_photometry.loc[('I/259/tyc2', 'tycho_bmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'BTmag', 'e_BTmag', 'VEGA', 'tycho_bmag'
        self._default_photometry.loc[('I/259/tyc2', 'tycho_vmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'VTmag', 'e_VTmag', 'VEGA', 'tycho_vmag'
        
        self._default_photometry.loc[('I/239/hip_main', 'hipparcos_hpmag'), ('magnitude', 'error', 'system', 'isochrone_analog')] = 'Hpmag', 'e_Hpmag', 'VEGA', 'hipparcos_hpmag'
        
        
        self._photometry = self._default_photometry.copy()
        
        self._isochrone_analogs = [
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
        
        
        
        
    @property
    def isochrone_analogs(self):
        
        analogs = self._isochrone_analogs
        
        analogs.sort()
        
        return analogs
            
            
            
            
    @property
    def photometry(self):
        
        return self._photometry
    
    
    
    
    def add(self, photometry_dict):
        
        for catalog in photometry_dict:
            
            for band in photometry_dict[catalog]:
                
                meta = photometry_dict[catalog][band]
                
                if len(meta) == 3:
                    
                    meta = list(meta)
                    meta.append(band)
                    meta = tuple(meta)
                    
                if len(meta) < 3:
                    
                    raise ValueError(
                        f"photometry_dict['{catalog}']['{band}'] must be a tuple with at least (mag_name, error_name, system)"
                        )
                    
                if catalog.lower() == 'local':
                    
                    if not (type(meta[0]) is float or type(meta[0]) is int) or not (type(meta[1]) is float or type(meta[1]) is int):
                        
                        raise TypeError(
                            "mag and error must be floats if passing local photometry"
                            )
                
                self._photometry.loc[(catalog, band), ('magnitude', 'error', 'system', 'isochrone_analog')] = meta
                
                
                
                
    def remove(self, photometry_dict):
        
        for catalog in photometry_dict:
            
            meta = photometry_dict[catalog]
                
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
            
                if type(meta) is str:
                    
                    if meta.lower() == 'all':
                        
                        self._photometry.drop(index=[catalog], inplace=True)
                        
                    else:
                        
                        self._photometry.drop(index=[(catalog, meta)], inplace=True)
                    
                elif type(meta) is list or type(meta) is tuple:
                    
                    for m in meta:
                        
                        self._photometry.drop(index=[(catalog, m)], inplace=True)
                                    
                                    
                                    
                                    
    def reset(self):
        
        self._photometry = self._default_photometry.copy()
        
        
        
        
class MetaDataFrame(pd.DataFrame):
    
    """
    Subclass of `pandas.DataFrame` that adds two arguments, meta_base_type and metadata,
    and gives the ability to directly convert to `astropy.table.Table` with relevant
    units, descriptions, and other metadata.
    
    Parameters
    ----------
    args, kwargs
        All positional and keyword arguments passable to a `DataFrame`. See
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
        for more information.
    meta_base_type : str, optional
        Which of 'posterior', 'photometry', or 'chains' is being made a `MetaDataFrame`.
        Adds specific metadata for those three options. If `None` (default), no
        metadata is assigned.
    metadata : dict, optional
        Additional metadata assigned to the `MetaDataFrame`. The default is `None`.
    
    """
    
    def __init__(self, *args, meta_base_type=None, metadata=None, **kwargs):
        
        super(MetaDataFrame, self).__init__(*args, **kwargs)
        
        
        if meta_base_type is not None:
            
            if type(meta_base_type) is not str:
                raise TypeError(
                    "`meta_base_type` must be given as a string"
                    )
            
        self._meta_base_type = meta_base_type
        
        
        if metadata is not None:
            
            if type(metadata) is not dict:
                raise TypeError(
                    '`metadata` must be given as a dictionary'
                    )
        
        self.metadata = metadata
        
        
        
        
    # taken directly from geopandas b/c they encounter same issue
    def __setattr__(self, attr, val):
        
        # have to special case metadata b/c pandas tries to use as column...
        
        if attr == "metadata":
            object.__setattr__(self, attr, val)
            
        else:
            super(MetaDataFrame, self).__setattr__(attr, val)
        
        
        
        
    @property
    def _constructor(self):
        
        return MetaDataFrame
        
        
        
        
        
    @property
    def _photometry_metadata(self):
        
        dict_ = {
            'band' : {
                'description' : 'band-pass filter label'
                }, 
            'apparent_magnitude' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'measured apparent magnitude'
                },
            'apparent_magnitude_error' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'measured apparent magnitude uncertainty'
                },
            'ABSOLUTE_MAGNITUDE' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'measured absolute magnitude converted from apparent magnitude'
                },
            'ABSOLUTE_MAGNITUDE_ERROR' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'measured absolute magnitude uncertainty converted from apparent magnitude uncertainty'
                },
            'flux' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'erg cm^-2 s^-1 angstrom^-1',
                'description' : 'measured flux density converted from apparent magnitude'
                },
            'flux_error' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'erg cm^-2 s^-1 angstrom^-1',
                'description' : 'measured flux density uncertainty converted from apparent magnitude uncertainty'
                },
            'parallax' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'arcsec',
                'description' : 'measured parallax'
                },
            'parallax_error' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'arcsec',
                'description' : 'measured parallax uncertainty'
                },
            'RUWE' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : '',
                'description' : 'renormalized unit weight error'
                },
            'zeropoint_flux' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'erg cm^-2 s^-1 angstrom^-1',
                'description' : 'zeropoint flux taken from the SVO Filter Profile Service'
                },
            'wavelength' : {
                'dtype' : 'float64',
                'format' : '.5g',
                'unit' : 'angstrom',
                'description' : 'effective wavelength taken from the SVO Filter Profile Service'
                },
            'system' : {
                'description' : 'band-pass filter magnitude system'
                },
            'isochrone_analog' : {
                'description' : 'which band-pass filter from the isochrone model matches this one'
                },
            'MEDIAN_ABSOLUTE_MAGNITUDE' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'median absolute magnitude from the estimated fit'
                },
            'MEDIAN_ABSOLUTE_MAGNITUDE_ERROR' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'median absolute magnitude uncertainty from the estimated fit'
                },
            'median_apparent_magnitude' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'median apparent magnitude converted from estimated median absolute magnitude'
                },
            'median_apparent_magnitude_error' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : ' median apparent magnitude uncertainty converted from estimated median absolute magnitude uncertainty'
                },
            'median_flux' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'erg cm^-2 s^-1 angstrom^-1',
                'description' : 'median flux density converted from estimated median apparent magnitude'
                },
            'median_flux_error' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'erg cm^-2 s^-1 angstrom^-1',
                'description' : 'median flux density uncertainty converted from estimated median apparent magnitude uncertainty'
                },
            'median_percent_error' : {
                'dtype' : 'float64',
                'format' : '.2f',
                'unit' : r'%',
                'description' : 'percent error between estimated median flux and measured flux'
                },
            'MAX_PROBABILITY_ABSOLUTE_MAGNITUDE' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'max-probability absolute magnitude from the estimated fit'
                },
            'MAX_PROBABILITY_ABSOLUTE_MAGNITUDE_ERROR' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'max-probability absolute magnitude uncertainty from the estimated fit'
                },
            'max_probability_apparent_magnitude' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'max-probability apparent magnitude converted from estimated max-probability absolute magnitude'
                },
            'max_probability_apparent_magnitude_error' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : ' max-probability apparent magnitude uncertainty converted from estimated max-probability absolute magnitude uncertainty'
                },
            'max_probability_flux' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'erg cm^-2 s^-1 angstrom^-1',
                'description' : 'max-probability flux density converted from estimated max-probability apparent magnitude'
                },
            'max_probability_flux_error' : {
                'dtype' : 'float64',
                'format' : '.5e',
                'unit' : 'erg cm^-2 s^-1 angstrom^-1',
                'description' : 'max-probability flux density uncertainty converted from estimated max-probability apparent magnitude uncertainty'
                },
            'max_probability_percent_error' : {
                'dtype' : 'float64',
                'format' : '.2f',
                'unit' : r'%',
                'description' : 'percent error between estimated max-probability flux and measured flux'
                }
            }
        
        return dict_
    
    
    
    
    @property
    def _posterior_metadata(self):
        
        dict_ = {
            'quoted_value' : {
                'description' : 'median: 50th percentile; max_probability: maximum value calculated from the log-probability function; uncertainty: average of upper and lower confidence intervals;\
                    +: upper confidence interval calculated as difference between 84th and 50th percentiles; -: lower confidence interval calculated as difference between 50th and 16th percentiles'
                }, 
            'age' : {
                'dtype' : 'float64',
                'format' : '.6g',
                'unit' : 'Myr',
                'description' : 'estimated stellar age'
                },
            'mass' : {
                'dtype' : 'float64',
                'format' : '.6g',
                'unit' : 'Msun',
                'description' : 'estimated stellar mass'
                },
            'Av' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'estimated V-band extinction'
                },
            'f' : {
                'dtype' : 'float64',
                'format' : '.5f',
                'unit' : 'mag',
                'description' : 'underestimation of measured magnitude uncertainties'
                },
            'Teff' : {
                'dtype' : 'float64',
                'format' : '.6g',
                'unit' : 'K',
                'description' : 'estimated stellar effective temperature'
                },
            'logL' : {
                'dtype' : 'float64',
                'format' : '.6g',
                'unit' : 'log(Lsun)',
                'description' : 'estimated stellar luminosity'
                },
            'logg' : {
                'dtype' : 'float64',
                'format' : '.6g',
                'unit' : 'log(cm s^-2)',
                'description' : 'estimated stellar surface gravity'
                },
            'radius' : {
                'dtype' : 'float64',
                'format' : '.6g',
                'unit' : 'Rsun',
                'description' : 'estimated stellar radius'
                },
            'density' : {
                'dtype' : 'float64',
                'format' : '.6g',
                'unit' : 'Msun Rsun^-3',
                'description' : 'estimated stellar density calculated as mass divided by cubed-radius'
                }
            }
        
        return dict_
    
    
    
    
    @property
    def _chain_metadata(self):
        
        dict_ = self._posterior_metadata
        
        dict_.pop('quoted_value')
        
        return dict_
    
    
    
    
    def to_astropy(self):
        
        """
        Convert to `astropy.table.Table` and applies metadata if available.
        See https://docs.astropy.org/en/stable/table/pandas.html for more information.
        
        """
        
        if self._meta_base_type.lower() == 'photometry':
            
            column_meta = self._photometry_metadata
            name = 'photometry'
            index_name = 'band'
            description = 'measured and estimated fit photometry'
            data = self.reset_index()
            
        elif self._meta_base_type.lower() == 'posterior':
            
            column_meta = self._posterior_metadata
            name = 'posterior'
            index_name = 'quoted_value'
            description = 'estimated stellar parameters'
            data = self.T.reset_index().rename(columns={'index':index_name})
            
        elif self._meta_base_type.lower() == 'chains':
            
            column_meta = self._chain_metadata
            name = 'chains'
            index_name = None
            description = 'posterior distributions for estimated stellar parameters'
            data = self
            
        else:
            
            column_meta = None
            name = None
            index_name = None
            description = None
            data = self
        
        
        table_meta = self.metadata
        
        
        table = Table.from_pandas(data)
        
        if index_name is not None:
            
            table.add_index(index_name)
        
        
        if column_meta is not None:
            
            for col in table.columns:
                
                col_data = table[col]
                col_meta = column_meta[col]
                
                if 'dtype' in col_meta.keys():
                    col_data.dtype = col_meta['dtype']
                    
                if 'format' in col_meta.keys():
                    col_data.format = col_meta['format']
                    
                if 'unit' in col_meta.keys():
                    col_data.unit = col_meta['unit']
                    
                if 'description' in col_meta.keys():
                    col_data.description = col_meta['description']
        
        
        if name is not None:
            
            table.name = name
        
        
        if description is not None:
            
            table.description = description
            
            
        if table_meta is not None:
            
            table.meta = table_meta
            
        
        return table
    
    
    
    

            
        
        
    
    
    
    
    
        
        
        
    
    
    
    
    

        
        
        
        
        
        
        
        
        
        
    