#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:09:39 2020

@author: mjfields
"""

__all__ = ['Estimate']


import numpy as np
import pandas as pd
import time
from datetime import datetime

from multiprocessing import Pool
from tqdm import tqdm

from dsa.config import MAGMODELPATH, INTERPMAGMODELPATH, STDMODELPATH, INTERPSTDMODELPATH, GRIDCACHE
from dsa.radius.photometry import MeasuredPhotometry, SyntheticPhotometry
from dsa.radius.simulation import Probability, MCMC
from dsa.radius.target import Target
from dsa.radius.metadata import InitialConditions, Moves, PhotometryMetadata, DSADataFrame
from dsa.utils import app_mag, app_mag_error, mag_to_flux, load_isochrone, WaitingAnimation, sigma


pd.options.display.float_format = '{:.6g}'.format




class Estimate(object):
    
    """
    Automates the data gathering and mcmc simulation to estimate the fit parameters.
    
    Parameters
    ----------
    target : str or `Target` object
        The target whose parameters are estimated via MCMC simulation.
    isochrone : str, optional
        If 'mag', uses the isochrone that incorporates the effects of magnetic fields
        (better for young stars). If 'std', uses the standard, non-magnetic isochrone.
        The default is 'mag'.
    interp_method : str, optional
        If 'true', uses the standard interpolation method of DFInterpolator.
        If 'nearest' uses nearest-neighbor interpolation.
    use_synphot : bool, optional
        Use the built-in synphot methods to calculate extinction or calculate 
        extinction with `numpy` arrays which is faster. The default is False.
    zero_extinction : bool, optional
        If `True`, set extinction to zero (Av=0). The default is `False`.
    meas_phot_kwargs : dict, optional
        Keyword arguments to pass to `:class: dsa.radius.MeasuredPhotometry`.
        The default is `None`.
    
    """
    
    def __init__(self, target, isochrone='mag', interp_method='true', use_synphot=False, zero_extinction=False, meas_phot_kwargs=None):
        
        ## setup target-specific metadata
        
        # check if target is `Target` object or just string
        if isinstance(target, Target):
            self.target = target.name
            self.coords = target.coords
            
            self._ic = target.initial_conditions.loc[self.target].copy()
            self._moves = target.moves
            self._phot_meta = target.photometry_meta.copy()
            
        else:
            self.target = target
            self.coords = None
            
            self._ic = InitialConditions().initial_conditions.copy()
            self._moves = Moves().moves
            self._phot_meta = PhotometryMetadata().photometry.copy()
            
            
        ## select the appropriate isochrone model
        
        self._isochrone = isochrone
        self._interp_method = interp_method
        
        
        if self._isochrone.lower() == 'mag':
            
            if self._interp_method.lower() == 'true':
                gridpath = MAGMODELPATH
                
            if self._interp_method.lower() == 'nearest':
                gridpath = INTERPMAGMODELPATH
        
        elif self._isochrone.lower() == 'std':
            
            if self._interp_method.lower() == 'true':
                gridpath = STDMODELPATH
                
            if self._interp_method.lower() == 'nearest':
                gridpath = INTERPSTDMODELPATH
            
        else:
            raise ValueError(
                "invalid isochrone: can only be 'mag' or 'std'"
                )
        
        
        if self._interp_method.lower() == 'true':
            self._model_grid = load_isochrone(gridpath)
            
            self._agelist = None
            self._masslist = None
        
        if self._interp_method.lower() == 'nearest':
            with WaitingAnimation("loading isochrone model grid", delay=0.25):
                grid = load_isochrone(gridpath)
                print('')
            
            self._agelist = grid.index.get_level_values('age').drop_duplicates()
            self._masslist = grid.index.get_level_values('mass').drop_duplicates()
            
            grid.to_pickle(GRIDCACHE)
            
            del grid
            
            self._model_grid = GRIDCACHE
            
            
        ## check if synphot is going to be used for the extinction calculation
        
        self._use_synphot = use_synphot
        
        
        ## check if extinction is going to be set to zero
        
        self._zero_extinction = zero_extinction
        
        
        ## handle any kwargs for MeasuredPhotometry
        
        if meas_phot_kwargs is None:
            self._meas_phot_kwargs = dict()
        else:
            self._meas_phot_kwargs = meas_phot_kwargs
            
        
        ## collect data, initialize classes, and setup functions
        
        self._mp = MeasuredPhotometry(self.target, self.coords, photometry_meta=self._phot_meta, **self._meas_phot_kwargs)
        
        self.photometry, self._termination_message = self._mp.get_data()
        
        if self.photometry is False:
            self._sp, self._prob, self.log_prob_fn = False, False, False
            
        else:
            self._sp = SyntheticPhotometry(
                self.photometry, 
                model_grid=self._model_grid, 
                interp_method=self._interp_method,
                extinction_kwargs={'use_synphot':self._use_synphot}, 
                interp_kwargs={'agelist':self._agelist, 'masslist':self._masslist}
                )
            
            self._prob = Probability(self.photometry, self._sp.photometry_model, self._ic, zero_extinction=self._zero_extinction)
            
            self.log_prob_fn = self._prob.log_probability
        
        self._pool = Pool()
        
        
        ## metadata parameters for output
        
        self._run_date = None
        self._sim_runtime = None
        self._posterior_extract_time = None
        
        
        
        
    def run(self, nwalkers, nsteps, progress=True, verbose=True):
        """
        Wrapper for `dsa.radius.MCMC.run` which runs MCMC simulation using `emcee`.

        Parameters
        ----------
        nwalkers : int
            The number of independent walkers in the simulation chain.
        nsteps : int
            The number of iterations of the simulation.
        progress : bool, optional
            If `True`, provides a progress bar during the sumulation.  The default is `True`.
        verbose : bool, optional
            If `True`, uses print statements to indicate the current status of the simulation.
            The defauls is `True`.

        Returns
        -------
        sampler : EnsembleSampler
            `emcee.EnsembleSampler` object containing all estimated values and metadata from the simulation.

        """
        
        self._run_date = datetime.today().strftime('%Y%m%d')
        start = time.time()
        
        
        if self.photometry is False:
            print(self._termination_message)
            
            return False
        
        if verbose:
            print(f"\nrunning MCMC for {self.target:s}:")
            
        time.sleep(1)
        
        mcmc = MCMC(nwalkers, nsteps, self.log_prob_fn, self._ic, self._moves, pool=self._pool, zero_extinction=self._zero_extinction)
        
        mcmc.run(progress=True)
        
        time.sleep(1)
        
        sampler = mcmc.sampler
        
        
        stop = time.time()
        delta = stop-start
        self._sim_runtime = time.strftime('%H:%M:%S', time.gmtime(delta))
        
        
        return sampler
        
        
        
        
    def posterior(self, sampler, thin=1, discard=0, force_true_interp=False, verbose=True):
        """
        Calculates full posterior distributions for the fit parameters and others, including radius, Teff, and density. 
        Interpolates estimated magnitudes from age and mass obtained from fit.
        See https://emcee.readthedocs.io/en/stable/tutorials/line/ for more general information.

        Parameters
        ----------
        sampler : EnsembleSampler
            `emcee.EnsembleSampler` object containing all estimated values and metadata from the simulation.
        thin : int, optional
            Use every `thin` values of the posterior. The defualt is 1.
        discard : int, optional
            Remove (burnin) the first `discard` elements from the posterior.
            The defult is 0.
        force_true_interp : bool, optional
            If `True`, the non-fit chains are interpolated using the 'true' interpolation
            method. If `False` (default), uses the same interpolation method as the
            MCMC simulation.
        verbose : bool, optional
            If `True`, uses print statements to indicate the current status of the simulation.
            The defauls is `True`.

        Returns
        -------
        posterior : DSADataFrame
            The estimated fit parameters and other stellar parameters, including uncertainties.
        photometry : DSADataFrame
            The measured and estimated magnitudes and other photometric data.
        posterior_chains : DSADataFrame
            The flattened lists of estimated or interpolated values of each parameter (including non-fit parameters) 
            at every step of the simulation (i.e., the posterior distributions).

        """
        
        start = time.time()
        
        
        if sampler is False:
            return False, False, False
        
        if verbose:
            print(f"\nextracting posterior for {self.target:s}:")
        
        samples = sampler.get_chain
            
            
        try:
            flat_samples = samples(discard=discard, thin=thin, flat=True)
        except ValueError:
            flat_samples = samples(flat=True)
        
        
        if verbose:
            print("\ncalculating max log probability:")
        
        log_prob, max_prob_index = self._max_log_probability(flat_samples)
        
        
        if self._zero_extinction:
            params = ['age', 'mass', 'f', 'radius', 'Teff', 'density']
            posterior_chains = pd.DataFrame(flat_samples, columns=['age', 'mass', 'f'])
            
        else:
            params = ['age', 'mass', 'Av', 'f', 'radius', 'Teff', 'density']
            posterior_chains = pd.DataFrame(flat_samples, columns=['age', 'mass', 'Av', 'f'])
        
        
        if verbose:
            print("\ngetting radius and Teff chains:")
            
            
            
        if force_true_interp:
            if self._interp_method == 'true':
                sp = self._sp
            
            else:
                if self._isochrone.lower() == 'mag':
                    grid = load_isochrone(MAGMODELPATH)
                    
                if self._isochrone.lower() == 'std':
                    grid = load_isochrone(STDMODELPATH)
                    
                sp = SyntheticPhotometry(
                    self.photometry,
                    model_grid=grid,
                    interp_method='true'
                    )
                
        else:
            sp = self._sp
        
        
        # try to use pool to parallelize this if possible
        if self._pool is  None:
            
            posterior_chains[['radius', 'Teff']] = pd.concat(
                [sp.interpolate_isochrone((posterior_chains['age'][i], posterior_chains['mass'][i])).loc[(posterior_chains['age'][i], posterior_chains['mass'][i]), ['radius', 'Teff']]
                  for i in tqdm(range(len(flat_samples)))],
                ignore_index=True)
        else:
            
            map_func = self._pool.imap
        
            time.sleep(1)
            
            posterior_chains[['radius', 'Teff']] = pd.concat(
                list(
                    res.loc[:, ['radius', 'Teff']] for res in tqdm(
                        map_func(
                            sp.interpolate_isochrone, 
                            ((posterior_chains['age'][i], posterior_chains['mass'][i]) for i in range(len(posterior_chains)))
                            ), total=len(posterior_chains)
                        )
                    ), 
                ignore_index=True
                )
    
            time.sleep(1)
        
    
        posterior_chains['density'] = posterior_chains['mass'] / (posterior_chains['radius']**3)
        
        posterior = pd.DataFrame(index=params)
        
        # calculate the median value (50th percentile), and upper and lower confidence 
        # (84th and 16th percentiles) for each parameter
        for p in params:
            mc = np.nanpercentile(posterior_chains[p], [16, 50, 84])
            q = np.diff(mc)
            
            posterior.loc[p, 'median'] = mc[1]
            posterior.loc[p, 'max_probability'] = posterior_chains.loc[max_prob_index, p]
            posterior.loc[p, 'uncertainty'] = np.mean([q[0], q[1]])
            posterior.loc[p, '+'] = q[1]
            posterior.loc[p, '-'] = q[0]
            
        posterior.index.names = ['parameter']
            
            
        if self._zero_extinction:
            # value of Av here doesn't matter as long as `zero_extinction=True`
            median_photometry_model, teff_lp = self._sp.photometry_model(posterior.loc['age', 'median'], posterior.loc['mass', 'median'], 0, zero_extinction=self._zero_extinction)
            max_prob_photometry_model, teff_lp = self._sp.photometry_model(posterior.loc['age', 'max_probability'], posterior.loc['mass', 'max_probability'], 0, zero_extinction=self._zero_extinction)
            
        else:
            median_photometry_model, teff_lp = self._sp.photometry_model(posterior.loc['age', 'median'], posterior.loc['mass', 'median'], posterior.loc['Av', 'median'])
            max_prob_photometry_model, teff_lp = self._sp.photometry_model(posterior.loc['age', 'max_probability'], posterior.loc['mass', 'max_probability'], posterior.loc['Av', 'max_probability'])
        
        
        photometry = self.photometry
        
        med_f_abs = posterior.loc['f', 'median']
        max_f_abs = posterior.loc['f', 'max_probability']
        
        
        if median_photometry_model is not False:
                    
            photometry['MEDIAN_ABSOLUTE_MAGNITUDE'] = median_photometry_model.loc[:, 'CORRECTED_MAGNITUDE']
            photometry['MEDIAN_ABSOLUTE_MAGNITUDE_ERROR'] = photometry.loc[:, 'ABSOLUTE_MAGNITUDE_ERROR'].apply(sigma, args=([med_f_abs]))
            
            photometry['median_apparent_magnitude'] = photometry.loc[:, 'MEDIAN_ABSOLUTE_MAGNITUDE'].apply(app_mag, args=([photometry['parallax'][0]]))
            photometry['median_apparent_magnitude_error'] = photometry.loc[:, 'MEDIAN_ABSOLUTE_MAGNITUDE_ERROR'].apply(app_mag_error, args=([photometry['parallax'][0], photometry['parallax_error'][0]]))
        
            
            for band in photometry.index:
                photometry.loc[band, ['median_flux', 'median_flux_error']] = mag_to_flux(*photometry.loc[band, ['median_apparent_magnitude', 'zeropoint_flux', 'median_apparent_magnitude_error']])
        
        
            photometry['median_percent_error'] = 100 * np.abs((photometry['flux'] - photometry['median_flux']) / photometry['flux'])
        
        
        if max_prob_photometry_model is not False:
        
            photometry['MAX_PROBABILITY_ABSOLUTE_MAGNITUDE'] = max_prob_photometry_model.loc[:, 'CORRECTED_MAGNITUDE']
            photometry['MAX_PROBABILITY_ABSOLUTE_MAGNITUDE_ERROR'] = photometry.loc[:, 'ABSOLUTE_MAGNITUDE_ERROR'].apply(sigma, args=([max_f_abs]))
            
            photometry['max_probability_apparent_magnitude'] = photometry.loc[:, 'MAX_PROBABILITY_ABSOLUTE_MAGNITUDE'].apply(app_mag, args=([photometry['parallax'][0]]))
            photometry['max_probability_apparent_magnitude_error'] = photometry.loc[:, 'MAX_PROBABILITY_ABSOLUTE_MAGNITUDE_ERROR'].apply(app_mag_error, args=([photometry['parallax'][0], photometry['parallax_error'][0]]))
        
            
            for band in photometry.index:
                photometry.loc[band, ['max_probability_flux', 'max_probability_flux_error']] = mag_to_flux(*photometry.loc[band, ['max_probability_apparent_magnitude', 'zeropoint_flux', 'max_probability_apparent_magnitude_error']])
        
        
            photometry['max_probability_percent_error'] = 100 * np.abs((photometry['flux'] - photometry['max_probability_flux']) / photometry['flux'])
        
            
        # percent errors should be the same between apparent and ABSOLUTE
        
        photometry.index.names = ['band']
        
        
        ## apply metadata
        
        stop = time.time()
        delta = stop-start
        self._posterior_extract_time = time.strftime('%H:%M:%S', time.gmtime(delta))
        
        
        metadata = {
            'target' : self.target,
            'coordinates' : self.coords,
            'isochrone' : self._isochrone,
            'interp_method' : self._interp_method,
            'use_synphot' : self._use_synphot,
            'zero_extinction' : self._zero_extinction,
            'force_posterior_interp_true' : force_true_interp,
            'nwalkers' : sampler.nwalkers,
            'nsteps' : len(samples()),
            'discard' : discard,
            'thin' : thin,
            'mean_acceptance_frac' : float(f'{np.mean(sampler.acceptance_fraction):.3f}'),
            'median_autocorr_time' : float(f'{np.median(sampler.get_autocorr_time(tol=0)):.3f}'),
            'date' : self._run_date,
            'sim_runtime' : self._sim_runtime,
            'posterior_extract_time' : self._posterior_extract_time
            }
        
        posterior = DSADataFrame(posterior.copy(), meta_base_type='posterior', metadata=metadata)
        photometry = DSADataFrame(photometry.copy(), meta_base_type='photometry', metadata=metadata)
        posterior_chains = DSADataFrame(posterior_chains.copy(), meta_base_type='chains', metadata=metadata)
        
        
        return posterior, photometry, posterior_chains
        
        
        
        
    def _max_log_probability(self, coords, progress=True):
        """
        Returns the maximum of the caluclated log probabilities and its index. 
        Simplified version of :func: `emcee.EnsembleSampler.sampler.compute_log_prob`
        for this use case (no need for blobs).
        
        Parameters
        ----------
        coords : numpy.ndarray
            The position matrix in parameter space for each fit parameter.
        progress : bool, optional
            If `True`, provides a progress bar during the calculation. The default is True.
            
        Returns
        -------
        log_prob : array
            The list of calculated log-probability for each coordinate.
        max_log_prob_index : int
            The index where `log_prob` is maximized.
        
        """
        
        p = coords
        
        
        if progress and self._pool is not None:
            map_func = self._pool.imap
            # imap gives tqdm an iterable
        elif not progress and self._pool is not None:
            map_func = self._pool.map
        else:
            map_func = map
            
            
        if progress:
            time.sleep(1)
            
            results = list(res for res in tqdm(map_func(self.log_prob_fn, [r for r in p]), total=len(p)))
            
            time.sleep(1)
        else:
            results = list(map_func(self.log_prob_fn, (p[i] for i in range(len(p)))))
            
        log_prob = np.array([float(l) for l in results])
        
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")
            
           
        
        max_log_prob_index = np.argmax(log_prob)
        
        
        return log_prob, max_log_prob_index
    
    
    
    
    def _get_log_likelihoods(self, coords, progress=True):
        """
        Returns a distribution of log-likelihood values from a given simulation.
        Calculates the likelihoods in an analogous way to `self._max_log_probability`.
        
        Parameters
        ----------
        coords : numpy.ndarray
            The position matrix in parameter space for each fit parameter.
        progress : bool, optional
            If `True`, provides a progress bar during the calculation. The default is True.
            
        Returns
        -------
        log_likelihood : array
            The list of calculated log-likelihood for each coordinate.
        
        """
        
        p = coords
        
        
        if progress and self._pool is not None:
            map_func = self._pool.imap
            # imap gives tqdm an iterable
        elif not progress and self._pool is not None:
            map_func = self._pool.map
        else:
            map_func = map
            
            
        if progress:
            time.sleep(1)
            
            results = list(res for res in tqdm(map_func(self._prob.log_likelihood, [r for r in p]), total=len(p)))
            
            time.sleep(1)
        else:
            results = list(map_func(self._prob.log_likelihood, (p[i] for i in range(len(p)))))
            
        log_likelihood = np.array([float(l) for l in results])
        
        if np.any(np.isnan(log_likelihood)):
            raise ValueError("Likelihood function returned NaN")
            
           
        
        return log_likelihood









