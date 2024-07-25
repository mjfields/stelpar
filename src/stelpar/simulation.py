# -*- coding: utf-8 -*-




import numpy as np
import emcee

import warnings

from .metadata import InitialConditions, Moves
from .utils import log_normal




__all__ = ['Probability', 'MCMC']




class Probability(object):
    
    """
    Bayesian statistical framework used by the MCMC simulation.
    
    Parameters
    ----------
    measured_photometry : DataFrame
        Contains the target's measured magnitudes and other photometry.
    phot_model_func : object
        The function that will be called to serve as the estimated photometry model.
    initial_conditions : DataFrame
        Contains the metadata defining boundary conditions, priors, and initial positions for the fit parameters.
    zero_extinction : bool, optional
        If `True`, set extinction to zero (Av=0). The default is `False`.
    
    """
    
    def __init__(self, measured_photometry, phot_model_func, initial_conditions=None, zero_extinction=False):
        
        self._measured_photometry = measured_photometry
        self._photometry_model = phot_model_func
        
        if initial_conditions is not None:
            self.initial_conditions = initial_conditions.copy()
            
        else:
            self.initial_conditions = InitialConditions().initial_conditions.copy()
            
        self._zero_extinction = zero_extinction
    
    
    
    
    def log_likelihood(self, theta):
        """
        Calculates the logarithm of the Bayesian probabilty.

        Parameters
        ----------
        theta : list or tuple
            The list of fit parameters.

        Returns
        -------
        float
            The log likelihood plus the log prior of the effective temperature.
            The latter is handled here instead of in `log_prior` because effective temperature is 
            not a fit parameter, and its prior acts as an additional prior on age and mass.

        """
        
        if self._zero_extinction:
            age, mass, f = theta
            # photometry_model won't care what Av is if zero_extinction=True
            Av = 0
            
        else:
            age, mass, Av, f = theta
        
        # Teff prior is handled here from the phot model func since it's not a fit parameter
        Teff_prior = self.initial_conditions.loc['Teff', 'prior']
        model, log_teff_prior = self._photometry_model(age, mass, Av, Teff_prior=Teff_prior, zero_extinction=self._zero_extinction)
        
        if model is False:
            self.log_teff_prior = 0
            
            return -np.inf
        
        self.log_teff_prior = log_teff_prior
        
        
        # suppress RuntimeWarnding
        # carried over from when previously taking log(f) so not sure if necessary
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            
            model['sigma2'] = self._measured_photometry['ABSOLUTE_MAGNITUDE_ERROR']**2 + f**2
        
        log_likelihood_series = ((self._measured_photometry['ABSOLUTE_MAGNITUDE'] - model['CORRECTED_MAGNITUDE'])**2 / model['sigma2']) + np.log(2*np.pi*model['sigma2'])
        
        
        return (-0.5 * log_likelihood_series.sum())
    
    
    
    
    def log_prior(self, theta):
        """
        Calculate the Bayesian log prior from the Gaussian prior distribution.

        Parameters
        ----------
        theta : list or tuple
            The list of fit parameters.

        Returns
        -------
        float
            The log prior.

        """
        
        amin = self.initial_conditions.loc['age', 'bounds'][0]
        amax = self.initial_conditions.loc['age', 'bounds'][1]
        
        mmin = self.initial_conditions.loc['mass', 'bounds'][0]
        mmax = self.initial_conditions.loc['mass', 'bounds'][1]
        
        avmin = self.initial_conditions.loc['Av', 'bounds'][0]
        avmax = self.initial_conditions.loc['Av', 'bounds'][1]
        
        fmin = self.initial_conditions.loc['f', 'bounds'][0]
        fmax = self.initial_conditions.loc['f', 'bounds'][1]
        
        
        if self._zero_extinction:
            age, mass, f = theta
            
            index = self.initial_conditions.index.drop('Av')
            
            if not ((amin <= age <= amax) and (mmin <= mass <= mmax) and (fmin <= f <= fmax)):
                return -np.inf
            
        else:
            age, mass, Av, f = theta
            
            index = self.initial_conditions.index
            
            if not ((amin <= age <= amax) and (mmin <= mass <= mmax) and (avmin <= Av <= avmax) and (fmin <= f <= fmax)):
                return -np.inf
        
        
        lp = 0
        
        # Teff prior is taken care of in log_likelihood so don't need to do it here
        for i in range(len(index)-1):
            x = theta[i]
            
            mu, sigma = self.initial_conditions.loc[index[i], 'prior']
            
            lp += log_normal(x, mu, sigma)
        
        return lp
    
    
    
    
    def log_probability(self, theta):
        """
        The sum of the log prior and the log likelihood.

        Parameters
        ----------
        theta : list or tuple
            The list of fit parameters.

        Returns
        -------
        float
            The Bayesian log probability.

        """
        
        lp = self.log_prior(theta)
        
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(theta)
        

        return lp + ll + self.log_teff_prior
    
    
    
    
class MCMC(object):
    
    """
    Uses `emcee` to run the MCMC simulations.
    See https://emcee.readthedocs.io/en/stable/ for more information.
    
    Parameters
    ----------
    nwalkers : int
        The number of independent walkers in the simulation chain.
    nsteps : int
        The number of iterations of the simulation.
    log_probability_func : object
        The Bayesian probabilty function evaluated by the simulation.
    initial_conditions : DataFrame, optional
        Contains the metadata defining boundary conditions, priors, and initial positions for the fit parameters.
    moves : list, optional
        List of tuples containing one or multiple `emcee` move functions and weights. 
        See https://emcee.readthedocs.io/en/stable/user/moves/ for more information.
        The default is `None`.
    pool : object, optional
        The pool object used to parallelize the simulation. The default is `None`.
    zero_extinction : bool, optional
        If `True`, set extinction to zero (Av=0). The default is `False`.
    walker_init_tol : int, optional
        How many attempts should be made to initialize the walker positions
        before the simulation starts? The deault is 1000.
    walker_init_context : object, optional
        Context manager within which the walkers are initialized. Will either
        be this library's `WaitingAnimation` or `contextlib.nullcontext`.
    
    """
    
    def __init__(self, nwalkers, nsteps, log_probability_func, initial_conditions=None, moves=None, pool=None, zero_extinction=False, walker_init_tol=1000, walker_init_context=None):
        
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        
        self._log_probability_func = log_probability_func
        
        self._pool = pool
        
        if initial_conditions is not None:
            self.initial_conditions = initial_conditions.copy()
            
        else:
            self.initial_conditions = InitialConditions().initial_conditions.copy()
            
        if moves is not None:
            self.moves = moves.copy()
            
        else:
            self.moves = Moves().moves.copy()
            
        self._zero_extinction = zero_extinction
        
        self._walker_init_tol = walker_init_tol
        
        self._walker_init_context = walker_init_context
        
        if self._zero_extinction:
            self.ndim = 3
            self.initial_conditions.drop(index='Av', inplace=True)
        else:
            self.ndim = 4
        
        self.sampler = emcee.EnsembleSampler(nwalkers=self.nwalkers,
                                     ndim=self.ndim,
                                     log_prob_fn=self._log_probability_func,
                                     pool=self._pool,
                                     moves=self.moves
                                     )
        
        
        
        
    # get initial positions making sure they create a finite log-probability
    def _get_pos(self):
        
        init_pos = list(self.initial_conditions['position'].dropna().values)
        perturbation = list(self.initial_conditions['perturbation'].dropna().values)
        
        log_prob = self._log_probability_func
        
        pos = []
        i = 0
        while len(pos) < self.nwalkers:
            
            i+=1
            
            p = init_pos + perturbation * np.random.randn(self.ndim)
            
            if np.isfinite(log_prob(p)):
                pos.append(p)
                
            if i > self._walker_init_tol:
                print("Failed to initialize walkers. Try changing the intial conditions.")
                
                return False
            
        return pos
    
    
    
    
    def run(self, progress=True):
        """
        Initiates the MCMC simulation. Updates `self.sampler` so nothing to explicitly return.

        Parameters
        ----------
        progress : bool, optional
            If `True`, provides a progress bar during the sumulation.  The default is True.

        """
        
        with self._walker_init_context:
            pos = self._get_pos()
            print('')
        
        if pos is False:
            
            self.sampler = False
            return
        
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            self.sampler.run_mcmc(pos, self.nsteps, progress=progress)
            
            
            
            
  
            
            
            
        
            
            
        
        
    



        
        
        
        
