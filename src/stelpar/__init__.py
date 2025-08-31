# -*- coding: utf-8 -*-

# suppress SyntaxWarning from Isochrones on first import of stelpar
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"isochrones(\.|$)")

from .photometry import MeasuredPhotometry, SyntheticPhotometry
from .simulation import Probability, MCMC
from .estimate import Estimate
from .target import Target
from . import plot, output

__version__ = "1.0.0"

__all__ = [
    'Estimate',
    'MCMC',
    'MeasuredPhotometry',
    'Probability',
    'SyntheticPhotometry',
    'Target',
    'plot',
    'output'
    ]