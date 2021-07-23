#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:32:03 2021

@author: mjfields
"""


from dsa.radius.photometry import MeasuredPhotometry, SyntheticPhotometry
from dsa.radius.simulation import Probability, MCMC
from dsa.radius.target import Target
from dsa.radius.estimate import Estimate
from dsa.config import MODELGRIDPATH, INTERPMODELGRIDPATH, MODELGRIDPICKLEFILE
from dsa.utils import load_isochrone, interpolate_true, interpolate_nearest, WaitingAnimation


import numpy as np
import timeit




def test_true_interpolation_function():
    
    pass




def test_speed_of_photometry_model(n, repeat):
    
    setup = """
grid = load_isochrone(INTERPMODELGRIDPATH)

agelist = grid.index.get_level_values('age').drop_duplicates()
masslist = grid.index.get_level_values('mass').drop_duplicates()

grid.to_pickle(MODELGRIDPICKLEFILE)

model_grid = MODELGRIDPICKLEFILE

mp = MeasuredPhotometry('K2-33')

photometry, termination_message = mp.get_data()

sp = SyntheticPhotometry(photometry, model_grid, interp_method='nearest', agelist=agelist, masslist=masslist)
"""
    
    func = "sp.photometry_model(1000.0, 1.0, 1.0)"
    
    with WaitingAnimation(f"calculating time for {repeat} iterations of {n} processes"):
        t = np.array(timeit.repeat(setup=setup, stmt=func, number=n, globals=globals(), repeat=repeat))
        print('')

    mean = np.mean(t/n)
    std = np.std(t/n)
    
    
    print(f"t = {mean:.3g} +\- {std:.3g} seconds")
    
