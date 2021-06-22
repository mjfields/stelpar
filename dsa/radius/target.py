#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:15:01 2020

@author: mjfields
"""


import pandas as pd

try:
    from IPython import display
except ModuleNotFoundError:
    pass

from astropy.coordinates import SkyCoord

from dsa.radius.metadata import InitialConditions, Moves, PhotometryMetadata


__all__ = ['Target']




class Target(object):
    
    """
    Creates the object that holds target-specific metadata that can be accessed
    and used by the simulation.
    
    """
    
    def __init__(self, name, coords=None, **coord_kwargs):
        
        self.name = name
        
        if isinstance(coords, SkyCoord):
            self.coords = coords
            self._ra = self.coords.ra
            self._dec = self.coords.dec
            
        elif coords is not None:
            self.coords = SkyCoord(*coords, **coord_kwargs)
            self._ra = self.coords.ra
            self._dec = self.coords.dec
            
        else:
            self.coords=coords
            self._ra = ''
            self._dec = ''
        
        self._ic = InitialConditions()
        self._moves = Moves()
        self._phot_meta = PhotometryMetadata()
    
    
    
    
    def show_metadata(self):
        
        try:
            display(f'Target: {self.name} {self._ra} {self._dec}\n')
            display('Initial Conditions:\n', self.initial_conditions)
            display('\nMoves:\n', self.moves)
            display('\nPhotometry Metadata:\n', self.photometry_meta)
            
        except:
            print(f'Target: {self.name} {self._ra} {self._dec}\n')
            print('Initial Conditions:\n', self.initial_conditions)
            print('\nMoves:\n', self.moves)
            print('\nPhotometry Metadata:\n', self.photometry_meta)
    
    
    
    
    @property
    def initial_conditions(self):
        
        index = pd.MultiIndex.from_product([[self.name], self._ic.initial_conditions.index.values], names=['target', 'parameter'])
        
        df = pd.DataFrame(data=self._ic.initial_conditions.values, index=index, columns=self._ic.initial_conditions.columns)
        
        return df
    
    
    
    
    @initial_conditions.setter
    def initial_conditions(self, ic):
        
        for key in ic:
            setattr(self._ic, key, ic[key])
            
            
            
            
    @property
    def moves(self):
        
        return self._moves.moves
    
    
    
    
    @moves.setter
    def moves(self, move):
        
        setattr(self._moves, 'moves', move)
        
        
        
        
    @property
    def isochrone_analogs(self):
        
        return self._phot_meta.isochrone_analogs
        
        
        
        
    @property
    def photometry_meta(self):
        
        return self._phot_meta.photometry
    
    
    
    
    def add_photometry(self, photometry_dict):
        
        return self._phot_meta.add(photometry_dict)
    
    
    
    
    def remove_photometry(self, photometry_dict):
        
        return self._phot_meta.remove(photometry_dict)
            
            
            
            
    def reset(self, params=None, conds=None, moves=True, photometry=True):
        
        self._ic.reset(params=params, conds=conds)
        
        if moves:
            self._moves.reset()
            
        if photometry:
            self._phot_meta.reset()











