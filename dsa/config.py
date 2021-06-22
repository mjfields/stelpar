#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:23:48 2021

@author: mjfields
"""


import os




MAGMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'resources/feiden_grid_mag.csv'))

INTERPMAGMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'resources/feiden_grid_mag_interpolated.csv'))

STDMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'resources/feiden_grid_std.csv'))

INTERPSTDMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'resources/feiden_grid_std_interpolated.csv'))

GRIDCACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), r'resources/.grid_cache'))

FILTERPROFILESPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'resources/Filter_Profiles'))

REMOTEHOMEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..'))