# -*- coding: utf-8 -*-


import os
import tempfile




MAGMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'datafiles/dsep_grid_mag.csv.gz'))

INTERPMAGMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'datafiles/dsep_grid_mag_interpolated.csv.gz'))

STDMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'datafiles/dsep_grid_std.csv.gz'))

INTERPSTDMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'datafiles/dsep_grid_std_interpolated.csv.gz'))

PARSECMODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'datafiles/parsec_grid.csv.gz'))

FILTERPROFILESPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), r'datafiles/Filter_Profiles'))

CACHEDIR = tempfile.TemporaryDirectory()
