# -*- coding: utf-8 -*-




import pandas as pd

try:
    from IPython import display
except ModuleNotFoundError:
    pass

from astropy.coordinates import SkyCoord

from .metadata import InitialConditions, Moves, PhotometryMetadata




__all__ = ['Target']




class Target(object):
    
    """
    Creates the object that holds target-specific metadata that can be accessed
    and used by the simulation.

    Parameters
    ----------
    name : str
        The target name. If ``coords=None``, the target name will be used when
        querying for photometry.
    coords : None, `list-like`, or `astropy.coordinates.SkyCoord`; optional
        Options include:
            - ``None`` (default): `name` will be used to query photometry.
            - `list-like` object of the form (RA, DEC): will create a `SkyCoord`
              instance. Additional options (e.g., units) must be passed to
              `coord_kwargs`.
            - `astropy.coordinates.SkyCoord`: `coord_kwargs` is ignored.
        If `list-like` or `astropy.coordinates.SkyCoord`, these coordinates
        will be used to query photometry.
    coord_kwargs : optional
        Any additional keyword arguments passed to `astropy.coordinates.SkyCoord`
        when passing a `list-like` to `coords`.
        
    
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
        
        
        
        
    def __repr__(self):
        
        return (
            f"{self.__class__.__name__}"
            "("
            f"name={self.name!r}, "
            f"coords={self.coords!r}"
            ")"
            )
    
    
    
    
    def show_metadata(self):
        """
        Displays a collection of all the metadata associated with the
        ``Target`` instance.
        """
        
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
        """
        A `pandas.DataFrame` containing initial conditions such as
        bounds, priors, and initial walker positions in parameter space.
        """
        
        index = pd.MultiIndex.from_product(
            [
                [self.name],
                self._ic.initial_conditions.index.values
            ],
            names=['target', 'parameter']
        )
        
        df = pd.DataFrame(
            data=self._ic.initial_conditions.values,
            index=index,
            columns=self._ic.initial_conditions.columns
        )
        
        return df
    
    
    
    
    @initial_conditions.setter
    def initial_conditions(self, ic):
        
        for key in ic:
            setattr(self._ic, key, ic[key])
            
            
            
            
    @property
    def moves(self):
        """
        The object from `emcee.moves` which control the algorithm for
        updating the positions of the walkers.

        See https://emcee.readthedocs.io/en/stable/user/moves/ for more information.
        """
        
        return self._moves.moves
    
    
    
    
    @moves.setter
    def moves(self, move):
        
        setattr(self._moves, 'moves', move)
        
        
        
        
    @property
    def isochrone_analogs(self):
        """
        A list of keywords associated with all the photometry bands available in
        the isochrone models.
        """
        
        return self._phot_meta.isochrone_analogs
        
        
        
        
    @property
    def photometry_meta(self):
        """
        A `pandas.DataFrame` containing metadata for the photometry query
        including catalog and band identifiers.
        """
        return self._phot_meta.photometry
    
    
    
    
    def add_photometry(self, photometry_dict):
        """
        Add catalogs and/or bands to the photometry query.

        Parameters
        ----------
        photometry_dict : dict
            A nested dictionary accessed by the catalog identifier
            containing column names for magnitude and error.
            See the `Target` tutorial for an example.
        """
        return self._phot_meta.add(photometry_dict)
    
    
    
    
    def remove_photometry(self, photometry_dict):
        """
        Remove catalogs and/or bands from being queried.

        Parameters
        ----------
        photometry_dict : dict
            A dictionary whose key is the catalog identifier(s).
            For each catalog there are two options:
                - Remove all photometry associated with that catalog:
                  the value of the dictionary is 'all'.
                - Remove one or multiple bands: the value of the dictionary
                  is the band keyword string or a list of such strings.
            See the `Target` tutorial for an example.
        """
        return self._phot_meta.remove(photometry_dict)
            
            
            
            
    def reset(self, params=None, conds=None, moves=True, photometry=True):
        """
        Resets the photometry metadata to default values.

        Parameters
        ----------
        params : list of str, optional
            The list of parameters whose initial conditions are to be reset to default. 
            If None, all parameters will be reset. The default is None.
        conds : list of str, optional
            The list of conditions which are to be reset to default. 
            If None, all conditions will be reset. The default is None.
        moves : bool, optional
            If `True`, resets moves to default (`emcee.moves.StretchMove`, 1.0).
            If `False`, does not reset the moves. The default is `True`.
        photometry : bool, optional
            If `True`, resets photometry metadata to default.
            If `False`, does not reset the photometry metadata.
            The default is `True`.
        """
        self._ic.reset(params=params, conds=conds)
        
        if moves:
            self._moves.reset()
            
        if photometry:
            self._phot_meta.reset()











