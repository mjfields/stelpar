import pytest
import numpy.testing as npt
import pandas.testing as pdt

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

import synphot

from stelpar.photometry import MeasuredPhotometry, SyntheticPhotometry
from stelpar.config import FILTERPROFILESPATH
from stelpar.utils import filter_profiles_setup

# For whatever reason, synphot only works properly if this is a global variable (otherwise flux is nan)
FILTER_PROFILES = filter_profiles_setup(FILTERPROFILESPATH)


@pytest.mark.parametrize(
    "star",
    [
        'ltt7379',
        'hd2892',
        'iras04171',
    ]
)
class TestMeasuredPhotometry:
    

    def test_simbad_plx_name(self, star, request):

        mp = MeasuredPhotometry(request.getfixturevalue(star)['name'])

        plx, e_plx = mp.simbad_plx()

        npt.assert_allclose(
            actual=np.array([plx[0], e_plx[0]]),
            desired=np.array(np.array(request.getfixturevalue(star)['plx'])),
            rtol=0.1
        )


    def test_simbad_plx_coords(self, star, request):

        mp = MeasuredPhotometry(
            name=request.getfixturevalue(star)['name'],
            coords=SkyCoord(
                request.getfixturevalue(star)['RA'],
                request.getfixturevalue(star)['DEC'],
                unit=(u.hourangle, u.deg)
                )
            )
            
        plx, e_plx = mp.simbad_plx()

        npt.assert_allclose(
            actual=np.array([plx[0], e_plx[0]]),
            desired=np.array(np.array(request.getfixturevalue(star)['plx'])),
            rtol=0.1
        )


    def test_get_data_name(self, star, request):

        mp = MeasuredPhotometry(request.getfixturevalue(star)['name'])

        photometry, term_message = mp.get_data()

        assert photometry is not False

        for key in request.getfixturevalue(star)['photometry'].keys():

            npt.assert_allclose(
                actual=np.array([photometry.loc[key, 'apparent_magnitude']]),
                desired=np.array([request.getfixturevalue(star)['photometry'][key]]),
                rtol=0.01
            )
    

    def test_get_data_coords(self, star, request):

        mp = MeasuredPhotometry(
            name=request.getfixturevalue(star)['name'],
            coords=SkyCoord(
                request.getfixturevalue(star)['RA'],
                request.getfixturevalue(star)['DEC'],
                unit=(u.hourangle, u.deg)
                )
            )

        photometry, term_message = mp.get_data()

        assert photometry is not False

        for key in request.getfixturevalue(star)['photometry'].keys():

            npt.assert_allclose(
                actual=np.array([photometry.loc[key, 'apparent_magnitude']]),
                desired=np.array([request.getfixturevalue(star)['photometry'][key]]),
                rtol=0.01
            )
    

@pytest.mark.parametrize(
    "Av", 
    [
        0, 
        0.1, 
        1.7,
        3,
        6,
        9
    ]
    )
@pytest.mark.parametrize(
    "Teff", 
    [
        2700,
        3300,
        5000,
        6500,
        10000
    ]
    )
@pytest.mark.parametrize(
    "star",
    [
        'ltt7379',
        'hd2892',
        'iras04171',
    ]
)
class TestSyntheticPhotometry:


    def test_effective_stimulus(self, Av, Teff, star, request):

        ### setup ###
        sp = synphot.SourceSpectrum(synphot.models.BlackBodyNorm1D, temperature=Teff)
        wave = np.arange(1000, 30000, 10) * u.angstrom
        phot = request.getfixturevalue(star)['photometry']
        band_array = np.zeros((len(phot), len(wave)))
        for i, band in enumerate(phot.keys()):
            band_array[i] = np.array(FILTER_PROFILES[band]._get_arrays(wave)[1])

        ### stelpar function ###
        sp_array = sp._get_arrays(wave)[1].value # PHOTLAM, default
        flam = synphot.units.convert_flux(wave.value, sp_array, synphot.units.FLAM).value
        flux = flam * band_array
        stelpar_val = SyntheticPhotometry._effective_stimulus(wave, flux, band_array) #* synphot.units.FLAM

        ### synphot function ###
        synphot_array = np.zeros(len(band_array))
        for i, band in enumerate(request.getfixturevalue(star)['photometry'].keys()):

            sp_obs = synphot.Observation(sp, FILTER_PROFILES[band], force='extrap')
            synphot_array[i] = sp_obs.effstim(flux_unit=synphot.units.FLAM).value
        
        ### assert ###
        npt.assert_allclose(
                actual=stelpar_val,
                desired=synphot_array,
                rtol=5.5e-4
            )
    

    def test_numpy_extinction(self, Av, Teff, star, request):

        ### setup ###
        mp = MeasuredPhotometry(request.getfixturevalue(star)['name'])
        photometry, term_message = mp.get_data()

        sp = SyntheticPhotometry(photometry_df=photometry)

        ### synphot ###
        synphot_df = sp.calculate_extinction(Av, Teff, use_synphot=True)
        synphot_extinction = synphot_df['extinction']

        ### numpy ###
        numpy_df = sp.calculate_extinction(Av, Teff, use_synphot=False)
        numpy_extinction = numpy_df['extinction']

        pdt.assert_frame_equal(
            left=synphot_df,
            right=numpy_df,
            check_exact=False,
            check_like=True,
            rtol=5e-4
        )