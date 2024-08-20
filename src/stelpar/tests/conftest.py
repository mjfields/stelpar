import pytest


@pytest.fixture(scope="module")
def catalogs():

    return [
        'II/246/out',
        'I/355/gaiadr3',
        'V/154/sdss16',
        'II/336/apass9',
        'I/259/tyc2',
        'I/239/hip_main'
    ]


@pytest.fixture(scope="module")
def ltt7379(): # from ESO spectrophotometric standards # G0-type
    
    return dict(
        name="LTT 7379",
        RA="18 36 25.95",
        DEC="-44 18 36.91",
        epoch="J2000",
        pm_RA=(-169.79, 0.03),
        pm_DEC=(-159.81, 0.03),
        plx=(9.44, 0.04), # mas
        RUWE=2.46,
        photometry={
            'johnson_vmag':10.18,
            'johnson_bmag':10.83,
            'gaia_gmag':10.06,
            'gaia_bpmag':10.38,
            'gaia_rpmag':9.57,
            '2mass_jmag':9.03,
            '2mass_hmag':8.72,
            '2mass_kmag':8.64,
            'tycho_bmag':10.90,
            'tycho_vmag':10.32,
            'hipparcos_hpmag':10.36,
        },
    )


@pytest.fixture(scope="module")
def hd2892(): # standard from Landolt2009 # K0V
    
    return dict(
        name="HD 2892",
        RA="00 32 12.15",
        DEC="+01 11 17.28",
        epoch="J2000",
        pm_RA=(7.34, 0.02),
        pm_DEC=(-0.61, 0.02),
        plx=(0.99, 0.02), # mas
        RUWE=0.91,
        photometry={
            'johnson_vmag':9.37,
            'johnson_bmag':10.68,
            'gaia_gmag':8.94,
            'gaia_bpmag':9.63,
            'gaia_rpmag':8.14,
            '2mass_jmag':7.09,
            '2mass_hmag':6.40,
            '2mass_kmag':6.22,
            'tycho_bmag':11.03,
            'tycho_vmag':9.55,
        },
    )


@pytest.fixture(scope="module")
def iras04171(): # from Lopez-Valdivia+2023 # M2
    
    return dict(
        name="IRAS 04171+2756",
        RA="04 20 26.07",
        DEC="+28 04 09.05",
        epoch="J2000",
        pm_RA=(8.32, 0.02),
        pm_DEC=(25.91, 0.02),
        plx=(7.81, 0.02), # mas
        RUWE=1.18,
        photometry={
            'johnson_vmag':14.76,
            'johnson_bmag':16.27,
            'gaia_gmag':13.45,
            'gaia_bpmag':14.91,
            'gaia_rpmag':12.27,
            '2mass_jmag':10.61,
            '2mass_hmag':9.95,
            '2mass_kmag':9.70,
            'sdss_gmag':15.55,
            'sdss_rmag':14.11,
            'sdss_imag':16.96,
            'sdss_zmag':12.03,
        },
    )