.. _photometry:

Photometry
==========

``stelpar`` collects and transforms measured photometry with :class:`MeasuredPhotometry`
to compare with the synthetic photometry in the evolution model. With 
:class:`SyntheticPhotometry` photometry is extinction-corrected and a measured
photometry model is formed for direct comparison with the evolutionary model.

.. autoclass:: stelpar.MeasuredPhotometry
    :members:
    :member-order: bysource

.. autoclass:: stelpar.SyntheticPhotometry
    :members:
    :member-order: bysource
    :exclude-members: wav_array, band_array