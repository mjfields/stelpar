# Changelog

All notable changes to this codebase from now on will be documented in this file.

# [1.0.0] (2025-08-31)

### Added

- Proper version control including a unified version across all areas of the project.
- Notebook with plot examples.
- Photometry class documentation.

### Changed

- Preferred installation instructions.
- Updated plot API guide.
    - Added documentation for magnitude eyecheck, flux SED, and time-series plots.
- Improved the usability of the corner plot function.
- Removed TOC from tutorial notebooks since this is handled internally by Sphinx/ReadtheDocs
- Included more extinction and temperature values in unit tests.

# [unreleased/0.1.0] (2021-2025)

### Added

- The main codebase

Including but not limited to:

- Pipeline that downloads, transforms, and unifies measured photometry; ready for comparison to synthetic photometry.
- Pipeline that corrects synthetic photometry for extinction and creates a synthetic model for comparison to measured photometry.
    - Including significant (optional) speed improvements using Numpy and Numba/JIT.
- Bayesian log-probability functions, including Gaussian and uniform prior handling.
- Functions that wrap MCMC capabilities via github.com/dfm/emcee.
- Pipeline that extracts posterior probability distributions for all fit (and non-fit) parameters and integrates the results.
    - Including (optional) maximum-likelihood results.
- Capability to save results in a hierarchical directory tree.
- Capability to load stored results from directory tree.
- Several plot types for analysis.
- Ability to configure target metadata and initial conditions; useful for improved results from MCMC.
    - Including add/drop photometry, Gaussian and uniform priors, user-input photometry/parallax, and target coordinate handling.
- Testing suite with Pytest.
- Tutorials with Jupyter notebooks.
- API documentation with Sphinx and ReadtheDocs.