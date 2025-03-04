# -*- coding: utf-8 -*-




import warnings

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mplcol
import corner
import seaborn as sns

from .utils import (
    plot_labels,
    phot_plot_labels,
    frac_res_sigma,
    residual,
    sigma
    )





def time_series(samples, savefile=None, show=True, alpha=0.4):
    
    """
    Makes a time-series plot of the fit parameters' positions.
    
    Parameters
    ----------
    samples : array
        The 3D chain of fit parameter posteriors.
    savefile : str, optional
        The file location to save the figure. If `None` (default), will not save the figure.
    show : bool, optional
        If `True` (default), displays the figure.
        
    Returns
    -------
    fig : Figure
        `matplotlib.figure.Figure` object which contains the plot elements.
    
    """
    
    ndim = samples.shape[2]
    
    fig, axes = plt.subplots(ndim, sharex=True)
    
    if ndim==3:
        labels = plot_labels(zero_extinction=True).loc[['age', 'mass', 'f'], 'fancy_label']
    elif ndim==4:
        labels = plot_labels().loc[['age', 'mass', 'Av', 'f'], 'fancy_label']
    
    for i in range(ndim):
        p = labels.index.values[i]
            
        ax = axes[i]
        ax.plot(samples[:, :, i], 'k', alpha=alpha)
        
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels.loc[p])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
        if i < ndim-1:
            ax.tick_params(bottom=False)
    
    axes[-1].set_xlabel('Step Number')
    
    
    if savefile is not None:
        fig.savefig(savefile)
        
    if not show:
        plt.close(fig)
        
    
    return fig
    
    
    
    
def _corner_plot(chain, bins=20, r=None, corner_kwargs=None, savefile=None, show=True):
    
    """
    Creates a corner plot showing histograms and 2D projections of the stellar
    parameters.
    
    Parameters
    ----------
    chain : DataFrame
        Contains the posterior distributions from which the corner plot will be made.
    bins : int, optional
        The number of bins to use in the histograms. The default is 20.
    r : iterable, optional
        Contains either tuples with (lower, upper) bounds for each parameter
        or floats that state the fraction of samples to include in the plots.
        See https://corner.readthedocs.io/en/latest/ for more specific information.
        If `None` (default), will use 99.7% of the samples for each parameter.
    corner_kwargs : dict, optional
        Additional keyword arguments to pass to corner. The default is `None`.
    savefile : str, optional
        The file location to save the figure. If `None` (default), will not save the figure.
    show : bool, optional
        If `True` (default), displays the figure.
        
    Returns
    -------
    fig : Figure
        `matplotlib.figure.Figure` object which contains the plot elements.
    
    """
    
    samples = chain.values
    ndim = samples.shape[1]
    
    if 'Av' not in chain.columns:
        labels = plot_labels(zero_extinction=True)['fancy_label'].tolist()
    else:
        labels = plot_labels()['fancy_label'].tolist()
    
    if r is None:
        r = [0.997]*ndim
        
        
    if corner_kwargs is None:
        corner_kwargs = dict()
        
    
    fig = corner.corner(samples, bins=bins, labels=labels, fill_contours=True, plot_datapoints=False, range=r, hist_kwargs={'linewidth':2.5}, **corner_kwargs)
    
    axes = np.array(fig.axes).reshape((ndim, ndim))
    
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            
            ax = axes[i][j]
                
            # labelpad doesn't work so doing it manually
            if j == 0:
                ax.yaxis.set_label_coords(-0.45, 0.5)
            if i == ndim-1:
                ax.xaxis.set_label_coords(0.5, -0.45)
                
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    
    if savefile is not None:
        fig.savefig(savefile)
        
    if not show:
        plt.close(fig)
        
    
    return fig




def corner_plot(
        chains, 
        savefile=None, 
        show=True, 
        nsamples=None,
        contour_fill_kws=None,
        contour_outline_kws=None, 
        diag_kws=None, 
        grid_kws=None,
        show_titles=False,
        title_fmt='.2f',
        title_kws=None,
    ):
    
    """
    Creates a corner plot showing smoothed histograms and 2D contours of the stellar
    parameters.
    
    Parameters
    ----------
    chains : DataFrame
        Contains the posterior distributions from which the corner plot will be made.
    savefile : str, optional
        The file location to save the figure. If `None` (default), will not save the figure.
    show : bool, optional
        If `True` (default), displays the figure.
    nsamples : int, optional
        The number of random samples to draw from the chains to make the plots.
        If `None`, will use the full chains. The default is `None`.
    :param {contour_fill, contour_outline, diag, grid}_kws: : dicts, optional
        Dictionaries of keyword arguments. See seaborn.pairplot and seaborn.PairGrid
        for more information. The difference is here `plot_kws` is broken into "fill" and "outline"
        keywords for more customization. See the `default_*_kws` below for an example.
        The defaults are `None`.
    show_titles : bool, optional
        Whether to show titles above each diagonal controlled by `title_{fmt, kws}`.
        The default is `False`
    title_fmt : str, optional
        The 
        
    Returns
    -------
    grid : PairGrid
        `seaborn.PairGrid` object which contains the plot elements.
    
    """

    for param in chains.columns:
        chains = chains.rename(
            columns={
                param : plot_labels().loc[param, 'fancy_label']
            }
        )
    
    if nsamples is not None:
        rng = np.random.default_rng()
        mask = rng.choice(np.arange(len(chains)), size=nsamples, replace=False)
    else:
        mask = np.arange(len(chains))

    ## setup grid
    default_grid_kws = dict(corner=True, diag_sharey=False)
    if grid_kws is None:
        grid_kws = default_grid_kws
    else:
        grid_kws = {**default_grid_kws, **grid_kws}

    grid = sns.PairGrid(chains.iloc[mask], **default_grid_kws)

    ## contour fill
    default_contour_fill_kws = dict(fill=True, color='gray', levels=4, cut=0)
    if contour_fill_kws is None:
        contour_fill_kws = default_contour_fill_kws
    else:
        contour_fill_kws = {**default_contour_fill_kws, **contour_fill_kws}
    
    grid.map_offdiag(sns.kdeplot, **contour_fill_kws)

    ## contour outline
    default_contour_outline_kws = dict(fill=False, color='black', levels=4, linewidths=3, cut=0)
    if contour_outline_kws is None:
        contour_outline_kws = default_contour_outline_kws
    else:
        contour_outline_kws = {**default_contour_outline_kws, **contour_outline_kws}
    
    grid.map_offdiag(sns.kdeplot, **contour_outline_kws)

    ## diagonal posteriors
    default_diag_kws = dict(fill=True, color='gray', edgecolor='black', linewidth=3, cut=0)
    if diag_kws is None:
        diag_kws = default_diag_kws
    else:
        diag_kws = {**default_diag_kws, **diag_kws}
    grid.map_diag(sns.kdeplot, **diag_kws)

    if show_titles:
        if title_kws is None:
            title_kws = dict()
        for label, chain, diag_ax in zip(chains.columns, chains.T.values, grid.diag_axes):
            p = np.percentile(chain, [16, 50, 84])
            q = np.diff(p)
            diag_ax.set_title(
                fr"${label.replace('$', '')} = {p[1]:{title_fmt}}_{{-{q[0]:{title_fmt}}}}^{{+{q[1]:{title_fmt}}}}$",
                **title_kws
            )

    if savefile is not None:
        grid.savefig(savefile)

    if not show:
        plt.close(grid.figure)

    return grid




def flux_v_wavelength(photometry, title=None, singlefig=True, savefile=None, show=True):
        
    """
    Creates a plot of flux density vs. wavelength.
    
    Parameters
    ----------
    photometry : DataFrame
        The measured and estimated magnitudes and other photometric data.
    title : str, optional
        The figure title. The defult is `None`.
    singlefig : bool, optional
        If `True` (default), presents the figure with one plot containing median and
        max-likelihood values. Otherwise, separates those values to different subplots 
        in the figure.
    savefile : str, optional
        The file location to save the figure. If `None` (default), will not save the figure.
    show : bool, optional
        If `True` (default), displays the figure.
        
    Returns
    -------
    fig : Figure
        `matplotlib.figure.Figure` object which contains the plot elements.
    
    """ 
    
    wav = photometry['wavelength'].divide(1e4) # microns
    
    obs_flux = photometry['flux']
    obs_flux_error = photometry['flux_error']
    
    med_flux = photometry['median_flux']
    med_flux_error = photometry['median_flux_error']
    med_frac_res = frac_res_sigma(obs_flux, obs_flux_error, med_flux, med_flux_error)
    
    max_prob = True
    try:
        max_flux = photometry['max_probability_flux']
        max_flux_error = photometry['max_probability_flux_error']
        max_frac_res = frac_res_sigma(obs_flux, obs_flux_error, max_flux, max_flux_error)
    except KeyError:
        max_prob = False

    if not singlefig and not max_prob:
        warnings.warn(
            "Unable to give 2-panel without max-probability values. "
            "Next time, pass `max_prob=True` to `Estimate.posterior()`."
        )
        singlefig=True
    
    
    
    mkrsize = 15
    mkredgewidth = 2
    elinewidth = 2
    alpha = 0.7
    ylabel_coords = (-0.085, 0.5)
    
    wav_label = r'$\lambda \ \left( \mu\mathrm{m} \right)$'
    flux_label = r'$F_{\lambda} \ \left( \mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1} \ \AA^{-1} \right)$'
    
    obs_color = 'gray'
    med_color = (*mplcol.to_rgb('navy'), alpha) 
    max_color = (*mplcol.to_rgb('lightgreen'), alpha) 

    obs_marker = 'o'
    med_marker = 's'
    max_marker = 'd'
    
    med_res_color = med_color
    max_res_color = max_color
    
    hcolor = 'black'
    hstyle = '--'
    
    if singlefig:
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, gridspec_kw={'height_ratios':[3,1]}, figsize=(12,8))
        
        ax1.errorbar(
            wav, 
            obs_flux, 
            yerr=obs_flux_error, 
            fmt=obs_marker, 
            markersize=mkrsize, 
            markeredgecolor='black', 
            markerfacecolor=obs_color, 
            markeredgewidth=mkredgewidth, 
            ecolor=obs_color,
            elinewidth=elinewidth,
            label='Observed',
            zorder=1
        )
        
        ax1.errorbar(
            wav, 
            med_flux, 
            yerr=med_flux_error,
            fmt=med_marker,
            markersize=mkrsize,
            markeredgecolor='black', 
            markerfacecolor=med_color,
            markeredgewidth=mkredgewidth,
            ecolor=med_color,
            elinewidth=elinewidth,
            label='Median',
            zorder=2
        )
        
        if max_prob:
            ax1.errorbar(
                wav, 
                max_flux, 
                yerr=max_flux_error,
                fmt=max_marker,
                markersize=mkrsize,
                markeredgecolor='black', 
                markerfacecolor=max_color,
                markeredgewidth=mkredgewidth,
                ecolor=max_color,
                elinewidth=elinewidth,
                label='Max-Likelihood',
                zorder=3
            )
        
        ax1.tick_params(top=False, bottom=False, labelbottom=False, labeltop=False, direction='inout', length=10)
        ax1.set_ylabel(flux_label)
        ax1.yaxis.set_label_coords(*ylabel_coords)
        
        # remove the errorbars from legend
        handles, labels = ax1.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        
        ax1.legend(handles, labels, labelspacing=1.5, borderpad=1)
        
        
        ax2.axhline(y=0, c=hcolor, linestyle=hstyle, zorder=0)
        
        ax2.errorbar(
            wav, 
            med_frac_res, 
            fmt=med_marker, 
            markersize=mkrsize, 
            markeredgecolor='black', 
            markerfacecolor=med_color, 
            markeredgewidth=mkredgewidth,
            ecolor=med_res_color,
            elinewidth=elinewidth,
            label='Median',
            zorder=1
        )
        
        if max_prob:
            ax2.errorbar(
                wav, 
                max_frac_res,
                fmt=max_marker, 
                markersize=mkrsize, 
                markeredgecolor='black', 
                markerfacecolor=max_color, 
                markeredgewidth=mkredgewidth,
                ecolor=max_res_color,
                elinewidth=elinewidth,
                label='Max-Likelihood',
                zorder=2
            )
        
        ax2.tick_params(top=True, direction='inout', length=10)
        ax2.set_xlabel(wav_label)
        ax2.set_ylabel('Fractional\nResidual '+r'($\sigma$)')
        ax2.yaxis.set_label_coords(*ylabel_coords)

        res_ylim = ax2.get_ylim()
        ax2.set_ylim(*np.array(res_ylim)+np.array([-1, 1]))
        
        
        
    else:
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, gridspec_kw={'height_ratios':[3,1]}, figsize=(24,8))
        
        ax1.errorbar(
            wav, 
            obs_flux, 
            yerr=obs_flux_error, 
            fmt=obs_marker, 
            markersize=mkrsize, 
            markeredgecolor='black', 
            markerfacecolor=obs_color, 
            markeredgewidth=mkredgewidth, 
            ecolor=obs_color,
            elinewidth=elinewidth,
            label='Observed',
            zorder=1
            )
        
        ax1.errorbar(
            wav, 
            med_flux, 
            yerr=med_flux_error,
            fmt=med_marker,
            markersize=mkrsize,
            markeredgecolor='black', 
            markerfacecolor=med_color,
            markeredgewidth=mkredgewidth,
            ecolor=med_color,
            elinewidth=elinewidth,
            label='Median',
            zorder=2
            )
        
        ax1.tick_params(top=False, bottom=False, labelbottom=False, labeltop=False, direction='inout', length=10)
        ax1.set_ylabel(flux_label)
        ax1.yaxis.set_label_coords(*ylabel_coords)
        
        flux_ylim = ax1.get_ylim()
        
        # remove the errorbars from legend
        handles, labels = ax1.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        
        ax1.legend(handles, labels, labelspacing=1.5, borderpad=1)
        
        
        ax3.axhline(y=0, c=hcolor, linestyle=hstyle, zorder=0)
        
        ax3.errorbar(
            wav, 
            med_frac_res,
            fmt=med_marker, 
            markersize=mkrsize, 
            markeredgecolor='black', 
            markerfacecolor=med_color, 
            markeredgewidth=mkredgewidth,
            ecolor=med_res_color,
            elinewidth=elinewidth,
            label='Median',
            )
        
        ax3.tick_params(top=True, direction='inout', length=10)
        ax3.set_xlabel(wav_label)
        ax3.set_ylabel('Fractional\nResidual '+r'($\sigma$)')
        ax3.yaxis.set_label_coords(*ylabel_coords)

        res_ylim = ax3.get_ylim()
        ax3.set_ylim(*np.array(res_ylim)+np.array([-1, 1]))
        
        
        ax2.errorbar(
            wav, 
            obs_flux, 
            yerr=obs_flux_error, 
            fmt=obs_marker, 
            markersize=mkrsize, 
            markeredgecolor='black', 
            markerfacecolor=obs_color, 
            markeredgewidth=mkredgewidth, 
            ecolor=obs_color,
            elinewidth=elinewidth,
            label='Observed',
            zorder=1
            )
        
        ax2.errorbar(
            wav, 
            max_flux, 
            yerr=max_flux_error,
            fmt=max_marker,
            markersize=mkrsize,
            markeredgecolor='black', 
            markerfacecolor=max_color,
            markeredgewidth=mkredgewidth,
            ecolor=max_color,
            elinewidth=elinewidth,
            label='Max-Likelihood',
            zorder=2
            )
        
        ax2.tick_params(top=False, bottom=False, labelbottom=False, labeltop=False, direction='inout', length=10)
        ax2.set_ylabel(flux_label)
        ax2.yaxis.set_label_coords(*ylabel_coords)
        
        ax2.set_ylim(flux_ylim)
        
        
        # remove the errorbars from legend
        handles, labels = ax2.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        
        ax2.legend(handles, labels, labelspacing=1.5, borderpad=1)
                
        
        ax4.axhline(y=0, c=hcolor, linestyle=hstyle, zorder=0)
        
        ax4.errorbar(
            wav, 
            max_frac_res,
            fmt=max_marker, 
            markersize=mkrsize, 
            markeredgecolor='black', 
            markerfacecolor=max_color, 
            markeredgewidth=mkredgewidth,
            ecolor=max_res_color,
            elinewidth=elinewidth,
            label='Max-Likelihood',
            )
        
        ax4.tick_params(top=True, direction='inout', length=10)
        ax4.set_xlabel(wav_label)
        ax4.set_ylabel('Fractional\nResidual '+r'($\sigma$)')
        ax4.yaxis.set_label_coords(*ylabel_coords)
        
        res_ylim = ax4.get_ylim()
        ax4.set_ylim(*np.array(res_ylim)+np.array([-1, 1]))
        
        
        
    fig.subplots_adjust(hspace=0)
    
    
    if title is not None:
        fig.suptitle(title)
    
    if savefile is not None:
        fig.savefig(savefile)
        
    if not show:
        plt.close(fig)
    
    
    return fig




def mag_v_wavelength(photometry, savefile=None, show=True):
    
    """
    Creates a plot of absolute magnitude vs. wavelength.
    
    Parameters
    ----------
    photometry : DataFrame
        The measured and estimated magnitudes and other photometric data.
    savefile : str, optional
        The file location to save the figure. If `None` (default), will not save the figure.
    show : bool, optional
        If `True` (default), displays the figure.
        
    Returns
    -------
    fig : Figure
        `matplotlib.figure.Figure` object which contains the plot elements.
    
    """ 
    
    wav = photometry['wavelength'].divide(1e4) # microns
    
    obs_mag = photometry['ABSOLUTE_MAGNITUDE']
    obs_mag_error = photometry['ABSOLUTE_MAGNITUDE_ERROR']
    
    med_mag = photometry['MEDIAN_ABSOLUTE_MAGNITUDE']
    med_mag_error = photometry['MEDIAN_ABSOLUTE_MAGNITUDE_ERROR']
    
    max_mag = photometry['MAX_PROBABILITY_ABSOLUTE_MAGNITUDE']
    max_mag_error = photometry['MAX_PROBABILITY_ABSOLUTE_MAGNITUDE_ERROR']
    
    med_res = residual(obs_mag, med_mag)
    max_res = residual(obs_mag, max_mag)
    
    med_res_error = sigma(obs_mag_error, med_mag_error)
    max_res_error = sigma(obs_mag_error, max_mag_error)
    
    
    mkrsize = 7
    mkredgewidth = 2.5
    elinewidth = 2
    alpha = 1.0
    ylabel_coords = (-0.085, 0.5)
    
    wav_label = r'$\mathbf{\\lambda} \\ \\left( \mathbf{\\mu}\mathrm{m} \\right)$'
    mag_label = 'Absolute Magnitude [mag]'
    
    obs_color = 'black'
    med_color = (0.35, 0.55, 0.35)
    max_color = '#2de37f'
    
    med_res_color = med_color
    max_res_color = max_color
    
    hcolor = 'black'
    hstyle = '--'
    
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, gridspec_kw={'height_ratios':[3,1]}, figsize=(10,8))
        
    ax1.errorbar(
        wav, 
        obs_mag, 
        yerr=obs_mag_error, 
        fmt='o', 
        markersize=mkrsize, 
        markeredgecolor=obs_color, 
        markerfacecolor=obs_color, 
        markeredgewidth=mkredgewidth, 
        ecolor=obs_color,
        elinewidth=elinewidth,
        label='Observed',
        zorder=1
        )
    
    ax1.errorbar(
        wav, 
        med_mag, 
        yerr=med_mag_error,
        fmt='s',
        markersize=mkrsize,
        markeredgecolor=med_color, 
        markerfacecolor='None',
        markeredgewidth=mkredgewidth,
        ecolor=med_color,
        elinewidth=elinewidth,
        label='Median',
        zorder=2
        )
    
    ax1.errorbar(
        wav, 
        max_mag, 
        yerr=max_mag_error,
        fmt='D',
        markersize=mkrsize,
        markeredgecolor=max_color, 
        markerfacecolor='None',
        markeredgewidth=mkredgewidth,
        ecolor=max_color,
        elinewidth=elinewidth,
        label='Max-Likelihood',
        zorder=3
        )
    
    ax1.tick_params(top=False, bottom=False, labelbottom=False, labeltop=False, direction='inout', length=10)
    ax1.set_ylabel(mag_label)
    ax1.yaxis.set_label_coords(*ylabel_coords)
    
    # remove the errorbars from legend
    handles, labels = ax1.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    
    ax1.legend(handles, labels)
    
    
    ax2.axhline(y=0, c=hcolor, linestyle=hstyle, zorder=0)
    
    ax2.errorbar(
        wav, 
        med_res, 
        yerr=med_res_error, 
        fmt='s', 
        markersize=mkrsize, 
        markeredgecolor=med_res_color, 
        markerfacecolor='None', 
        markeredgewidth=mkredgewidth,
        ecolor=med_res_color,
        elinewidth=elinewidth,
        label='Median',
        alpha=alpha,
        zorder=1
                  )
    
    ax2.errorbar(
        wav, 
        max_res, 
        yerr=max_res_error, 
        fmt='D', 
        markersize=mkrsize, 
        markeredgecolor=max_res_color, 
        markerfacecolor='None', 
        markeredgewidth=mkredgewidth,
        ecolor=max_res_color,
        elinewidth=elinewidth,
        label='Max-Likelihood',
        alpha=alpha,
        zorder=2
                  )
    
    ax2.tick_params(top=True, direction='inout', length=10)
    ax2.set_xlabel(wav_label)
    ax2.set_ylabel('Residual')
    ax2.yaxis.set_label_coords(*ylabel_coords)
    
    
    fig.subplots_adjust(hspace=0)
    
    
    if savefile is not None:
        fig.savefig(savefile)
        
    if not show:
        plt.close(fig)
    
    
    return fig




def mag_v_wavelength_eyecheck(photometry, name=None, savefile=None, show=True):
    
    """
    Creates a plot of absolute magnitude vs. wavelength.
    
    Parameters
    ----------
    photometry : DataFrame
        The measured and estimated magnitudes and other photometric data.
    savefile : str, optional
        The file location to save the figure. If `None` (default), will not save the figure.
    show : bool, optional
        If `True` (default), displays the figure.
        
    Returns
    -------
    fig : Figure
        `matplotlib.figure.Figure` object which contains the plot elements.
    
    """ 
    
    wav = photometry['wavelength'].divide(1e4) # microns
    
    obs_mag = photometry['apparent_magnitude']
    obs_mag_error = photometry['apparent_magnitude_error']
    
    labels = [phot_plot_labels()[band] for band in photometry.index]
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
        
    ax.errorbar(
        wav, 
        obs_mag, 
        yerr=obs_mag_error, 
        marker='o', 
        linestyle='',
        markersize=12, 
        markerfacecolor='lightblue', 
        markeredgecolor='black', 
        markeredgewidth=1.5, 
        ecolor='black',
        elinewidth=1.5,
        )
    
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel("$m$ [mag]")
    
    ylim = ax.get_ylim()
    
    xoffset = 0
    yoffset = -0.05 * (ylim[1] - ylim[0])
    
    for w, m, l in zip(wav, obs_mag, labels):
        plt.text(w+xoffset, m+yoffset, l)
        
    ax.set_ylim(ylim[0]+yoffset, ylim[1])
    
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    
    if name is not None:
        ax.set_title(name)
    
    
    if savefile is not None:
        fig.savefig(savefile)
    
    if not show:
        plt.close(fig)
    
    
    # return fig








