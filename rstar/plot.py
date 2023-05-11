#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import numpy as np

import matplotlib.pyplot as plt
import corner

from .utils import plot_labels, frac_res, frac_res_error, residual, sigma




plt.style.use('default')

plt.rcParams['font.family'] = 'stixgeneral'
plt.rcParams['font.weight'] = 'demi'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'stixgeneral:demi'
plt.rcParams['mathtext.rm'] = 'stixgeneral:demi'
plt.rcParams['mathtext.it'] = 'stixgeneral:demi:italic'
plt.rcParams['mathtext.sf'] = 'sans:demi' # used for stubborn symbols e.g. \star

plt.rcParams['lines.linewidth'] = 4

plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.labelweight'] = 'demi'
plt.rcParams['axes.labelpad'] = 10.0
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.formatter.limits'] = [-4, 4]
plt.rcParams['axes.edgecolor'] = 'black'
 
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 3

plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 3

plt.rcParams['legend.fontsize'] = 22
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.framealpha'] = 0.5
plt.rcParams['legend.loc'] = 'best'

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.titleweight'] = 'demi'

plt.rcParams['savefig.bbox'] = 'tight'





def time_series(samples, savefile=None, show=True):
    
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
        ax.plot(samples[:, :, i], 'k', alpha=0.4)
        
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
    
    
    
    
def corner_plot(chain, bins=20, r=None, corner_kwargs=None, savefile=None, show=True):
    
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
    
    max_flux = photometry['max_probability_flux']
    max_flux_error = photometry['max_probability_flux_error']
    
    med_frac_res = frac_res(obs_flux, med_flux)
    max_frac_res = frac_res(obs_flux, max_flux)
    
    med_frac_error = frac_res_error(obs_flux, obs_flux_error, med_flux, med_flux_error)
    max_frac_error = frac_res_error(obs_flux, obs_flux_error, max_flux, max_flux_error)
    
    
    mkrsize = 10
    mkredgewidth = 2.5
    elinewidth = 2.5
    # capsize=7
    # capthick=2.0
    alpha = 1
    ylabel_coords = (-0.085, 0.5)
    
    wav_label = '$\mathbf{\\lambda} \\ \\left( \mathbf{\\mu}\mathrm{m} \\right)$'
    flux_label = '$\mathbf{F_{\\lambda}} \\ \\left( \mathrm{erg} \\ \mathrm{cm}\mathbf{^{-2}} \\ \mathrm{s}\mathbf{^{-1}} \\ \\AA\mathbf{^{-1}} \\right)$'
    
    obs_color = 'black'
    med_color = 'mediumblue' # (0.35, 0.55, 0.35)
    max_color = 'green' # '#2de37f'
    
    med_res_color = med_color
    max_res_color = max_color
    
    fill = 'white'
    
    hcolor = 'black'
    hstyle = '--'
    
    
    
    if singlefig:
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, gridspec_kw={'height_ratios':[3,1]}, figsize=(12,8))
        
        ax1.errorbar(
            wav, 
            obs_flux, 
            yerr=obs_flux_error, 
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
            med_flux, 
            yerr=med_flux_error,
            fmt='s',
            markersize=mkrsize,
            markeredgecolor=med_color, 
            markerfacecolor=fill,
            markeredgewidth=mkredgewidth,
            ecolor=med_color,
            elinewidth=elinewidth,
            label='Median',
            alpha=alpha,
            zorder=2
            )
        
        ax1.errorbar(
            wav, 
            max_flux, 
            yerr=max_flux_error,
            fmt='^',
            markersize=mkrsize,
            markeredgecolor=max_color, 
            markerfacecolor=fill,
            markeredgewidth=mkredgewidth,
            ecolor=max_color,
            elinewidth=elinewidth,
            label='Max-Likelihood',
            alpha=alpha,
            zorder=3
            )
        
        ax1.tick_params(top=False, bottom=False, labelbottom=False, labeltop=False, direction='inout', length=10)
        ax1.set_ylabel(flux_label)
        ax1.yaxis.set_label_coords(*ylabel_coords)
        
        # remove the errorbars from legend
        handles, labels = ax1.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        
        ax1.legend(handles, labels)
        
        
        ax2.axhline(y=0, c=hcolor, linestyle=hstyle, zorder=0)
        
        ax2.errorbar(
            wav, 
            med_frac_res, 
            yerr=med_frac_error, 
            fmt='s', 
            markersize=mkrsize, 
            markeredgecolor=med_res_color, 
            markerfacecolor=fill, 
            markeredgewidth=mkredgewidth,
            ecolor=med_res_color,
            elinewidth=elinewidth,
            label='Median',
            alpha=alpha,
            zorder=1
                      )
        
        ax2.errorbar(
            wav, 
            max_frac_res, 
            yerr=max_frac_error, 
            fmt='^', 
            markersize=mkrsize, 
            markeredgecolor=max_res_color, 
            markerfacecolor=fill, 
            markeredgewidth=mkredgewidth,
            ecolor=max_res_color,
            elinewidth=elinewidth,
            label='Max-Likelihood',
            alpha=alpha,
            zorder=2
                      )
        
        ax2.tick_params(top=True, direction='inout', length=10)
        ax2.set_xlabel(wav_label)
        ax2.set_ylabel('Fractional\nResidual')
        ax2.yaxis.set_label_coords(*ylabel_coords)
        
        
        
    else:
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, gridspec_kw={'height_ratios':[3,1]}, figsize=(24,8))
        
        ax1.errorbar(
            wav, 
            obs_flux, 
            yerr=obs_flux_error, 
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
            med_flux, 
            yerr=med_flux_error,
            fmt='s',
            markersize=mkrsize,
            markeredgecolor=med_color, 
            markerfacecolor=fill,
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
        
        ax1.legend(handles, labels)
        
        
        ax3.axhline(y=0, c=hcolor, linestyle=hstyle, zorder=0)
        
        ax3.errorbar(
            wav, 
            med_frac_res, 
            yerr=med_frac_error, 
            fmt='s', 
            markersize=mkrsize, 
            markeredgecolor=med_res_color, 
            markerfacecolor=fill, 
            markeredgewidth=mkredgewidth,
            ecolor=med_res_color,
            elinewidth=elinewidth,
            label='Median',
            alpha=alpha
                      )
        
        ax3.tick_params(top=True, direction='inout', length=10)
        ax3.set_xlabel(wav_label)
        ax3.set_ylabel('Fractional\nResidual')
        ax3.yaxis.set_label_coords(*ylabel_coords)
        
        res_ylim = ax3.get_ylim()
        
        
        ax2.errorbar(
            wav, 
            obs_flux, 
            yerr=obs_flux_error, 
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
        
        ax2.errorbar(
            wav, 
            max_flux, 
            yerr=max_flux_error,
            fmt='^',
            markersize=mkrsize,
            markeredgecolor=max_color, 
            markerfacecolor=fill,
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
        
        ax2.legend(handles, labels)
                
        
        ax4.axhline(y=0, c=hcolor, linestyle=hstyle, zorder=0)
        
        ax4.errorbar(
            wav, 
            max_frac_res, 
            yerr=max_frac_error, 
            fmt='^', 
            markersize=mkrsize, 
            markeredgecolor=max_res_color, 
            markerfacecolor=fill, 
            markeredgewidth=mkredgewidth,
            ecolor=max_res_color,
            elinewidth=elinewidth,
            label='Max-Likelihood',
            alpha=alpha
                      )
        
        ax4.tick_params(top=True, direction='inout', length=10)
        ax4.set_xlabel(wav_label)
        ax4.set_ylabel('Fractional\nResidual')
        ax4.yaxis.set_label_coords(*ylabel_coords)
        
        ax4.set_ylim(res_ylim)
        
        
        
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
    
    wav_label = '$\mathbf{\\lambda} \\ \\left( \mathbf{\\mu}\mathrm{m} \\right)$'
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








