#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:30:37 2026

@author: CoraDeFrancesco
"""

# Stellar Astro module

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

from astropy.modeling.models import BlackBody

def int_func(func, x_min, x_max, dx):
    """
    Integrate a given function between x_min and x_max with spacing dx.
    Endpoints are handled in the following manner:
        x_min is always the first value in the variable of integration.
        The integrating variable is incremented by dx.
        The last variable of integration is the closest value to x_max without
        going over. This means that x_max is generally not included in the
        integration array.

    Parameters
    ----------
    func : method
        Function from which to generate y values.
    x_min : float
        Minimum variable of integration. Must have units.
    x_max : float
        Maximum variable of integration. Must have units.
    dx : float
        Spacing in x. Must have units compatible with limits.

    Returns
    -------
    out : float
        Area unter func between x_min and x_max.

    """
    
    n_x = ((x_max - x_min) / dx).to(u.dimensionless_unscaled).value
    x_arr = np.zeros(int(n_x))*dx.unit # do this to play nice with assigning
                                       # values with units to idxs inside x_arr
    for i in range(0, len(x_arr)):
        
        x_arr[i] = x_min + (i*dx)
        
    
    y_arr = func(x_arr)
    
    out = np.trapezoid(y_arr, x_arr)
    
    return out

def int_trapz(y, x):
    """
    Integration using the trapezoid rule.
    Formula from https://en.wikipedia.org/wiki/Trapezoidal_rule

    Parameters
    ----------
    y : arr
        Dependent variable.
    x : arr
        Independent variable of integration.

    Returns
    -------
    Area under y.

    """
    
    int_arr = np.zeros(len(x))
    for i in range(1, len(x)):
        delta_x = x[i] - x[i-1]
        int_arr[i] = delta_x * (y[i-1] + y[i])/2
    out = np.sum(int_arr)
    
    return(out)

def get_flux(wavenums, T):
    '''
    Calculate the flux in units of erg cm-1 Hz-1 s-1
    by integrating astropy BB (specific intensity) over
    solid angle.

    Parameters
    ----------
    wavenums : array or array-like
        Wavenumber in units (1/length).
    T : float
        temperature in units of Kelvin.

    Returns
    -------
    array
        Flux.

    '''
    
    # dummy check
    if (T.unit != 'K'):
        raise TypeError('T must have units of Kelvin')
    
    # get intensity
    bb = BlackBody(temperature = T, scale = 1)
    specific_intensity = bb(wavenums.to(u.Hz, equivalencies=u.spectral()))
    
    # integrate over solid angle
    F = specific_intensity * 4*np.pi*u.sr
    
    return F.to(u.erg * u.cm**(-2)* u.Hz**(-1) * u.s**(-1))