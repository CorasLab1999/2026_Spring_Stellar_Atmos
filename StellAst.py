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