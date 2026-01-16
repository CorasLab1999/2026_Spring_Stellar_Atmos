#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:29:30 2026

@author: entertainment
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

from astropy.modeling.models import BlackBody

import StellAst

#%% Part A: Create Planck Function

# Astropy function
# Documentation here: https://docs.astropy.org/en/stable/modeling/physical_models.html

T = 10_000 * u.K
A = 1# unitless, to get default units

bb = BlackBody(temperature = T, scale = A)

wavenums = np.linspace(0.0001, 12, 100) * u.micron**(-1)

specific_intensity = bb(wavenums.to(u.Hz, equivalencies=u.spectral()))

plt.figure(dpi=200)
plt.plot(wavenums, specific_intensity)
plt.title('astropy BB')
plt.show()
plt.clf()

# Create my own function

flux = StellAst.get_flux(wavenums, T)

plt.figure(dpi=200)
plt.plot(wavenums, flux)
plt.title('My Function')
plt.show()
plt.clf()


# Other function

#%% Part B: Bv vs v

#%% Part C: Temperature Curves