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

wavenums = np.linspace(0.0001, 12, 1000) * u.micron**(-1)

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

#%% Part B: Bv vs v

bb10 = BlackBody(temperature = 10_000*u.K, scale = A)
bb7 = BlackBody(temperature = 7_000*u.K, scale = A)
bb3 = BlackBody(temperature = 3_000*u.K, scale = A)


Bv10 = bb10(wavenums.to(u.Hz, equivalencies=u.spectral()))
Bv7 = bb7(wavenums.to(u.Hz, equivalencies=u.spectral()))
Bv3 = bb3(wavenums.to(u.Hz, equivalencies=u.spectral()))

plt.style.use(['classic', 'seaborn-notebook'])

# Bv vs wavenum
fig, ax = plt.subplots(dpi=200)
ax.plot(wavenums, Bv10, '-', color='red', alpha=1, lw=3, label='T = 10,000 K')
ax.plot(wavenums, Bv7, '-.', color='green', alpha=1, lw=3, label='T = 7,000 K')
ax.plot(wavenums, Bv3, '--' , color='blue', alpha=1, lw=3, label='T = 3,000 K')

ax.set(
    xscale="linear",
    xlabel=fr"$\tilde\nu$ [{wavenums.unit}]",
    ylabel=fr"$B_\nu (T)$ [{Bv10.unit}]",
)

plt.legend()

plt.savefig('images/plot_Bv_v.pdf')

plt.show()


#%% log Bv vs wavenum

fig, ax = plt.subplots(dpi=200)
ax.plot(wavenums, Bv10, '-', color='red', alpha=1, lw=3, label='T = 10,000 K')
ax.plot(wavenums, Bv7, '-.', color='green', alpha=1, lw=3, label='T = 7,000 K')
ax.plot(wavenums, Bv3, '--' , color='blue', alpha=1, lw=3, label='T = 3,000 K')

ax.set(
    xscale="linear",
    yscale='log',
    xlabel=fr"$\tilde\nu$ [{wavenums.unit}]",
    ylabel=fr"$B_\nu (T)$ [{Bv10.unit}]",
)

plt.ylim(bottom=1e-8, top=0.5e-3)

plt.legend(loc='top right')

plt.savefig('images/plot_logBv_v.pdf')

plt.show()

#%% log Bv vs log wavenum

fig, ax = plt.subplots(dpi=200)
ax.plot(wavenums, Bv10, '-', color='red', alpha=1, lw=3, label='T = 10,000 K')
ax.plot(wavenums, Bv7, '-.', color='green', alpha=1, lw=3, label='T = 7,000 K')
ax.plot(wavenums, Bv3, '--' , color='blue', alpha=1, lw=3, label='T = 3,000 K')

ax.set(
    xscale="log",
    yscale='log',
    xlabel=fr"$\tilde\nu$ [{wavenums.unit}]",
    ylabel=fr"$B_\nu (T)$ [{Bv10.unit}]",
)

plt.xlim(right=12)
plt.ylim(top=0.5e-3)

plt.savefig('images/plot_logBv_logv.pdf')

plt.legend(loc='lower left')

plt.show()



#%% Part C: Temperature Curves

# repeat all 3 curves, showing
# T = 10_000 K
# T = 7_000 K
# T = 3_000 K


















