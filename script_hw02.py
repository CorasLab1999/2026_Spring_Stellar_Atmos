#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:51:18 2026

@author: cad6543
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from astropy import units as u
from astropy import constants as const

from astropy.modeling.models import BlackBody

import StellAst

plt.style.use(['classic', 'seaborn-v0_8-notebook'])

#%% Setup

x = np.linspace(0, 100, 50)
y = np.sin(x)

#%% Check my integrator

int_canned = np.trapezoid(y, x)
int_stell = StellAst.int_trapz(y, x)

print(np.abs(int_canned- int_stell))

# Integrators differ on the order of 10^-16. Let's call this agreement.

#%% Wrapper function

bb = BlackBody(temperature=7500*u.K, scale=1)

bb_int = StellAst.int_func(bb, 0.0001*(1/u.micron), 12*(1/u.micron), 0.1*(1/u.micron))

# ok it works :D

#%% Test convergence with SB (part b)


# plot BB

x_bb = np.linspace(0.00001, 100, 1000) * (1/u.micron)

fig, ax = plt.subplots(dpi=200)
ax.plot(x_bb, bb(x_bb), '-', color='red', alpha=1, lw=3, label='T = 7,500 K')

ax.set(
    xscale="linear",
    xlabel=fr"$\tilde\nu$ [{x_bb.unit}]",
    ylabel=fr"$B_\nu (T)$ [{bb(x_bb).unit}]",
)

plt.legend()

plt.savefig('images/plot_Bv_v.pdf')

plt.show()

# So we see that convergence should happen much before 100 1/um

# The integration of Planck is SB
# Use wikipedia formula https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law#

# Help with understanding units from https://people.tamu.edu/~kevinkrisciunas/planck.pdf

SB_T = const.sigma_sb * (7500*u.K)**4 / (np.pi*u.sr * const.c)
print(SB_T)
print(bb_int.to(SB_T.unit, equivalencies=u.spectral()) )

truth = SB_T
calc = bb_int.to(SB_T.unit, equivalencies=u.spectral())
precision = np.abs(1 - (calc/truth).to(u.dimensionless_unscaled))
print(precision)

const = 5 * bb_int.unit

#%% Precision vs x_min

x_min_arr = np.logspace(-3, 1, 50, base=10)

precision_x_min = np.zeros(len(x_min_arr))

for i, x_min in enumerate(x_min_arr):
    
    bb_int = StellAst.int_func(bb, x_min*(1/u.micron), 20*(1/u.micron), 0.1*(1/u.micron))
    
    calc = bb_int.to(SB_T.unit, equivalencies=u.spectral())
    
    precision = np.abs(1 - (calc/truth).to(u.dimensionless_unscaled))
    
    precision_x_min[i] = precision
    

fig, ax = plt.subplots(dpi=200)
ax.scatter(x_min_arr, precision_x_min, color='red')

ax.set_xlabel(fr"Minimum $\tilde\nu$")
ax.set_ylabel("Precision (%)")

ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 11)

#plt.savefig('images/plot_prec_xmin.pdf')

plt.show()

#%% with const err

x_min_arr = np.logspace(-3, 1, 50, base=10)

precision_x_min = np.zeros(len(x_min_arr))

for i, x_min in enumerate(x_min_arr):
    
    bb_int = StellAst.int_func(bb, x_min*(1/u.micron), 20*(1/u.micron), 0.1*(1/u.micron)) + const
    
    calc = bb_int.to(SB_T.unit, equivalencies=u.spectral())
    
    precision = np.abs(1 - (calc/truth).to(u.dimensionless_unscaled))
    
    precision_x_min[i] = precision
    

fig, ax = plt.subplots(dpi=200)
ax.scatter(x_min_arr, precision_x_min, color='red')

ax.set_xlabel(fr"Minimum $\tilde\nu$")
ax.set_ylabel("Precision (%) + Const Err")

#ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 11)

#plt.savefig('images/plot_prec_xmin_err.pdf')

plt.show()



#%% Precision vs x_max

x_max_arr = np.logspace(-1, np.log10(50), 100, base=10)

precision_x_max = np.zeros(len(x_max_arr))

for i, x_max in enumerate(x_max_arr):
    
    bb_int = StellAst.int_func(bb, 0.001*(1/u.micron), x_max*(1/u.micron), 0.1*(1/u.micron))
    
    calc = bb_int.to(SB_T.unit, equivalencies=u.spectral())
    
    precision = np.abs(1 - (calc/truth).to(u.dimensionless_unscaled))
    
    precision_x_max[i] = precision
    

fig, ax = plt.subplots(dpi=200)
ax.scatter(x_max_arr, precision_x_max, color='blue')

ax.set_xlabel(fr"Maximum $\tilde\nu$")
ax.set_ylabel("Precision (%)")

ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 51)

#plt.savefig('images/plot_prec_xmax.pdf')

plt.show()

#%% with const err

x_max_arr = np.logspace(-1, np.log10(50), 100, base=10)

precision_x_max = np.zeros(len(x_max_arr))

for i, x_max in enumerate(x_max_arr):
    
    bb_int = StellAst.int_func(bb, 0.001*(1/u.micron), x_max*(1/u.micron), 0.1*(1/u.micron)) + const
    
    calc = bb_int.to(SB_T.unit, equivalencies=u.spectral())
    
    precision = np.abs(1 - (calc/truth).to(u.dimensionless_unscaled))
    
    precision_x_max[i] = precision
    

fig, ax = plt.subplots(dpi=200)
ax.scatter(x_max_arr, precision_x_max, color='blue')

ax.set_xlabel(fr"Maximum $\tilde\nu$")
ax.set_ylabel("Precision (%)")

#ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 51)

#plt.savefig('images/plot_prec_xmax_err.pdf')

plt.show()

#%% Precision vs dx

dx_arr = np.logspace(-4, np.log10(5), 100, base=10)

precision_dx = np.zeros(len(dx_arr))

for i, dx in enumerate(dx_arr):
    
    bb_int = StellAst.int_func(bb, 0.001*(1/u.micron), 20*(1/u.micron), dx*(1/u.micron))
    
    calc = bb_int.to(SB_T.unit, equivalencies=u.spectral())
    
    precision = np.abs(1 - (calc/truth).to(u.dimensionless_unscaled))
    
    precision_dx[i] = precision
    

fig, ax = plt.subplots(dpi=200)
ax.scatter(dx_arr, precision_dx, color='green')

ax.set_xlabel(fr"$\Delta \tilde\nu$")
ax.set_ylabel("Precision (%)")

ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 5.1)

#plt.savefig('images/plot_prec_dx.pdf')

plt.show()

#%% with const err

dx_arr = np.logspace(-4, np.log10(5), 100, base=10)

precision_dx = np.zeros(len(dx_arr))

for i, dx in enumerate(dx_arr):
    
    bb_int = StellAst.int_func(bb, 0.001*(1/u.micron), 20*(1/u.micron), dx*(1/u.micron)) + const
    
    calc = bb_int.to(SB_T.unit, equivalencies=u.spectral())
    
    precision = np.abs(1 - (calc/truth).to(u.dimensionless_unscaled))
    
    precision_dx[i] = precision
    

fig, ax = plt.subplots(dpi=200)
ax.scatter(dx_arr, precision_dx, color='green')

ax.set_xlabel(fr"$\Delta \tilde\nu$")
ax.set_ylabel("Precision (%)")

#ax.set_ylim(-0.01, 1.01)
ax.set_xlim(0, 5.1)

plt.savefig('images/plot_prec_dx_err.pdf')

plt.show()

#%% Test convergence with parameters (part d)

# check difference in value from last point

#%% x_min

x_min_arr = np.logspace(-3, np.log10(2), 50, base=10)

bb_evals = np.zeros(len(x_min_arr))
conv_x_min = np.zeros(len(x_min_arr))

for i, x_min in enumerate(x_min_arr):
    
    bb_int = StellAst.int_func(bb, x_min*(1/u.micron), 20*(1/u.micron), 0.1*(1/u.micron))
    
    bb_evals[i] = bb_int.value
    
    if i > 0:
        conv = np.abs(bb_evals[i] - bb_evals[i-1])
    else:
        conv = 0
    
    conv_x_min[i] = conv
    
fig, ax = plt.subplots(dpi=200)
ax.scatter(x_min_arr, conv_x_min, color='red')

ax.set_xlabel(fr"Minimum $\tilde\nu$")
ax.set_ylabel(r"Convergence ($\Delta y$)")

#ax.set_ylim(-0.01, 1.01)
ax.set_xlim(1e-3, 2)
ax.set_xscale('log')

#plt.savefig('images/plot_conv_xmin.pdf')

plt.show()

#%% x_max

x_max_arr = np.logspace(-1, np.log10(50), 200, base=10)

bb_evals = np.zeros(len(x_max_arr))
conv_x_max = np.zeros(len(x_max_arr))

for i, x_max in enumerate(x_max_arr):
    
    bb_int = StellAst.int_func(bb, 0.01*(1/u.micron), x_max*(1/u.micron), 0.1*(1/u.micron))
    
    bb_evals[i] = bb_int.value
    
    if i > 0:
        conv = np.abs(bb_evals[i] - bb_evals[i-1])
    else:
        conv = 0
    
    conv_x_max[i] = conv
    
fig, ax = plt.subplots(dpi=200)
ax.plot(x_max_arr, conv_x_max, color='blue')

ax.set_xlabel(fr"Maximum $\tilde\nu$")
ax.set_ylabel(r"Convergence ($\Delta y$)")

#ax.set_ylim(bottom=-0.00001)
ax.set_xlim(1e-1, 50)
ax.set_xscale('log')

#plt.savefig('images/plot_conv_xmin.pdf')

plt.show()

#%% dx 

dx_arr = np.logspace(-4, np.log10(5), 100, base=10)

bb_evals = np.zeros(len(dx_arr))
conv_dx = np.zeros(len(dx_arr))

for i, dx in enumerate(dx_arr):
    
    bb_int = StellAst.int_func(bb, 0.01*(1/u.micron), 20*(1/u.micron), dx*(1/u.micron))
    
    bb_evals[i] = bb_int.value
    
    if i > 0:
        conv = np.abs(bb_evals[i] - bb_evals[i-1])
    else:
        conv = 0
    
    conv_dx[i] = conv
    
    
fig, ax = plt.subplots(dpi=200)
ax.plot(dx_arr, conv_dx, color='blue')

ax.set_xlabel(fr"$\Delta \tilde\nu$")
ax.set_ylabel(r"Convergence ($\Delta y$)")

#ax.set_ylim(bottom=-0.00001)
ax.set_xlim(1e-4, 5)
ax.set_xscale('log')

# plt.savefig('images/plot_conv_dx.pdf')

plt.show()












