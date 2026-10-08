"""Tone-based spectral cascade of a microwave chain.

Purpose: find where harmonics and spurs end up in a MW setup and how strong
they are relative to the input power. Phase noise, noise floor and
impedance mismatch are not modelled.

A Spectrum is a list of discrete tones (frequency in GHz, power in dBm).
Every component takes one or more Spectrum objects and returns a new one,
so a setup is modelled by chaining calls in the same order as the hardware:

    lo  = source(16, 6.0, harmonic_dBc=-30, name='LO')
    awg = source(0, 0.1, harmonic_dBc=-50, f_s_GHz=2.4, name='AWG')
    rf  = iq_mixer(lo, awg, conv_loss_dB=7, lo_rf_isolation_dB=35,
                   sideband_suppression_dBc=-30)
    rf  = amplifier(rf, gain_dB=20, p1db_dBm=20, oip3_dBm=30)
    rf  = bandpass(rf, 5.9, 6.3)
    rf.table()           # tones at the end of the chain, strongest first
    plot_spectrum(rf)    # spectrum at the end of the chain
    plot_stages(rf)      # spectrum after every component

Each tone remembers its origin (e.g. '(2·LO)+AWG') and its phase, so that
coherent paths (90° hybrids, splitters, combiners) add up correctly. Tones
of different origin at the same frequency are kept as separate entries.

Conventions: frequencies in GHz, powers in dBm, suppressions in dBc (sign is
ignored, a suppression always lowers the level). Gains / losses may be given
as a number or as a function of frequency f_GHz -> dB.

Modules
-------
spectrum     Tone and Spectrum data classes
sources      source (also AWG), analog_source, pll_synthesizer + PLL_PRESETS
signal_path  attenuators, filters, diplexer, hybrid, splitter/combiner,
             amplifier, mixers
parts        PARTS: measured datasheet tables of real filters / diplexers
optics       eom_laser_output: laser sidebands from the RF spectrum (phase EOM)
plot_utils   plot_spectrum, plot_stages, plot_spectra, plot_awg_spectrum
"""

from .spectrum import *
from .parts import *
from .sources import *
from .signal_path import *
from .optics import *
from .plot_utils import *
