# ISTA_rot_1

## mw_toolkit

A small Python package to model the spectrum of a microwave chain and of a laser
phase-modulated by it. A spectrum is a list of discrete tones (frequency, power,
origin, kind); every component takes spectra and returns a new one, so a setup is
written in the same order as the hardware is assembled. Harmonics, spurs, images, leakage and
intermodulation can be traced to their origin. Noise, phase noise and impedance
mismatch are not modelled.

| Module | Contents |
|---|---|
| `spectrum` | `Tone`, `Spectrum` (RF or optical, `table()`, `stages()`) |
| `sources` | `source` (incl. AWG images/aliases), `analog_source`, `pll_synthesizer` + `PLL_PRESETS` (Windfreak SynthHD) |
| `signal_path` | attenuator, filters (Butterworth or measured tables), diplexer, 90° hybrid, splitter/combiner, amplifier (P1dB, OIP2/3), mixer, IQ mixer (explicit I/Q phases) |
| `parts` | `PARTS`: measured datasheet tables of Mini-Circuits filters, diplexers and splitters |
| `optics` | `eom_laser_output`: laser sidebands from the RF spectrum (Bessel functions), `modulation_depth`, `rf_power_for_beta` |
| `plot_utils` | `plot_spectrum`, `plot_stages`, `plot_stage_rows`, `plot_spectra` |

A small example:

```python
from mw_toolkit import *

lo  = pll_synthesizer(-1.3, 9.4, preset='SynthHD v2', name='LO')
i   = source(4, 0.8, name='AWG')
q   = source(4, 0.8, phase_deg=-90, name='AWG')        # Q 90° behind I -> upper sideband
rf  = iq_mixer(lo, i, q, conv_loss_dB=8.5, lo_rf_isolation_dB=40)
rf  = filter_from_table(rf, part='ZVBP-10R5G-S+')
opt = eom_laser_output(rf, laser_power_mW=10, vpi_V=3.0, insertion_loss_dB=3.5)
plot_stages(rf); plot_spectrum(opt); opt.table(10)
```

The required dependecies are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt` in a python virtualenv
