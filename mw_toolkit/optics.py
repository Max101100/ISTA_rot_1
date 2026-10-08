"""Optical side: a laser phase-modulated by the RF spectrum in an EOM.

The RF spectrum (e.g. the combined output of the MW chain) drives a phase
electro-optic modulator. Every RF tone k gives a modulation depth

    V_peak = sqrt(2 · 50 Ω · P_RF)
    beta_k = pi · V_peak / V_pi(f_k)

and the optical field becomes (Jacobi-Anger)

    E_out = E_in · prod_k  sum_n J_n(beta_k) · exp(i n (2 pi f_k t + phi_k))

i.e. optical lines at nu_laser + sum_k n_k f_k with relative power
|prod_k J_{n_k}(beta_k)|^2. Lines from different RF combinations that land
on the same optical frequency add coherently.

The result is a Spectrum with domain='optical': f_GHz is the offset from
the laser frequency, P_dBm the optical power in that line.
"""

import warnings

import numpy as np
from scipy.special import jv

from .spectrum import Spectrum, Tone

__all__ = ['modulation_depth', 'rf_power_for_beta', 'eom_laser_output']


def _paren(origin):
    return origin if origin.isalnum() else f'({origin})'


def modulation_depth(P_dBm, vpi_V):
    """Phase modulation depth beta (rad) of an RF tone with power P_dBm
    into 50 Ω, for an EOM with half-wave voltage vpi_V."""
    V_peak = np.sqrt(2 * 50 * 10 ** ((P_dBm - 30) / 10))
    return np.pi * V_peak / vpi_V


def rf_power_for_beta(beta, vpi_V):
    """RF power (dBm into 50 Ω) that gives modulation depth beta.

    Useful targets: beta = 1.84 maximises the first sideband of a single
    tone (J1² = 34 %); beta = 1.08 on two tones gives both first sidebands
    the same, largest possible power (11.5 % each).
    """
    V_peak = beta * vpi_V / np.pi
    return 10 * np.log10(V_peak ** 2 / (2 * 50)) + 30


def _line_kind(terms):
    """Kind of an optical line from the RF tones that made it."""
    moving = [(n, t) for n, t in terms if n != 0]
    if not moving:
        return 'signal'                                  # laser carrier
    for _, t in moving:
        if t.kind != 'signal':
            return t.kind                                # sideband of a spur
    if len(moving) > 1:
        return 'intermod'                                # mixed sideband
    return 'signal' if abs(moving[0][0]) == 1 else 'harmonic'   # higher order


def eom_laser_output(rf_spectrum, laser_power_mW, vpi_V=3.0,
                     insertion_loss_dB=3.5, rf_loss_dB=0.0, max_order=None,
                     floor_dBc=-100, max_rf_dBm=None, label=None):
    """Optical spectrum of a laser phase-modulated by rf_spectrum in an EOM.

    laser_power_mW : optical power into the EOM.
    vpi_V : half-wave voltage V_pi, a number or a function f_GHz -> V
        (V_pi rises with frequency; take it from the EOM datasheet).
    insertion_loss_dB : optical insertion loss of the EOM.
    rf_loss_dB : RF cable / connector loss between the chain and the EOM.
    max_order : highest sideband order per RF tone. Default: automatic,
        ceil(beta) + 3, which keeps > 99.9 % of the power; tones with
        beta < 0.05 are limited to first order.
    floor_dBc : optical lines weaker than this (relative to the laser
        power) are dropped.
    max_rf_dBm : if given, warn when the total RF power exceeds it
        (the EOM's maximum RF input).

    Returns a Spectrum with domain='optical' (f_GHz = offset from the laser).
    """
    vpi = vpi_V if callable(vpi_V) else (lambda f: vpi_V)
    rf = [t for t in rf_spectrum if t.P_dBm > -150]
    if max_rf_dBm is not None:
        total = 10 * np.log10(sum(10 ** ((t.P_dBm - rf_loss_dB) / 10) for t in rf))
        if total > max_rf_dBm:
            warnings.warn(f'total RF power {total:.1f} dBm exceeds the EOM limit '
                          f'of {max_rf_dBm:g} dBm')

    threshold = 10 ** (floor_dBc / 10)
    tones = sorted(((t, modulation_depth(t.P_dBm - rf_loss_dB, vpi(t.f_GHz)))
                    for t in rf), key=lambda tb: -tb[1])

    # offset -> [complex amplitude, |amplitude| of strongest path, its terms]
    lines = {0.0: [1.0 + 0j, 1.0, ()]}
    for tone, beta in tones:
        if beta < 0.05:
            m = 1
        else:
            m = max_order if max_order is not None else int(np.ceil(beta)) + 3
        phase = np.deg2rad(tone.phase_deg)
        new = {}
        for off, (a, _, terms) in lines.items():
            for n in range(-m, m + 1):
                amp = a * jv(n, beta) * np.exp(1j * n * phase)
                if abs(amp) ** 2 < threshold:
                    continue
                key = round(off + n * tone.f_GHz, 9)
                path = terms + ((n, tone),)
                if key in new:
                    entry = new[key]
                    entry[0] += amp
                    if abs(amp) > entry[1]:
                        entry[1], entry[2] = abs(amp), path
                else:
                    new[key] = [amp, abs(amp), path]
        lines = new

    P0 = laser_power_mW * 10 ** (-insertion_loss_dB / 10)
    out = []
    for off, (a, _, terms) in lines.items():
        if abs(a) == 0:
            continue
        moving = [(n, t) for n, t in terms if n != 0]
        origin = ('laser' if not moving else
                  ' '.join(f'{n:+d}·{_paren(t.origin)}' for n, t in moving))
        out.append(Tone(off, 10 * np.log10(P0 * abs(a) ** 2), origin,
                        _line_kind(terms), np.rad2deg(np.angle(a))))

    if label is None:
        v = '' if callable(vpi_V) else f', V_pi {vpi_V:g} V'
        label = f'EOM output ({laser_power_mW:g} mW laser{v}, IL {insertion_loss_dB:g} dB)'
    return Spectrum(out, label, (rf_spectrum,), domain='optical')
