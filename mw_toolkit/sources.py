"""Frequency-generating components: generic CW source / AWG, analog signal
generator and PLL synthesizers (e.g. Windfreak SynthHD, via PLL_PRESETS)."""

import numpy as np

from .spectrum import Spectrum, Tone, _sup

__all__ = ['source', 'analog_source', 'pll_synthesizer', 'PLL_PRESETS']


def _fold(f_GHz, f_s_GHz):
    """Alias a frequency into the first Nyquist zone [0, f_s/2]."""
    f = f_GHz % f_s_GHz
    return f_s_GHz - f if f > f_s_GHz / 2 else f


def source(P_dBm, f_GHz, harmonic_dBc=None, n_harmonics=5, spurs=None,
           f_s_GHz=None, image_zones=4, phase_deg=0.0, name='src', label=None):
    """CW source (MW generator, AWG channel, LO, ...).

    harmonic_dBc : float or list, optional
        Harmonic suppression. A single number is used for every harmonic
        n = 2..n_harmonics, a list gives [2nd, 3rd, ...] individually.
    spurs : list of (f_GHz, dBc), optional
        Non-harmonic spurs at known frequencies, e.g. from a datasheet or a
        spectrum-analyser measurement.
    f_s_GHz : float, optional
        Sample rate of a DAC-based source (AWG). Enables two effects that
        only sampled sources have:
        - harmonics above Nyquist are folded back to |n*f0 - k*f_s|;
        - images of the carrier at k*f_s ± f_b (f_b = carrier folded into
          the 1st Nyquist zone), up to image_zones * f_s/2. Their level
          follows the sinc roll-off of a standard (NRZ) DAC, which relative
          to the carrier simplifies to 20*log10(f0 / f_image) dBc.
          An AWG output (reconstruction) filter is not included; model it
          with a lowpass() after the source.
    """
    tones = [Tone(f_GHz, P_dBm, name, 'signal', phase_deg)]

    if f_s_GHz is not None:
        f_b = _fold(f_GHz, f_s_GHz)
        for k in range(int(image_zones // 2) + 1):
            for s, sym in ((-1, '−'), (1, '+')):
                f = k * f_s_GHz + s * f_b
                if f <= 0 or f > image_zones * f_s_GHz / 2 or np.isclose(f, f_GHz):
                    continue
                origin = (f'{name} 1st zone' if k == 0
                          else f'{k if k > 1 else ""}f_s{sym}{name} image')
                tones.append(Tone(f, P_dBm + 20 * np.log10(f_GHz / f), origin,
                                  'image', s * phase_deg))

    if harmonic_dBc is not None:
        levels = (list(harmonic_dBc) if np.iterable(harmonic_dBc)
                  else [harmonic_dBc] * (n_harmonics - 1))
        for n, dBc in enumerate(levels, start=2):
            f, origin = n * f_GHz, f'{n}·{name}'
            if f_s_GHz is not None and f > f_s_GHz / 2:
                f, origin = _fold(f, f_s_GHz), f'{origin} alias'
            tones.append(Tone(f, P_dBm - _sup(dBc), origin, 'harmonic', n * phase_deg))

    for f, dBc in spurs or []:
        tones.append(Tone(f, P_dBm - _sup(dBc), f'{name} spur', 'spur'))

    return Spectrum(tones, label or f'{name}: {P_dBm:g} dBm @ {f_GHz:g} GHz')


def analog_source(P_dBm, f_GHz, harmonic_dBc=-30, n_harmonics=5,
                  nonharmonic_dBc=-70, spur_offsets_MHz=(), subharmonics=(),
                  phase_deg=0.0, name='Analog', label=None):
    """Analog lab signal generator (e.g. Keysight MXG/PSG, R&S SMB/SMA).

    The defaults are typical datasheet values for such instruments; replace
    them with the numbers for your generator and frequency band.

    harmonic_dBc : float or list
        As in source(): one value for all harmonics, or [2nd, 3rd, ...].
    nonharmonic_dBc, spur_offsets_MHz :
        Non-harmonic spurs at f ± offset. Generators specify these as a limit
        (e.g. < -70 dBc at > 10 kHz offset) without positions, so none are
        drawn unless you give offsets (e.g. from a spectrum-analyser check).
    subharmonics : list of (factor, dBc), optional
        Subharmonics at factor · f, copied from the "Subharmonics" line of
        the datasheet. They come from an internal frequency multiplier: in
        bands where the output is an oscillator at f/N multiplied by N, its
        leakage appears at k·f/N. Example for the R&S SMB100B at
        6-12 GHz ("0.5 × fout or 1.5 × fout < -60 dBc"):
            subharmonics=[(0.5, -60), (1.5, -60)]
        Leave empty in bands without a multiplier (SMB100B: f <= 3 GHz).
    """
    spurs = [(f_GHz + s * off / 1e3, nonharmonic_dBc)
             for off in spur_offsets_MHz for s in (-1, 1)]
    tones = source(P_dBm, f_GHz, harmonic_dBc, n_harmonics, spurs,
                   phase_deg=phase_deg, name=name).tones
    for factor, dBc in subharmonics:
        tones.append(Tone(factor * f_GHz, P_dBm - _sup(dBc),
                          f'{factor:g}·{name}', 'spur'))
    return Spectrum(tones, label or f'{name}: {P_dBm:g} dBm @ {f_GHz:g} GHz')


# Device presets for pll_synthesizer(). Harmonic levels are tables of
# (fundamental frequency in GHz, dBc), read off the "harmonic content with
# 0 dBm fundamental" graphs of the Windfreak datasheets (SynthHD v2 data sheet
# v0.1d, SynthHD v1.4 data sheet v0.2b), accurate to roughly ±3 dB.
# Add your own synthesizer (Valon, LMX2594 board, ...) as another entry.
PLL_PRESETS = {
    'SynthHD v2': dict(
        f_range_GHz=(0.01, 15.0),
        vco_range_GHz=(3.4, 6.8),       # divided below, doubled above
        reference_MHz=27.0,             # internal reference, 10 or 27 MHz
        loop_bw_kHz=30.0,
        harmonic_dBc=[
            # 2nd harmonic
            [(0.1, -11), (0.5, -10), (1, -11), (1.5, -14), (2, -11), (3, -11),
             (4, -12), (5, -12), (5.5, -14), (6, -15), (6.5, -18), (7, -17),
             (7.5, -15), (8, -17), (8.5, -20), (9, -21), (9.5, -17), (10, -12),
             (11, -11), (11.5, -12), (12, -15), (12.5, -18), (13, -18),
             (13.5, -35), (14, -27), (14.5, -34), (15, -22)],
            # 3rd harmonic
            [(0.1, -21), (0.5, -14), (1, -13), (1.3, -23), (2, -17),
             (2.5, -30), (3, -23), (3.5, -17), (4, -16), (5, -16), (6, -16),
             (6.5, -20), (7, -35), (7.5, -27), (8, -26), (8.5, -27), (9, -40),
             (9.5, -35), (10, -30), (10.5, -45), (11, -38), (11.5, -42),
             (12, -45), (12.5, -48), (13, -43), (13.5, -52), (14, -57),
             (15, -55)],
        ],
        subharmonic_dBc=None,   # none shown in the v2 datasheet
    ),
    'SynthHD v1.4': dict(
        f_range_GHz=(0.054, 13.6),
        vco_range_GHz=(3.4, 6.8),       # divided below, doubled above
        reference_MHz=27.0,             # internal reference, 10 or 27 MHz
        loop_bw_kHz=30.0,
        harmonic_dBc=[
            # 2nd harmonic
            [(0.3, -33), (0.45, -22), (0.8, -19), (1.15, -17), (1.5, -18),
             (2.2, -18), (2.9, -17), (3.25, -14), (3.6, -20), (3.95, -15),
             (4.1, -32), (4.3, -25), (4.65, -20), (5, -22), (5.35, -19),
             (5.7, -16), (6.05, -18), (6.4, -19), (6.75, -28), (7.1, -22),
             (7.45, -23), (7.8, -19), (8.15, -20), (8.5, -22), (8.85, -24),
             (9.2, -30), (9.4, -30), (9.55, -20), (9.9, -21), (10.25, -23),
             (10.95, -23), (11.3, -26), (11.65, -28), (12, -30), (12.35, -31),
             (12.7, -33), (13.05, -38), (13.4, -47)],
            # 3rd harmonic
            [(0.1, -15), (0.45, -11), (0.8, -11), (1.15, -13), (1.5, -17),
             (2.2, -17), (2.55, -27), (2.9, -30), (3.25, -34), (3.6, -22),
             (3.95, -25), (4.3, -20), (4.65, -15), (5, -22), (5.35, -22),
             (5.7, -25), (6.05, -27), (6.4, -24), (6.75, -31), (7.1, -32),
             (7.45, -34), (7.8, -36), (8.15, -44), (8.5, -50), (8.85, -57),
             (9.2, -67), (9.55, -55), (9.9, -58), (10.25, -60), (10.6, -55),
             (10.95, -65), (11.3, -63), (11.65, -64), (12, -67), (12.35, -65),
             (12.7, -60), (13.05, -62), (13.4, -50)],
        ],
        subharmonic_dBc=[(6.8, -9), (7.45, -6), (7.8, -2), (8.15, -3), (8.5, -3),
                     (9.2, -2), (9.55, -6), (9.9, -5), (10.25, -4), (10.6, 0),
                     (11.3, 0), (12, 0), (12.7, 1), (13.05, 3), (13.6, 5)],
    ),
}


def _level(x, f_GHz):
    """A dBc value given as a number or as a table [(f_GHz, dBc), ...]."""
    if np.iterable(x):
        f, dBc = zip(*x)
        return float(np.interp(f_GHz, f, dBc))
    return x


def pll_synthesizer(P_dBm, f_GHz, preset=None, harmonic_dBc=None,
                    subharmonic_dBc=None, vco_range_GHz=None,
                    f_range_GHz=None, reference_MHz=None, loop_bw_kHz=None,
                    boundary_spur_dBc=-50, phase_deg=0.0, name=None, label=None):
    """PLL synthesizer: a VCO locked to a reference, with an output divider
    below the VCO range and a frequency doubler above it.

    This covers the SynthHD and similar boards (Valon, ADF5355 / LMX2594
    based synthesizers, ...). preset loads the numbers of a known device from
    PLL_PRESETS, e.g. pll_synthesizer(18, 8.4, preset='SynthHD v2');
    any argument you pass explicitly overrides the preset.

    harmonic_dBc : list, one entry per harmonic [2nd, 3rd, ...]
        Each entry is a number or a table [(f_GHz, dBc), ...] over the
        fundamental frequency, as in datasheet graphs.
    subharmonic_dBc : number or table, optional
        Level of the f/2 subharmonic (doubler leakage) for outputs above
        the VCO range, in dBc *with sign* (it can be above the carrier:
        up to +5 dBc on the SynthHD v1.4). None: no subharmonic.
    vco_range_GHz : VCO range, default (3.4, 6.8) GHz as in ADF5355-type PLLs.
    reference_MHz, loop_bw_kHz, boundary_spur_dBc :
        Integer-boundary spurs. If the VCO frequency lies Δ away from a
        multiple of the reference, spurs appear at ±Δ around the carrier
        (scaled by the divider / doubler). Datasheets rarely give their
        level; boundary_spur_dBc inside the loop bandwidth, falling
        40 dB/decade outside it, is a typical fractional-N value. Spurs
        below -100 dBc are left out. reference_MHz=None: no boundary spurs.
    """
    given = dict(harmonic_dBc=harmonic_dBc, subharmonic_dBc=subharmonic_dBc,
                 vco_range_GHz=vco_range_GHz, f_range_GHz=f_range_GHz,
                 reference_MHz=reference_MHz, loop_bw_kHz=loop_bw_kHz)
    p = dict(vco_range_GHz=(3.4, 6.8), loop_bw_kHz=30.0)     # generic defaults
    if preset is not None:
        p.update(PLL_PRESETS[preset])
    p.update({k: v for k, v in given.items() if v is not None})
    name = name or preset or 'PLL'

    if p.get('f_range_GHz') is not None:
        f_min, f_max = p['f_range_GHz']
        if not f_min <= f_GHz <= f_max:
            raise ValueError(f'{preset or name} covers {f_min}-{f_max} GHz, '
                             f'got {f_GHz} GHz')

    tones = [Tone(f_GHz, P_dBm, name, 'signal', phase_deg)]
    for n, dBc in enumerate(p.get('harmonic_dBc') or [], start=2):
        tones.append(Tone(n * f_GHz, P_dBm - _sup(_level(dBc, f_GHz)),
                          f'{n}·{name}', 'harmonic', n * phase_deg))

    vco_min, vco_max = p['vco_range_GHz']
    if f_GHz > vco_max and p.get('subharmonic_dBc') is not None:
        tones.append(Tone(f_GHz / 2, P_dBm + _level(p['subharmonic_dBc'], f_GHz),
                          f'½·{name} (doubler)', 'spur'))

    # VCO frequency: output = VCO · 2 (doubler), or VCO / 2^k (divider)
    if p.get('reference_MHz') is not None:
        ref = p['reference_MHz']
        scale = 2.0 if f_GHz > vco_max else 1.0
        while f_GHz / scale < vco_min:
            scale /= 2
        f_vco_MHz = f_GHz / scale * 1e3
        offset_MHz = abs(f_vco_MHz - round(f_vco_MHz / ref) * ref)
        offset_out_GHz = offset_MHz * scale / 1e3
        rolloff = 40 * np.log10(max(offset_MHz * 1e3 / p['loop_bw_kHz'], 1))
        if offset_MHz > 0 and _sup(boundary_spur_dBc) + rolloff < 100:
            P_spur = P_dBm - _sup(boundary_spur_dBc) - rolloff
            for s in (-1, 1):
                tones.append(Tone(f_GHz + s * offset_out_GHz, P_spur,
                                  f'{name} int. boundary {s * offset_out_GHz * 1e6:+.0f} kHz',
                                  'spur'))

    return Spectrum(tones, label or f'{name}: {P_dBm:g} dBm @ {f_GHz:g} GHz')
