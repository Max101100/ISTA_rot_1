"""Plotting of Spectrum objects."""

import numpy as np
from matplotlib import pyplot as plt

__all__ = ['KIND_COLORS', 'plot_spectrum', 'plot_stages', 'plot_stage_rows', 'plot_spectra',
           'plot_awg_spectrum']


# tone kinds, in the fixed order used for plotting colours
KIND_COLORS = {
    'signal':   '#2a78d6',
    'harmonic': '#eb6834',
    'spur':     '#1baf7a',
    'image':    '#eda100',
    'leakage':  '#e87ba4',
    'intermod': '#008300',
}


def plot_spectrum(spectrum, ax=None, f_range_GHz=None, floor=None, dBc=False,
         n_labels=6, title=None, legend=True):
    """Stem plot coloured by tone kind.

    dBc : plot relative to the strongest signal tone instead of in dBm.
    floor : bottom of the y axis in the plotted unit (dBm, or dBc if
        dBc=True); default 10 dB below the weakest tone.
    n_labels : the signal tone(s) and up to n_labels of the strongest
        other tones are labelled with their origin (overlapping labels
        are skipped).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    tones = spectrum.in_range(f_range_GHz)
    ref = spectrum.reference().P_dBm if (dBc and spectrum.tones) else 0
    y = lambda t: t.P_dBm - ref
    ax.set_title(spectrum.label if title is None else title)
    optical = spectrum.domain == 'optical'
    ax.set_xlabel('Offset from laser (GHz)' if optical else 'Frequency (GHz)')
    ax.set_ylabel('Power (dBc)' if dBc else 'Power (dBm)')
    ax.grid(alpha=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if not tones:
        return ax

    y_max = max(map(y, tones))
    if floor is None:
        floor = max(min(map(y, tones)) - 10, y_max - 120)

    for kind, color in KIND_COLORS.items():
        sel = [t for t in tones if t.kind == kind and y(t) > floor]
        if not sel:
            continue
        f = [t.f_GHz for t in sel]
        P = [y(t) for t in sel]
        ax.vlines(f, floor, P, color=color, lw=2)
        ax.plot(f, P, 'o', color=color, ms=6, ls='none', label=kind)

    if f_range_GHz is None:
        f_range_GHz = _default_ranges([spectrum])[spectrum.domain]
    f_lo, f_hi = f_range_GHz
    char_w = 0.009 * (f_hi - f_lo)                  # approx. label char width in GHz
    top = y_max + 0.22 * (y_max - floor)            # headroom for labels
    dy = 0.1 * (top - floor)
    others = sorted((t for t in tones if t.kind != 'signal' and y(t) > floor),
                    key=lambda t: -t.P_dBm)
    placed, n_other = [], 0
    for t in [t for t in tones if t.kind == 'signal'] + others:
        w = char_w * max(len(t.origin), 8) / 2
        if t.kind != 'signal':
            if n_other >= n_labels:
                break
            if any(abs(t.f_GHz - f) < w + pw and abs(y(t) - P) < dy
                   for f, P, pw in placed):
                continue                            # would overlap an existing label
            n_other += 1
        placed.append((t.f_GHz, y(t), w))
        ax.annotate(f'{t.origin}\n{t.f_GHz:g} GHz', (t.f_GHz, y(t)),
                    textcoords='offset points', xytext=(0, 6), ha='center',
                    va='bottom', fontsize=8)

    ax.set_ylim(floor, top)
    ax.set_xlim(*f_range_GHz)
    if legend:
        ax.legend(frameon=False, loc='upper right')
    return ax


def plot_stages(spectrum, f_range_GHz=None, dBc=False, floor=None, n_labels=4,
                height=3.2):
    """One subplot per component, from the sources to this spectrum."""
    return _stacked(spectrum.stages(), f_range_GHz, dBc, floor, n_labels, height)


def _default_ranges(spectra):
    """Common x range per domain: 0..f_max for RF, symmetric for optical."""
    ranges = {}
    rf = [t.f_GHz for s in spectra if s.domain == 'rf' for t in s]
    if rf:
        ranges['rf'] = (0, 1.05 * max(rf))
    opt = [abs(t.f_GHz) for s in spectra if s.domain == 'optical' for t in s]
    if opt:
        m = 1.05 * max(opt) or 1.0
        ranges['optical'] = (-m, m)
    return ranges


def _stacked(spectra, f_range_GHz, dBc, floor, n_labels, height):
    ranges = (_default_ranges(spectra) if f_range_GHz is None
              else {'rf': f_range_GHz, 'optical': f_range_GHz})
    one_domain = len({s.domain for s in spectra}) == 1
    fig, axes = plt.subplots(len(spectra), 1, figsize=(10, height * len(spectra)),
                             squeeze=False)
    first = {}                                  # share x only within a domain
    for s, ax in zip(spectra, axes[:, 0]):
        if s.domain in first:
            ax.sharex(first[s.domain])
        else:
            first[s.domain] = ax
        plot_spectrum(s, ax=ax, f_range_GHz=ranges[s.domain], dBc=dBc, floor=floor,
                      n_labels=n_labels, legend=False)
        if one_domain:
            ax.label_outer()
    kinds = {t.kind for s in spectra for t in s}
    handles = [plt.Line2D([], [], color=c, marker='o', ls='-', lw=2, label=k)
               for k, c in KIND_COLORS.items() if k in kinds]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (height * len(spectra))))
    return axes[:, 0]


def plot_stage_rows(rows, f_range_GHz=None, dBc=False, floor=None, n_labels=4,
                    height=3.2, width=7.0):
    """Plot a setup stage by stage, with several spectra side by side per stage.

    rows : dict {stage title: spectrum or list of spectra}, e.g.
        {'Stage 1: frequency generation': [lo, awg],
         'Stage 2: 90° hybrid':           [if_i, if_q],
         'Stage 3: IQ mixer':             rf}
        A list of lists works too (stages are then numbered).
    All panels share one frequency axis. Returns the axes as a list of rows,
    so you can draw on them afterwards, e.g. axes[0][1].axhline(-76).
    """
    if not isinstance(rows, dict):
        rows = {f'Stage {i + 1}': r for i, r in enumerate(rows)}
    rows = {title: list(r) if isinstance(r, (list, tuple)) else [r]
            for title, r in rows.items()}
    spectra = [s for r in rows.values() for s in r]
    ranges = (_default_ranges(spectra) if f_range_GHz is None
              else {'rf': f_range_GHz, 'optical': f_range_GHz})
    n_cols = max(len(r) for r in rows.values())

    fig = plt.figure(figsize=(width * n_cols, (height + 0.4) * len(rows)),
                     layout='constrained')
    subfigs = np.atleast_1d(fig.subfigures(len(rows), 1))
    axes, first = [], {}                    # share x only within a domain
    last_domain = list(rows.values())[-1][0].domain
    for i, (subfig, (title, row)) in enumerate(zip(subfigs, rows.items())):
        subfig.suptitle(title, fontweight='bold', ha='left', x=0.01)
        row_axes = subfig.subplots(1, n_cols, squeeze=False)[0]
        for j, ax in enumerate(row_axes):
            if j >= len(row):
                ax.remove()                     # fewer spectra in this stage
                continue
            if row[j].domain in first:
                ax.sharex(first[row[j].domain])
            else:
                first[row[j].domain] = ax
            plot_spectrum(row[j], ax=ax, f_range_GHz=ranges[row[j].domain], dBc=dBc,
                          floor=floor, n_labels=n_labels, legend=False)
            if j > 0:
                ax.set_ylabel('')
            if i < len(rows) - 1 and row[j].domain == last_domain:
                ax.set_xlabel('')               # the bottom row shows it
        axes.append(list(row_axes[:len(row)]))

    kinds = {t.kind for s in spectra for t in s}
    handles = [plt.Line2D([], [], color=c, marker='o', ls='-', lw=2, label=k)
               for k, c in KIND_COLORS.items() if k in kinds]
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles),
               frameon=False)
    return axes


def plot_spectra(*spectra, f_range_GHz=None, dBc=False, floor=None, n_labels=4,
                 height=3.2):
    """Plot several spectra (e.g. both diplexer ports) below each other."""
    return _stacked(spectra, f_range_GHz, dBc, floor, n_labels, height)


def plot_awg_spectrum(carrier_power_dBm, carrier_freq_GHz,
                      harmonic_suppression_dBc, nonharmonic_suppression_dBc,
                      n_harmonics=5, spur_freqs_GHz=None, f_max_GHz=None,
                      noise_floor_dBm=None, ax=None):
    """Plot the expected output spectrum of an AWG.

    Parameters
    ----------
    carrier_power_dBm : float
        Power of the carrier tone in dBm.
    carrier_freq_GHz : float
        Carrier frequency in GHz.
    harmonic_suppression_dBc : float
        Harmonic suppression in dBc (positive or negative sign accepted).
        Harmonics n*f0 (n = 2..n_harmonics) are drawn at P - |suppression|.
    nonharmonic_suppression_dBc : float
        Non-harmonic (spurious) suppression in dBc. Drawn as a dashed
        worst-case spur level and, if `spur_freqs_GHz` is given, as spurs
        at those frequencies.
    n_harmonics : int
        Highest harmonic order to show.
    spur_freqs_GHz : list of float, optional
        Frequencies of known non-harmonic spurs (e.g. images, clock leakage).
    f_max_GHz : float, optional
        Upper end of the frequency axis (default: just past the last harmonic).
    noise_floor_dBm : float, optional
        Bottom of the plot / noise floor (default: 20 dB below the lowest line).
    ax : matplotlib Axes, optional
    """
    harm_dBm = carrier_power_dBm - abs(harmonic_suppression_dBc)
    spur_dBm = carrier_power_dBm - abs(nonharmonic_suppression_dBc)

    if f_max_GHz is None:
        f_max_GHz = (n_harmonics + 0.5) * carrier_freq_GHz
    if noise_floor_dBm is None:
        noise_floor_dBm = min(harm_dBm, spur_dBm) - 20

    harm_freqs = carrier_freq_GHz * np.arange(2, n_harmonics + 1)
    harm_freqs = harm_freqs[harm_freqs <= f_max_GHz]
    spur_freqs = np.asarray([] if spur_freqs_GHz is None else spur_freqs_GHz, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    def lines(freqs, level, color, label, marker):
        if len(freqs) == 0:
            return
        ax.vlines(freqs, noise_floor_dBm, level, color=color, lw=2)
        ax.plot(freqs, np.full(len(freqs), level), marker, color=color,
                ms=8, ls='none', label=label)

    lines([carrier_freq_GHz], carrier_power_dBm, '#2a78d6',
          f'Carrier ({carrier_power_dBm:g} dBm)', 'o')
    lines(harm_freqs, harm_dBm, '#eb6834',
          f'Harmonics (−{abs(harmonic_suppression_dBc):g} dBc)', 's')
    lines(spur_freqs, spur_dBm, '#1baf7a',
          f'Non-harmonic spurs (−{abs(nonharmonic_suppression_dBc):g} dBc)', '^')
    ax.axhline(spur_dBm, color='#1baf7a', ls='--', lw=1,
               label=None if len(spur_freqs) else
               f'Non-harmonic spur limit (−{abs(nonharmonic_suppression_dBc):g} dBc)')

    # label harmonic orders
    ax.annotate(r'$f_0$', (carrier_freq_GHz, carrier_power_dBm),
                textcoords='offset points', xytext=(0, 8), ha='center')
    for n, f in enumerate(harm_freqs, start=2):
        ax.annotate(f'${n}f_0$', (f, harm_dBm), textcoords='offset points',
                    xytext=(0, 8), ha='center')

    ax.set_xlim(0, f_max_GHz)
    ax.set_ylim(noise_floor_dBm, carrier_power_dBm + 10)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power (dBm)')
    ax.set_title(f'AWG spectrum, carrier {carrier_freq_GHz:g} GHz @ {carrier_power_dBm:g} dBm')
    ax.grid(alpha=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc='upper right')
    return ax
