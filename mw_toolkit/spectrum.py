"""Spectrum data classes: a Spectrum is a list of discrete Tones.

Conventions: frequencies in GHz, powers in dBm, suppressions in dBc (sign is
ignored, a suppression always lowers the level).
"""

from dataclasses import dataclass

import numpy as np

__all__ = ['FLOOR_DBM', 'Tone', 'Spectrum']

FLOOR_DBM = -200.0   # tones below this level are dropped


@dataclass(frozen=True)
class Tone:
    f_GHz: float
    P_dBm: float
    origin: str
    kind: str = 'signal'
    phase_deg: float = 0.0

    @property
    def amplitude(self):
        """Complex amplitude, |a|^2 = P in mW."""
        return np.sqrt(10 ** (self.P_dBm / 10)) * np.exp(1j * np.deg2rad(self.phase_deg))


class Spectrum:
    """A list of discrete tones: frequencies (GHz) and their powers (dBm).

    Attributes
    ----------
    tones : list of Tone
    label : str
        Name of the component that produced this spectrum.
    parents : tuple of Spectrum
        The input spectra of that component; used by plot_stages().
    domain : 'rf' or 'optical'
        'rf': f_GHz is an absolute frequency (> 0).
        'optical': f_GHz is the offset from the laser frequency; the laser
        carrier sits at 0 and lower sidebands at negative offsets.
    """

    def __init__(self, tones=(), label='', parents=(), domain='rf'):
        keep = (lambda t: t.f_GHz > 1e-12) if domain == 'rf' else (lambda t: True)
        self.tones = [t for t in tones if keep(t) and t.P_dBm > FLOOR_DBM]
        self.label = label
        self.parents = tuple(parents)
        self.domain = domain

    @classmethod
    def from_arrays(cls, f_GHz, P_dBm, label='', origin='measured', kind='signal'):
        """Build a spectrum from plain frequency / power arrays."""
        return cls([Tone(f, P, origin, kind) for f, P in zip(f_GHz, P_dBm)], label)

    # --- data access ---
    @property
    def f_GHz(self):
        return np.array([t.f_GHz for t in self.tones])

    @property
    def P_dBm(self):
        return np.array([t.P_dBm for t in self.tones])

    def __len__(self):
        return len(self.tones)

    def __iter__(self):
        return iter(self.tones)

    def __add__(self, other):
        """Put two spectra on one line without coherent addition (e.g. two sources)."""
        label = ' + '.join(l for l in (self.label, other.label) if l)
        return Spectrum(self.tones + other.tones, label, (self, other), self.domain)

    def __repr__(self):
        return f'Spectrum({self.label!r}, {len(self)} tones)'

    def reference(self):
        """Strongest signal tone (strongest tone if there is no signal)."""
        sig = [t for t in self.tones if t.kind == 'signal'] or self.tones
        return max(sig, key=lambda t: t.P_dBm)

    def stages(self):
        """All spectra leading up to this one, in signal-flow order."""
        seen, order = set(), []

        def visit(s):
            if id(s) in seen:
                return
            seen.add(id(s))
            for p in s.parents:
                visit(p)
            order.append(s)
        visit(self)
        return order

    # --- output ---
    def table(self, n=None, f_range_GHz=None):
        """Print the tones sorted by power. dBc is relative to reference()."""
        tones = sorted(self.in_range(f_range_GHz), key=lambda t: -t.P_dBm)[:n]
        ref = self.reference().P_dBm if self.tones else 0
        print(f'{self.label}')
        f_head = 'f (GHz)' if self.domain == 'rf' else 'offset GHz'
        print(f'{f_head:>10} {"P (dBm)":>9} {"P (dBc)":>9}  {"kind":<9} origin')
        for t in tones:
            print(f'{t.f_GHz:10.4f} {t.P_dBm:9.1f} {t.P_dBm - ref:9.1f}  {t.kind:<9} {t.origin}')

    def in_range(self, f_range_GHz):
        """Tones with f_range_GHz[0] <= f <= f_range_GHz[1] (all if None)."""
        if f_range_GHz is None:
            return list(self.tones)
        return [t for t in self.tones if f_range_GHz[0] <= t.f_GHz <= f_range_GHz[1]]


def _sup(x):
    """Suppression in dBc -> positive number of dB below the reference."""
    return abs(x)
