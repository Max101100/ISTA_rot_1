"""Library of real parts: measured insertion-loss tables from datasheets.

Each table is a list of (frequency in GHz, insertion loss in dB), copied from
the "Typical Performance Data" table of the datasheet (typical values at
25 °C). Between points the loss is interpolated linearly; outside the table
the first / last value is used, so tables should cover the frequencies your
tones reach.

Used by filter_from_table(), diplexer(part=...), diplexer_combine(part=...)
and power_splitter / power_combiner(part=...).
Add a part by copying an entry and filling in its datasheet table.
"""

import numpy as np

__all__ = ['PARTS']

PARTS = {
    # --- filters ---------------------------------------------------------------
    'VLF-1200+': dict(
        type='filter',
        description='Mini-Circuits lowpass, DC-1200 MHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/VLF-1200+.pdf',
        loss=[(0.05, 0.07), (0.25, 0.14), (0.7, 0.32), (1.2, 0.69), (1.4, 1.45),
              (1.53, 3.63), (1.6, 7.44), (1.7, 17.61), (1.865, 30.16), (2, 30.8),
              (3.5, 37.28), (5, 37.8), (6, 27), (6.2, 16.43), (7, 15.93)],
    ),
    'VLF-2500+': dict(
        type='filter',
        description='Mini-Circuits lowpass, DC-2400 MHz (loss < 1 dB), SMA',
        datasheet='https://www.minicircuits.com/pdfs/VLF-2500+.pdf',
        loss=[(0.05, 0.07), (0.5, 0.18), (1.5, 0.38), (2.5, 0.77), (2.9, 1.77),
              (3.075, 3.71), (3.25, 7.85), (3.45, 14.79), (3.675, 24.24),
              (3.8, 29.61), (5, 37.24), (6.1, 31.81), (8, 22.21), (15, 19.62),
              (20, 15.64)],
    ),
    'VLF-3800+': dict(
        type='filter',
        description='Mini-Circuits lowpass, DC-3900 MHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/VLF-3800+.pdf',
        loss=[(0.04, 0.01), (1.55, 0.28), (3.06, 0.47), (3.9, 0.65), (4.51, 1),
              (4.76, 1.87), (4.85, 2.6), (4.93, 3.55), (5.12, 7),
              (5.38, 14.48), (5.7, 30.38), (6, 30.58), (8.3, 36.33),
              (13, 35.19), (20, 10.55)],
    ),
    'ZLSS-11G-S+': dict(
        type='filter',
        description='Mini-Circuits suspended-substrate lowpass, DC-11000 MHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/ZLSS-11G-S+.pdf',
        loss=[(0.01, 0.01), (0.1, 0.03), (1, 0.14), (5, 0.45), (11, 1.37),
              (11.45, 3.13), (11.6, 6.11), (11.8, 11.73), (12.1, 20.41),
              (12.5, 30.86), (13, 42.54), (14.5, 71.91), (15, 80.48),
              (17.5, 103.93), (20, 99.68), (25, 114.63), (26.5, 105.28),
              (30, 95.97), (31.5, 87.12), (33, 108.78)],
    ),
    'ZVBP-10R7G-S+': dict(
        type='filter',
        description='Mini-Circuits cavity bandpass, 10450-10950 MHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/ZVBP-10R7G-S+.pdf',
        loss=[(0.1, 84.89), (0.5, 115.61), (3, 110.27), (9.3, 67.6), (10.065, 30.28),
              (10.18, 20.03), (10.265, 10.39), (10.33, 3.01), (10.45, 0.42),
              (10.6, 0.39), (10.7, 0.36), (10.8, 0.38), (10.95, 0.42), (11.08, 3.17),
              (11.145, 10.13), (11.24, 20.11), (11.365, 30.28), (12.3, 68.46),
              (15, 95.88), (20, 90.79)],
    ),
    # No measured table in the datasheet: values are the typical rejection of
    # its spec table (DC-8.25 GHz: 87 dB, 8.25-15: 37 dB, pass 16.45-17.45:
    # 0.42 dB, 18.75-23.2: 35 dB, 23.2-40: 78 dB); the slopes between the
    # bands are interpolated, not measured.
    'ZVBP-K16R95G+': dict(
        type='filter',
        description='Mini-Circuits cavity bandpass, 16450-17450 MHz, 2.92 mm',
        datasheet='https://www.minicircuits.com/pdfs/ZVBP-K16R95G+.pdf',
        loss=[(0.01, 87), (8.25, 87), (8.4, 37), (15, 37), (16.3, 10), (16.45, 0.42),
              (17.45, 0.42), (18, 10), (18.75, 35), (23.2, 35), (23.4, 78), (40, 78)],
    ),
    'ZVBP-10R5G-S+': dict(
        type='filter',
        description='Mini-Circuits cavity bandpass, 9750-11250 MHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/ZVBP-10R5G-S+.pdf',
        loss=[(0.1, 95.44), (0.5, 83.67), (3, 68.15), (5.95, 52.31),
              (8.2, 30.29), (8.8, 19.24), (9.4, 3.44), (9.45, 2.36),
              (9.75, 0.24), (10.5, 0.24), (11.25, 0.24), (11.65, 2.3),
              (11.7, 3.21), (12.5, 20.49), (13.2, 30.26), (15.1, 45.64),
              (16, 49.94), (17, 52.87), (17.5, 53.59), (18, 53.52)],
    ),

    # --- splitters / combiners (loss = excess over the ideal 10·log10(n) split, per port)
    'ZX10-2-183+': dict(
        type='splitter',
        n_ways=2,
        description='Mini-Circuits 2-way Wilkinson splitter/combiner, 1.5-18 GHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/ZX10-2-183+.pdf',
        loss=[[(1.5, 0.21), (2, 0.16), (4, 0.25), (5, 0.32), (6, 0.31),
               (7, 0.42), (8, 0.40), (9, 0.61), (10, 0.53), (11, 0.60),
               (12, 0.69), (14, 0.86), (15, 0.71), (16, 0.95), (18, 1.38)],
              [(1.5, 0.23), (2, 0.17), (4, 0.29), (5, 0.35), (6, 0.36),
               (7, 0.50), (8, 0.41), (9, 0.66), (10, 0.62), (11, 0.67),
               (12, 0.76), (14, 0.94), (15, 0.83), (16, 1.07), (18, 1.57)]],
    ),

    # --- diplexers (low = common <-> low pass port, high = common <-> high pass port)
    'ZDSS-5G6G-S+': dict(
        type='diplexer',
        description='Mini-Circuits diplexer, DC-5000 / 6000-20000 MHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/ZDSS-5G6G-S+.pdf',
        low=[(0.01, 0.01), (2.5, 0.59), (4, 0.68), (5, 1.09), (5.075, 1.22),
             (5.2, 1.78), (5.3, 3.34), (5.725, 18.38), (5.775, 21.29),
             (5.975, 30.87), (6, 32.01), (7.2, 73.53), (10, 95.22),
             (15, 99.02), (20, 101.08)],
        high=[(0.01, 81.59), (2.5, 63.37), (4, 53.55), (5, 35.2),
              (5.075, 30.65), (5.2, 21.16), (5.3, 12.75), (5.725, 3),
              (5.775, 2.7), (5.975, 2), (6, 1.92), (7.2, 0.83), (10, 0.83),
              (15, 0.8), (20, 0.61)],
    ),
    'ZDSS-3G4G-S+': dict(
        type='diplexer',
        description='Mini-Circuits diplexer, DC-3000 / 4000-20000 MHz, SMA',
        datasheet='https://www.minicircuits.com/pdfs/ZDSS-3G4G-S+.pdf',
        low=[(0.01, 0.01), (0.1, 0.05), (2, 0.47), (2.925, 0.91), (3, 1.02),
             (3.1, 1.23), (3.225, 3.27), (3.3, 9.61), (3.675, 20.91),
             (3.9, 29.73), (4, 33.65), (5, 54.81), (10, 105.28), (15, 102.76),
             (20, 83.64)],
        high=[(0.01, 86.6), (0.1, 94.84), (2, 62.02), (2.925, 29.9), (3, 26.22),
              (3.1, 19.66), (3.225, 9.16), (3.3, 5.5), (3.675, 1.27),
              (3.9, 0.71), (4, 0.72), (5, 0.58), (10, 0.43), (15, 0.32),
              (20, 0.55)],
    ),
}


def _table_loss(table):
    """Insertion-loss table [(f_GHz, dB), ...] -> function f_GHz -> dB."""
    f, dB = zip(*sorted(table))
    return lambda f_GHz: float(np.interp(f_GHz, f, dB))


def _part(name, kind):
    if name not in PARTS:
        raise KeyError(f'unknown part {name!r}; known: {", ".join(PARTS)}')
    if PARTS[name]['type'] != kind:
        raise ValueError(f'{name} is a {PARTS[name]["type"]}, not a {kind}')
    return PARTS[name]
