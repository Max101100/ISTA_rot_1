"""Components the signal passes through: attenuators, filters, diplexers,
hybrids, splitters / combiners, amplifiers and mixers.

Gains / losses may be given as a number or as a function f_GHz -> dB.
"""

from dataclasses import replace

import numpy as np

from .parts import _part, _table_loss
from .spectrum import Spectrum, Tone, _sup

__all__ = ['attenuator', 'butterworth_loss_dB', 'filter_', 'lowpass', 'highpass',
           'bandpass', 'filter_from_table', 'touchstone_component', 'diplexer',
           'diplexer_combine', 'hybrid_90', 'power_splitter', 'power_combiner',
           'amplifier', 'iq_mixer', 'mixer']


def _db(x, f_GHz):
    """Evaluate a gain/loss given as a number or as a function of frequency."""
    return x(f_GHz) if callable(x) else x


def _from_amplitude(template, a):
    P = 10 * np.log10(abs(a) ** 2) if abs(a) > 0 else -np.inf
    return replace(template, P_dBm=P, phase_deg=np.rad2deg(np.angle(a)))


def _kind(*tones):
    """Kind of a product: 'signal' only if every parent is a signal."""
    for t in tones:
        if t.kind != 'signal':
            return t.kind
    return 'signal'


def _paren(origin):
    return origin if origin.isalnum() else f'({origin})'


# --- linear components --------------------------------------------------------

def attenuator(spectrum, attenuation_dB, label=None):
    """Attenuator or cable: every tone loses attenuation_dB (number or f -> dB)."""
    if label is None:
        label = ('Attenuator' if callable(attenuation_dB)
                 else f'Attenuator {attenuation_dB:g} dB')
    return Spectrum([replace(t, P_dBm=t.P_dBm - _db(attenuation_dB, t.f_GHz))
                     for t in spectrum], label, (spectrum,))


def butterworth_loss_dB(f_GHz, kind, f_c_GHz, order=5):
    """Rejection (positive dB) of an ideal Butterworth filter.

    kind : 'lowpass', 'highpass', 'bandpass' or 'bandstop'.
    f_c_GHz : cutoff, or (f_low, f_high) for band filters.
    """
    f = np.asarray(f_GHz, dtype=float)
    if kind == 'lowpass':
        x = f / f_c_GHz
    elif kind == 'highpass':
        x = f_c_GHz / f
    elif kind in ('bandpass', 'bandstop'):
        f1, f2 = f_c_GHz
        f0 = np.sqrt(f1 * f2)
        x = (f / f0 - f0 / f) * f0 / (f2 - f1)
        if kind == 'bandstop':
            x = 1 / x
    else:
        raise ValueError(f'unknown filter kind {kind!r}')
    return 10 * np.log10(1 + np.abs(x) ** (2 * order))


def filter_(spectrum, kind, f_c_GHz, order=5, insertion_loss_dB=0.5,
            max_rejection_dB=80, label=None):
    """Ideal Butterworth filter with insertion loss and a finite stopband.

    kind : 'lowpass', 'highpass', 'bandpass' or 'bandstop'.
    f_c_GHz : cutoff (3 dB) frequency, or (f_low, f_high) for band filters.
    max_rejection_dB caps the rejection, as real filters leak (typ. 50-80 dB).
    """
    def loss(f):
        rej = min(butterworth_loss_dB(f, kind, f_c_GHz, order), max_rejection_dB)
        return insertion_loss_dB + rej
    if label is None:
        fc = ('{:g}–{:g}'.format(*f_c_GHz) if np.iterable(f_c_GHz)
              else f'{f_c_GHz:g}')
        label = f'{kind.capitalize()} {fc} GHz'
    return attenuator(spectrum, loss, label)


def lowpass(spectrum, f_c_GHz, order=5, insertion_loss_dB=0.5,
            max_rejection_dB=80, label=None):
    return filter_(spectrum, 'lowpass', f_c_GHz, order, insertion_loss_dB,
                   max_rejection_dB, label)


def highpass(spectrum, f_c_GHz, order=5, insertion_loss_dB=0.5,
             max_rejection_dB=80, label=None):
    return filter_(spectrum, 'highpass', f_c_GHz, order, insertion_loss_dB,
                   max_rejection_dB, label)


def bandpass(spectrum, f_low_GHz, f_high_GHz, order=5, insertion_loss_dB=1.0,
             max_rejection_dB=80, label=None):
    """Butterworth bandpass with 3 dB edges at f_low_GHz and f_high_GHz."""
    return filter_(spectrum, 'bandpass', (f_low_GHz, f_high_GHz), order,
                   insertion_loss_dB, max_rejection_dB, label)


def touchstone_component(spectrum, path, port_out=2, port_in=1, label=None):
    """Apply measured S-parameters |S_out,in(f)| (and phase) from a Touchstone file.

    Needs scikit-rf (pip install scikit-rf).
    """
    import skrf
    ntw = skrf.Network(path)
    s = ntw.s[:, port_out - 1, port_in - 1]
    f = ntw.f / 1e9
    out = []
    for t in spectrum:
        if not f[0] <= t.f_GHz <= f[-1]:
            raise ValueError(f'{t.f_GHz} GHz is outside the data of {path}')
        st = np.interp(t.f_GHz, f, s.real) + 1j * np.interp(t.f_GHz, f, s.imag)
        out.append(_from_amplitude(t, t.amplitude * st))
    return Spectrum(out, label or f'S{port_out}{port_in} of {path}', (spectrum,))


def filter_from_table(spectrum, part=None, table=None, label=None):
    """Filter (or any two-port) with a measured insertion-loss table.

    part : name of a filter in PARTS, e.g. 'ZVBP-10R5G-S+', or
    table : your own [(f_GHz, loss_dB), ...], e.g. from a datasheet or a
        network-analyser measurement.
    """
    if part is not None:
        table = _part(part, 'filter')['loss']
    if table is None:
        raise ValueError('give part= or table=')
    return attenuator(spectrum, _table_loss(table), label or part or 'Filter (table)')


def _diplexer_losses(f_cross_GHz, order, insertion_loss_dB, max_rejection_dB, part):
    """Low / high port loss as functions of f_GHz, plus labels."""
    if part is not None:
        p = _part(part, 'diplexer')
        return (_table_loss(p['low']), _table_loss(p['high']), part,
                f'{part} low port', f'{part} high port')
    if f_cross_GHz is None:
        raise ValueError('give f_cross_GHz= or part=')

    def port(kind):
        return lambda f: insertion_loss_dB + min(
            butterworth_loss_dB(f, kind, f_cross_GHz, order), max_rejection_dB)
    return (port('lowpass'), port('highpass'), f'Diplexer {f_cross_GHz:g} GHz',
            f'Diplexer low port (< {f_cross_GHz:g} GHz)',
            f'Diplexer high port (> {f_cross_GHz:g} GHz)')


def diplexer(spectrum, f_cross_GHz=None, order=5, insertion_loss_dB=0.5,
             max_rejection_dB=60, part=None):
    """Diplexer used as a splitter: returns (low_port, high_port) spectra.

    Either an ideal Butterworth lowpass / highpass pair sharing the crossover
    f_cross_GHz (isolation capped at max_rejection_dB), or a real part from
    PARTS with its measured tables, e.g. part='ZDSS-5G6G-S+'.
    """
    low_loss, high_loss, _, low_label, high_label = _diplexer_losses(
        f_cross_GHz, order, insertion_loss_dB, max_rejection_dB, part)
    return (attenuator(spectrum, low_loss, low_label),
            attenuator(spectrum, high_loss, high_label))


def diplexer_combine(low_in, high_in, f_cross_GHz=None, order=5,
                     insertion_loss_dB=0.5, max_rejection_dB=60, part=None,
                     label=None):
    """Diplexer used as a combiner: low_in into the low pass port, high_in
    into the high pass port, returns the spectrum at the common port.

    Each input is filtered by its port, so e.g. the harmonics of the low-band
    signal are removed before they reach the output. Same filter options as
    diplexer(): f_cross_GHz (ideal Butterworth) or part='ZDSS-5G6G-S+'.
    """
    low_loss, high_loss, name, _, _ = _diplexer_losses(
        f_cross_GHz, order, insertion_loss_dB, max_rejection_dB, part)
    low = attenuator(low_in, low_loss)
    high = attenuator(high_in, high_loss)
    return Spectrum(_coherent_sum((low, 1), (high, 1)), label or name,
                    (low_in, high_in))


# --- coherent splitting / combining --------------------------------------------

def _coherent_sum(*weighted):
    """Sum spectra with complex weights; tones with equal (f, origin) add coherently."""
    acc = {}
    for spectrum, w in weighted:
        for t in spectrum or []:
            key = (round(t.f_GHz, 9), t.origin)
            a = t.amplitude * w
            acc[key] = (acc[key][0] + a, acc[key][1]) if key in acc else (a, t)
    return [_from_amplitude(t, a) for a, t in acc.values()]


def hybrid_90(in_1, in_2=None, insertion_loss_dB=0.3, amp_imbalance_dB=0.0,
              phase_imbalance_deg=0.0):
    """90° (quadrature) hybrid coupler. Returns (out_0deg, out_90deg).

    As a splitter (in_2=None): out_0deg gets in_1 at -3 dB and 0°,
    out_90deg gets in_1 at -3 dB and -90°.
    As a combiner: each output is the coherent sum
        out_0deg  = (in_1 + in_2·e^{-j90°}) / √2
        out_90deg = (in_1·e^{-j90°} + in_2) / √2
    so tones of equal origin add or cancel depending on their phases.

    amp_imbalance_dB / phase_imbalance_deg are applied to the -90° path.
    """
    il = 10 ** (-insertion_loss_dB / 20) / np.sqrt(2)
    quad = (10 ** (amp_imbalance_dB / 20)
            * np.exp(-1j * np.deg2rad(90 + phase_imbalance_deg)))
    parents = (in_1,) if in_2 is None else (in_1, in_2)
    out_0 = Spectrum(_coherent_sum((in_1, il), (in_2, il * quad)),
                     '90° hybrid, 0° port', parents)
    out_90 = Spectrum(_coherent_sum((in_1, il * quad), (in_2, il)),
                      '90° hybrid, 90° port', parents)
    return out_0, out_90


def _splitter_losses(n_ways, insertion_loss_dB, part):
    """Per-port excess loss functions f_GHz -> dB, plus n_ways and a name."""
    if part is not None:
        p = _part(part, 'splitter')
        return p['n_ways'], [_table_loss(t) for t in p['loss']], part
    return n_ways, [lambda f: _db(insertion_loss_dB, f)] * n_ways, None


def power_splitter(spectrum, n_ways=2, insertion_loss_dB=0.5, label=None, part=None):
    """In-phase (Wilkinson) splitter: returns n_ways copies.

    insertion_loss_dB : excess loss on top of the 10·log10(n) split, a number
        or a function f_GHz -> dB (e.g. from the datasheet), or
    part : a splitter from PARTS, e.g. 'ZX10-2-183+' (measured loss per port).
    """
    n_ways, losses, name = _splitter_losses(n_ways, insertion_loss_dB, part)
    name = label or name or 'Splitter'
    return [attenuator(spectrum,
                       lambda f, loss=loss: 10 * np.log10(n_ways) + loss(f),
                       f'{name} port {i + 1}/{n_ways}') for i, loss in enumerate(losses)]


def power_combiner(*spectra, insertion_loss_dB=0.5, label=None, part=None):
    """In-phase (Wilkinson) combiner: coherent sum / √N.

    insertion_loss_dB : excess loss, a number or a function f_GHz -> dB, or
    part : a splitter from PARTS used as combiner, e.g. 'ZX10-2-183+'
        (input i gets the measured loss of port i).
    """
    n, losses, name = _splitter_losses(len(spectra), insertion_loss_dB, part)
    if n != len(spectra):
        raise ValueError(f'{part} combines {n} inputs, got {len(spectra)}')
    lossy = [attenuator(s, loss) for s, loss in zip(spectra, losses)]
    w = 1 / np.sqrt(len(spectra))
    return Spectrum(_coherent_sum(*[(s, w) for s in lossy]),
                    label or (f'{name} combiner' if name else f'{len(spectra)}-way combiner'),
                    spectra)


# --- nonlinear components ------------------------------------------------------

def amplifier(spectrum, gain_dB, p1db_dBm=None, oip2_dBm=None, oip3_dBm=None,
              nonlinear_range_dB=40, max_nonlinear_tones=6, label=None):
    """Amplifier with gain, soft compression and 2nd/3rd-order distortion.

    gain_dB : float or f_GHz -> dB.
    p1db_dBm : output 1 dB compression point. The total output power is
        compressed with a Rapp model (smoothness 2); all tones are reduced
        by the same amount.
    oip2_dBm, oip3_dBm : output intercept points. Distortion products are
        computed from the output tones using the standard intercept rules
            HD2   = 2·P - OIP2 - 6 dB          at 2f
            HD3   = 3·P - 2·OIP3 - 9.5 dB      at 3f
            IM2   = P1 + P2 - OIP2             at f1 ± f2
            IM3   = 2·P1 + P2 - 2·OIP3         at 2f1 ± f2
        (the 6 / 9.5 dB offsets follow from a memoryless polynomial).
        Only tones within nonlinear_range_dB of the strongest output tone,
        and at most max_nonlinear_tones of them, take part.
    """
    out = [replace(t, P_dBm=t.P_dBm + _db(gain_dB, t.f_GHz)) for t in spectrum]

    if p1db_dBm is not None and out:
        p = 2
        x1 = (10 ** (p / 10) - 1) ** (1 / (2 * p))       # V_lin / V_sat at P1dB
        p_sat = p1db_dBm + 1 - 20 * np.log10(x1)
        p_tot = 10 * np.log10(sum(10 ** (t.P_dBm / 10) for t in out))
        x = 10 ** ((p_tot - p_sat) / 20)
        compression = 10 * np.log10(1 + x ** (2 * p)) / p
        out = [replace(t, P_dBm=t.P_dBm - compression) for t in out]

    strong = sorted(out, key=lambda t: -t.P_dBm)
    strong = [t for t in strong[:max_nonlinear_tones]
              if t.P_dBm > strong[0].P_dBm - nonlinear_range_dB] if strong else []

    products = []
    if oip2_dBm is not None:
        for t in strong:
            products.append(Tone(2 * t.f_GHz, 2 * t.P_dBm - oip2_dBm - 6.02,
                                 f'2·{_paren(t.origin)}', 'harmonic', 2 * t.phase_deg))
        for i, a in enumerate(strong):
            for b in strong[i + 1:]:
                P = a.P_dBm + b.P_dBm - oip2_dBm
                products += [
                    Tone(a.f_GHz + b.f_GHz, P, f'{_paren(a.origin)}+{_paren(b.origin)}',
                         'intermod', a.phase_deg + b.phase_deg),
                    Tone(abs(a.f_GHz - b.f_GHz), P, f'{_paren(a.origin)}−{_paren(b.origin)}',
                         'intermod', a.phase_deg - b.phase_deg)]
    if oip3_dBm is not None:
        for t in strong:
            products.append(Tone(3 * t.f_GHz, 3 * t.P_dBm - 2 * oip3_dBm - 9.54,
                                 f'3·{_paren(t.origin)}', 'harmonic', 3 * t.phase_deg))
        for a in strong:
            for b in strong:
                if a is b:
                    continue
                P = 2 * a.P_dBm + b.P_dBm - 2 * oip3_dBm
                for s, sym in ((1, '+'), (-1, '−')):
                    products.append(Tone(abs(2 * a.f_GHz + s * b.f_GHz), P,
                                         f'2·{_paren(a.origin)}{sym}{_paren(b.origin)}',
                                         'intermod', 2 * a.phase_deg + s * b.phase_deg))

    if label is None:
        label = ('Amplifier' if callable(gain_dB) else f'Amplifier {gain_dB:+g} dB')
    # mark distortion created here, so it can be told apart from input tones at
    # the same frequency (e.g. a source harmonic vs. the amplifier's own one)
    products = [replace(t, origin=f'{t.origin} [{label}]') for t in products]
    return Spectrum(out + products, label, (spectrum,))


def iq_mixer(lo, if_, q_in=None, conv_loss_dB=7.0, sideband='upper',
             sideband_suppression_dBc=None, amp_imbalance_dB=0.0,
             phase_imbalance_deg=0.0, lo_rf_isolation_dB=35.0,
             if_rf_isolation_dB=None, spur_table=None, conv_loss_ref='hybrid',
             floor_dBm=-150, label=None, q_phase_deg=None, q_gain_dB=0.0):
    """IQ mixer (single-sideband upconverter), linear model.

    Two ways to use it:

    1. Explicit I and Q ports (physical model):
           iq_mixer(lo, i_in, q_in)                 # e.g. two AWG channels
           iq_mixer(lo, i_in, q_phase_deg=-90)      # Q = I rotated by -90°
       if_ is the spectrum at the I port, q_in the one at the Q port; the
       phase of each tone is taken from the spectra (source(..., phase_deg=),
       hybrid_90 outputs, ...). q_phase_deg / q_gain_dB rotate / scale the Q
       port on top (or create Q from I if q_in is None). Tones of I and Q are
       paired by frequency and origin, so give both AWG channels the same
       name. For every IF tone the ideal mixer RF = I·cos(w_LO t) - Q·sin(w_LO t)
       gives the sideband phasors
           USB (f_LO + f) = (a_I + j·a_Q) / 2,   LSB (f_LO - f) = (a_I* + j·a_Q*) / 2
       so Q 90° behind I -> upper sideband only, Q 90° ahead -> lower only,
       Q in phase -> both at half power (total power is conserved).
       amp_imbalance_dB / phase_imbalance_deg are the mixer's own I/Q
       imbalance, applied to the Q path. `sideband` and
       sideband_suppression_dBc are not used: the phases decide.

    2. One IF port (behavioural model, as before): iq_mixer(lo, if_) with
       `sideband` and sideband_suppression_dBc (or the imbalance).

    lo : spectrum at the LO port. The strongest tone is taken as the LO;
        every other LO tone (harmonics, spurs) is mixed as well, at its own
        level relative to the LO.
    lo_rf_isolation_dB : LO leakage (carrier feedthrough) = P_LO - isolation.
    if_rf_isolation_dB : IF feedthrough, ignored if None.
    spur_table : dict {(m, n): dBc}, optional
        Higher-order products at |m·f_LO ± n·f_IF|, in dBc relative to the
        wanted output, as listed in mixer datasheets.
    conv_loss_ref : how the datasheet defines conv_loss_dB.
        'hybrid' (default, Mini-Circuits / Marki): relative to the total IF
            power into an ideal external 90° hybrid, i.e. both I and Q ports
            together (+3 dB above one port). P_RF = P_port + 3 dB - CL.
        'port': relative to the power at one I/Q port. P_RF = P_port - CL.
    The model is linear: keep the I/Q drive below the mixer's compression
    point (about +10 dBm per port for the ZMIQ-143H-S+).
    """
    if q_in is None and q_phase_deg is None:
        return _iq_mixer_single(lo, if_, conv_loss_dB, sideband, sideband_suppression_dBc,
                                amp_imbalance_dB, phase_imbalance_deg, lo_rf_isolation_dB,
                                if_rf_isolation_dB, spur_table, conv_loss_ref, floor_dBm, label)
    if sideband_suppression_dBc is not None:
        raise ValueError('with explicit I and Q the sidebands follow from their phases; '
                         'use amp_imbalance_dB / phase_imbalance_deg for the mixer imbalance')

    gain_ref = 10 * np.log10(2) if conv_loss_ref == 'hybrid' else 0.0
    q_rot = 10 ** (q_gain_dB / 20) * np.exp(1j * np.deg2rad(q_phase_deg or 0.0))
    mixer_q = 10 ** (amp_imbalance_dB / 20) * np.exp(-1j * np.deg2rad(phase_imbalance_deg))

    # pair I and Q tones by (frequency, origin): [template tone, a_I, a_Q]
    pairs = {}
    for t in if_:
        pairs[(round(t.f_GHz, 9), t.origin)] = [t, t.amplitude, 0j]
    for t in (if_ if q_in is None else q_in):
        key = (round(t.f_GHz, 9), t.origin)
        a = t.amplitude * q_rot
        if key in pairs:
            pairs[key][2] += a
        else:
            pairs[key] = [t, 0j, a]

    if q_in is not None and not any(abs(a_i) > 0 and abs(a_q) > 0 for _, a_i, a_q in pairs.values()):
        import warnings
        warnings.warn('iq_mixer: no tone of the I port has a partner on the Q port (same '
                      'frequency and origin), so the I/Q phase has no effect. Give both '
                      'channels the same name, e.g. source(..., name="AWG") for CH1 and CH2.')

    def level(a):                       # |a|^2 in mW -> dBm at the RF port
        return 10 * np.log10(abs(a) ** 2) + gain_ref - conv_loss_dB if abs(a) > 0 else -np.inf

    lo_main = lo.reference()
    out = []
    for L in lo:
        rel = L.P_dBm - lo_main.P_dBm               # LO spurs mix at their dBc
        for t, a_i, a_q in pairs.values():
            a_q = a_q * mixer_q
            usb = 0.5 * (a_i + 1j * a_q)
            lsb = 0.5 * (np.conj(a_i) + 1j * np.conj(a_q))
            P_u, P_l = level(usb) + rel, level(lsb) + rel
            base = _kind(L, t)
            for a, P, f, sym, other in ((usb, P_u, L.f_GHz + t.f_GHz, '+', P_l),
                                        (lsb, P_l, abs(L.f_GHz - t.f_GHz), '−', P_u)):
                if not np.isfinite(P):
                    continue
                kind = base
                if base == 'signal' and P < other - 3:
                    kind = 'image'                  # the suppressed sideband
                name = f'{_paren(L.origin)}{sym}{_paren(t.origin)}'
                out.append(Tone(f, P, name + (' image' if kind == 'image' else ''), kind,
                                L.phase_deg + np.rad2deg(np.angle(a))))
        out.append(replace(L, P_dBm=L.P_dBm - lo_rf_isolation_dB,
                           origin=f'{L.origin} leak', kind='leakage'))

    for t, a_i, a_q in pairs.values():
        p_in = abs(a_i) ** 2 + abs(a_q) ** 2       # I and Q leak, added in power
        if if_rf_isolation_dB is not None and p_in > 0:
            out.append(replace(t, P_dBm=10 * np.log10(p_in) - if_rf_isolation_dB,
                               origin=f'{t.origin} leak', kind='leakage'))
        a_q = a_q * mixer_q
        a_main = max(abs(0.5 * (a_i + 1j * a_q)), abs(0.5 * (np.conj(a_i) + 1j * np.conj(a_q))))
        P_wanted = level(a_main)
        for (m, n), dBc in (spur_table or {}).items():
            if (m, n) == (1, 1):
                continue                            # wanted + image handled above
            for sign, sym in ((1, '+'), (-1, '−')):
                out.append(Tone(abs(m * lo_main.f_GHz + sign * n * t.f_GHz), P_wanted - _sup(dBc),
                                f'{m}·{_paren(lo_main.origin)}{sym}{n}·{_paren(t.origin)}',
                                'intermod'))

    out = [t for t in out if t.P_dBm > floor_dBm]
    parents = (lo, if_) if q_in is None else (lo, if_, q_in)
    return Spectrum(out, label or 'IQ mixer', parents)


def _iq_mixer_single(lo, if_, conv_loss_dB=7.0, sideband='upper',
             sideband_suppression_dBc=None, amp_imbalance_dB=0.0,
             phase_imbalance_deg=0.0, lo_rf_isolation_dB=35.0,
             if_rf_isolation_dB=None, spur_table=None, conv_loss_ref='hybrid',
             floor_dBm=-150, label=None):
    """Behavioural IQ mixer: one IF spectrum, sideband chosen by `sideband`.

    lo : spectrum at the LO port. The strongest tone is taken as the LO;
        every other LO tone (harmonics, spurs) is mixed as well, at its own
        level relative to the LO.
    if_ : the IF spectrum at ONE of the I/Q ports, i.e. the complex IF tone
        driven on I and Q (e.g. one output of hybrid_90).
    sideband : 'upper' (f_LO + f_IF) or 'lower' (f_LO - f_IF) is wanted.
    sideband_suppression_dBc : suppression of the unwanted sideband. If not
        given it is computed from amp_imbalance_dB and phase_imbalance_deg
        of the I/Q paths (perfect balance -> no image).
    lo_rf_isolation_dB : LO leakage (carrier feedthrough) = P_LO - isolation.
    if_rf_isolation_dB : IF feedthrough, ignored if None.
    spur_table : dict {(m, n): dBc}, optional
        Higher-order products at |m·f_LO ± n·f_IF|, in dBc relative to the
        wanted output, as listed in mixer datasheets.
    conv_loss_ref : how the datasheet defines conv_loss_dB.
        'hybrid' (default, Mini-Circuits / Marki): relative to the total IF
            power into an ideal external 90° hybrid, i.e. both I and Q ports
            together (+3 dB above one port). P_RF = P_port + 3 dB - CL.
        'port': relative to the power at one I/Q port. P_RF = P_port - CL.
    """
    s = 1 if sideband == 'upper' else -1
    gain_ref = 10 * np.log10(2) if conv_loss_ref == 'hybrid' else 0.0
    if sideband_suppression_dBc is None:
        e = 10 ** (amp_imbalance_dB / 20)
        c = np.cos(np.deg2rad(phase_imbalance_deg))
        ratio = (1 - 2 * e * c + e ** 2) / (1 + 2 * e * c + e ** 2)
        image_dB = 10 * np.log10(ratio) if ratio > 0 else -np.inf
    else:
        image_dB = -_sup(sideband_suppression_dBc)

    lo_main = lo.reference()
    out = []
    for L in lo:
        rel = L.P_dBm - lo_main.P_dBm                # LO spurs mix at their dBc
        for I in if_:
            P = I.P_dBm + gain_ref - conv_loss_dB + rel
            name = f'{_paren(L.origin)}{"+" if s > 0 else "−"}{_paren(I.origin)}'
            image = f'{_paren(L.origin)}{"−" if s > 0 else "+"}{_paren(I.origin)}'
            out.append(Tone(abs(L.f_GHz + s * I.f_GHz), P, name, _kind(L, I),
                            L.phase_deg + s * I.phase_deg))
            out.append(Tone(abs(L.f_GHz - s * I.f_GHz), P + image_dB,
                            f'{image} image', 'image', L.phase_deg - s * I.phase_deg))
        out.append(replace(L, P_dBm=L.P_dBm - lo_rf_isolation_dB,
                           origin=f'{L.origin} leak', kind='leakage'))

    for I in if_:
        if if_rf_isolation_dB is not None:
            out.append(replace(I, P_dBm=I.P_dBm - if_rf_isolation_dB,
                               origin=f'{I.origin} leak', kind='leakage'))
        P_wanted = I.P_dBm + gain_ref - conv_loss_dB
        for (m, n), dBc in (spur_table or {}).items():
            if (m, n) == (1, 1):
                continue                            # wanted + image handled above
            for sign, sym in ((1, '+'), (-1, '−')):
                f = abs(m * lo_main.f_GHz + sign * n * I.f_GHz)
                out.append(Tone(f, P_wanted - _sup(dBc),
                                f'{m}·{_paren(lo_main.origin)}{sym}{n}·{_paren(I.origin)}',
                                'intermod'))

    out = [t for t in out if t.P_dBm > floor_dBm]
    return Spectrum(out, label or 'IQ mixer', (lo, if_))


def mixer(lo, if_, conv_loss_dB=7.0, lo_rf_isolation_dB=35.0,
          if_rf_isolation_dB=None, spur_table=None, floor_dBm=-150, label=None):
    """Double-balanced (non-IQ) mixer: both sidebands at equal level.

    Also works as a downconverter: products land at |f_LO ± f_in|.
    """
    return iq_mixer(lo, if_, conv_loss_dB=conv_loss_dB, sideband='upper',
                    sideband_suppression_dBc=0,
                    lo_rf_isolation_dB=lo_rf_isolation_dB,
                    if_rf_isolation_dB=if_rf_isolation_dB,
                    spur_table=spur_table, conv_loss_ref='port',
                    floor_dBm=floor_dBm, label=label or 'Mixer')
