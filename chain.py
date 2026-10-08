# MW chain: AWG -> 90° hybrid -> IQ mixer (LO from a second source).
# Each "# %%" block is one cell: in VS Code click "Run Cell" above it,
# or copy the blocks into notebook cells in this order.

# %% imports
import numpy as np
from matplotlib import pyplot as plt

from mw_toolkit import *


# %% chain
# AWG: 1.8 GHz, 10 dBm, harmonics -30 dBc, non-harmonic spurs <= -76 dBc
awg = source(10, 1.8, harmonic_dBc=-30, n_harmonics=5, name='AWG')
awg_spur_dBc = -76     # spur positions unknown -> drawn as a limit line

# 90° hybrid (0.4 dB insertion loss) splits the AWG into the I and Q drives
if_i, if_q = hybrid_90(awg, insertion_loss_dB=0.4)

# LO: 8.4 GHz, 18 dBm
lo = source(18, 8.4, name='LO')

# IQ mixer; image from the I/Q imbalance (±0.2 dB, 4.5°) -> -27.8 dBc,
# slightly worse than the 30 dBc spec, so this is the conservative choice
rf = iq_mixer(lo, if_i, conv_loss_dB=7.2, sideband='upper',
              amp_imbalance_dB=0.2, phase_imbalance_deg=4.5,
              lo_rf_isolation_dB=40, if_rf_isolation_dB=25,
              label='IQ mixer RF output')

axes = plot_stages(rf, dBc=True, floor=-90, n_labels=6)
for ax, s in zip(axes, rf.stages()):
    if s is not lo:
        ax.axhline(awg_spur_dBc, color='#1baf7a', ls='--', lw=1)
        ax.text(ax.get_xlim()[1], awg_spur_dBc, ' AWG spur limit −76 dBc',
                va='center', ha='right', fontsize=8, color='#1baf7a',
                bbox=dict(fc='white', ec='none', pad=1))
plt.show()

rf.table()


# %% LO source comparison
# The same 8.4 GHz / 18 dBm LO from three kinds of source
lo_options = [
    analog_source(18, 8.4, harmonic_dBc=-30, name='Analog'),   # typical generator values
    pll_synthesizer(18, 8.4, preset='SynthHD v2'),     # harmonics from datasheet
    pll_synthesizer(18, 8.4, preset='SynthHD v1.4'),   # + 1/2 subharmonic (doubler)
]
plot_spectra(*lo_options, dBc=True, floor=-60)
plt.show()

for s in lo_options:
    s.table()
    print()


# %% full setup with real parts
# IF path: SMB100B (B103) -> VLF-2500+ lowpass -> 90° hybrid -> ZMIQ-143H-S+ I/Q
f_if = 1.8
gen = analog_source(7, f_if, harmonic_dBc=-30, nonharmonic_dBc=-76, name='SMB100B')
gen = filter_from_table(gen, part='VLF-2500+')
if_i, if_q = hybrid_90(gen, insertion_loss_dB=0.4)

# LO path: SynthHD ch1 -> ZVA-183-S+ -> 6 dB attenuator -> ZMIQ-143H-S+ LO (18.5 dBm)
lo = pll_synthesizer(-1.3, 8.4, preset='SynthHD v2', name='LO')
lo = amplifier(lo, gain_dB=lambda f: 26.5 if f <= 18 else 0,   # specified to 18 GHz only
               p1db_dBm=25.3, oip3_dBm=33.0, label='ZVA-183-S+')
lo = attenuator(lo, 6, label='BW-S6W2+')

# IQ mixer (8-14 GHz values of the ZMIQ-143H-S+ datasheet)
rf = iq_mixer(lo, if_i, conv_loss_dB=8.5, sideband='upper', sideband_suppression_dBc=-30,
              lo_rf_isolation_dB=40, if_rf_isolation_dB=25, label='ZMIQ-143H-S+')


# %% after the IQ mixer: filter both lines, then combine
# 9.9-10.4 GHz line: the LO (8.4 GHz) is too close for a plain highpass, so the
# lower edge is done by the cavity bandpass; the lowpass then removes what lies
# above the bandpass edge (11.25 GHz), e.g. LO+(2·IF) at 11.4-12.4 GHz, and
# everything above 18 GHz, where the bandpass is not specified.
rf_bp = filter_from_table(rf, part='ZVBP-10R5G-S+')      # 9.75-11.25 GHz
# amplifier only on this line (after the bandpass, so it sees one tone), not
# after the combiner where the strong 3.1 GHz signal would saturate it
rf_amp = amplifier(rf_bp, gain_dB=lambda f: 26.5 if f <= 18 else 0,
                   p1db_dBm=25.3, oip3_dBm=33.0, label='ZVA-183-S+ (10 GHz line)')
rf_lp = filter_from_table(rf_amp, part='ZLSS-11G-S+')    # DC-11 GHz, removes amp harmonics
# attenuator: sets the modulation depth at the EOM. With V_pi = 3 V the first
# 10.2 GHz sideband is largest at beta ~ 1.84, i.e. ~15 dBm RF -> 6 dB here.
# Without it the amplifier's ~21 dBm over-modulate (beta ~ 3.9, J1 ~ 0).
att_10GHz_dB = 6
rf_lp = attenuator(rf_lp, att_10GHz_dB, label=f'{att_10GHz_dB} dB attenuator (BW-S6W2+)')

# 3.1 GHz line: SynthHD ch2. Its spurs are harmonics (6.2, 9.3 GHz) -> lowpass.
# A highpass is not needed: the SynthHD puts nothing below 3.1 GHz (VCO at
# 6.2 GHz divided by 2, no subharmonic).
ch2 = pll_synthesizer(10, 3.1, preset='SynthHD v2', name='CH2')
ch2_lp = filter_from_table(ch2, part='VLF-3800+')         # DC-3.9 GHz

# combiner: diplexer (low loss, filters each input once more).
# Alternative: power_combiner(ch2_lp, rf_lp) -> 3 dB + 0.5 dB loss on both lines.
out = diplexer_combine(ch2_lp, rf_lp, part='ZDSS-5G6G-S+',
                       label='Output (ZDSS-5G6G-S+)')

axes = plot_stage_rows({
    'IQ mixer output / SynthHD ch2':     [rf, ch2],
    'Bandpass ZVBP-10R5G-S+ / lowpass VLF-3800+': [rf_bp, ch2_lp],
    'Amplifier ZVA-183-S+ / lowpass ZLSS-11G-S+': [rf_amp, rf_lp],
    'Combined output':                   out,
}, floor=-100)
plt.show()

out.table(12)
print()
print('relative to the 10.2 GHz signal:')
ref = max(t.P_dBm for t in out if t.origin == 'LO+SMB100B')      # wanted LO + IF
for t in sorted(out, key=lambda t: -t.P_dBm)[:12]:
    print(f'{t.f_GHz:8.3f} GHz {t.P_dBm - ref:7.1f} dBc  {t.origin}')


# %% laser through the EOM (phase modulator)
P_laser_mW = 10.0          # optical power into the EOM - set to your laser power
opt = eom_laser_output(out, laser_power_mW=P_laser_mW, vpi_V=3.0, insertion_loss_dB=3.5)

plot_spectrum(opt, floor=-60)
plt.show()

print(f'beta(3.1 GHz)  = {modulation_depth(max(t.P_dBm for t in out if t.origin == "CH2"), 3.0):.2f} rad')
print(f'beta(10.2 GHz) = {modulation_depth(max(t.P_dBm for t in out if t.origin == "LO+SMB100B"), 3.0):.2f} rad')
opt.table(14)



# %% Variant A: offset-IF probe (10.6-10.95 GHz) + fixed pump (16.835 GHz)
# LO 10.0 GHz split in two:
#   path 1 -> ZMIQ-143H-S+ LO, AWG (SDG7102A) I/Q at 0.6-0.95 GHz -> probe = LO + IF
#   path 2 -> ZX05-24MH-S+ LO, SynthHD ch2 6.835 GHz on its IF -> pump = LO + 6.835 GHz
# The offset IF puts LO leakage (10.0 GHz) and image (9.0-9.4 GHz) outside the
# probe band, where the ZVBP-10R7G-S+ removes them: no IQ calibration needed.
va_f_if = 0.8              # AWG frequency, 0.6-0.95 GHz -> probe 10.6-10.95 GHz
va_q_phase_deg = -90       # AWG CH2 (Q) phase vs CH1 (I): -90 -> upper sideband (LO + IF)
va_awg_fs = 5.0            # AWG sample rate: 5 GSa/s or 2.5 GSa/s - check by measurement!
va_att_probe_dB = 1        # sets the probe modulation depth at the EOM
va_att_pump_dB = 0         # sets the pump modulation depth at the EOM
va_vpi_V = 4.5             # NIR-MPX800-LN-20: 4-5 V at 50 kHz
# V_pi rises with frequency: -3 dB electro-optic bandwidth ~20 GHz (datasheet:
# 16-20 GHz). The steepness above it is an assumption (4th-order roll-off).
va_vpi = lambda f: va_vpi_V * np.sqrt(1 + (f / 20.0) ** 8)
va_laser_mW = 10.0         # optical power into the EOM

zva = dict(gain_dB=lambda f: 26.5 if f <= 18 else 0, p1db_dBm=25.3, oip3_dBm=33.0)  # ZVA-183-S+

# LO: SynthHD ch1 10.0 GHz -> ZVA-183-S+ -> ZX10-2-183+ splitter
va_lo = pll_synthesizer(-1.3, 10.0, preset='SynthHD v2', name='LO')
va_lo = amplifier(va_lo, label='ZVA-183-S+ (LO)', **zva)
va_lo_iq, va_lo_mx = power_splitter(va_lo, part='ZX10-2-183+')
va_lo_iq = attenuator(va_lo_iq, 1.0, label='1 dB att -> ZMIQ LO (~18.5 dBm)')
va_lo_mx = attenuator(va_lo_mx, 6.0, label='BW-S6W2+ -> ZX05-24MH-S+ LO (~13 dBm)')

# probe: AWG CH1 (I) and CH2 (Q) -> VLF-1200+ each -> ZMIQ-143H-S+ I/Q ports
#        -> ZVBP-10R7G-S+ -> ZVA-183-S+ -> ZLSS-11G-S+
# Both channels get the same name so the mixer pairs their tones; the sideband
# follows from the phases, the image from the ZMIQ's own imbalance (8-14 GHz typ.).
va_awg_i = source(4, va_f_if, harmonic_dBc=-40, n_harmonics=4, f_s_GHz=va_awg_fs, name='AWG')
va_awg_q = source(4, va_f_if, harmonic_dBc=-40, n_harmonics=4, f_s_GHz=va_awg_fs,
                  phase_deg=va_q_phase_deg, name='AWG')


va_awg_i = filter_from_table(va_awg_i, part='VLF-1200+', label='VLF-1200+ (I)')
va_awg_q = filter_from_table(va_awg_q, part='VLF-1200+', label='VLF-1200+ (Q)')


va_rf = iq_mixer(va_lo_iq, va_awg_i, va_awg_q, conv_loss_dB=8.5,
                 amp_imbalance_dB=0.2, phase_imbalance_deg=4.5,
                 lo_rf_isolation_dB=40, if_rf_isolation_dB=25, label='ZMIQ-143H-S+')


va_probe = filter_from_table(va_rf, part='ZVBP-10R7G-S+')
va_probe = amplifier(va_probe, label='ZVA-183-S+ (probe)', **zva)
va_probe = filter_from_table(va_probe, part='ZLSS-11G-S+')
va_probe = attenuator(va_probe, va_att_probe_dB, label=f'{va_att_probe_dB} dB att (probe)')

# pump: ZX05-24MH-S+ (LO = 10 GHz, IF = ch2 6.835 GHz) -> ZVBP-K16R95G+ -> ZVA-183-S+
va_ch2 = pll_synthesizer(0, 6.835, preset='SynthHD v2', name='CH2')
va_pump_mix = mixer(va_lo_mx, va_ch2, conv_loss_dB=9, lo_rf_isolation_dB=30, if_rf_isolation_dB=20,
                    label='ZX05-24MH-S+')
va_pump = filter_from_table(va_pump_mix, part='ZVBP-K16R95G+')
va_pump = amplifier(va_pump, label='ZVA-183-S+ (pump)', **zva)
va_pump = attenuator(va_pump, va_att_pump_dB, label=f'{va_att_pump_dB} dB att (pump)')

# combine (ZX10-2-183+ as combiner) -> EOM (NIR-MPX800-LN-20)
va_out = power_combiner(va_probe, va_pump, part='ZX10-2-183+',
                        label='EOM input (ZX10-2-183+ combiner)')
va_opt = eom_laser_output(va_out, laser_power_mW=va_laser_mW, vpi_V=va_vpi, insertion_loss_dB=4.0)

plot_stage_rows({
    'LO and sources':          [va_lo, va_awg_i, va_ch2],
    'Mixers':                  [va_rf, va_pump_mix],
    'Probe / pump to the EOM': [va_probe, va_pump],
    'EOM input':               va_out,
    'Laser after the EOM':     va_opt,
}, floor=-90, n_labels=3)
plt.show()

va_P_probe = max(t.P_dBm for t in va_out if t.origin == 'LO+AWG')
va_P_pump = max(t.P_dBm for t in va_out if t.origin == 'LO+CH2')
print(f'probe {10 + va_f_if:.3f} GHz: {va_P_probe:5.1f} dBm, V_pi {va_vpi(10 + va_f_if):.1f} V, '
      f'beta = {modulation_depth(va_P_probe, va_vpi(10 + va_f_if)):.2f}')
print(f'pump  {10 + 6.835:.3f} GHz: {va_P_pump:5.1f} dBm, V_pi {va_vpi(16.835):.1f} V, '
      f'beta = {modulation_depth(va_P_pump, va_vpi(16.835)):.2f}')
print('\nEOM input relative to the probe:')
for t in sorted(va_out, key=lambda t: -t.P_dBm)[:10]:
    print(f'{t.f_GHz:8.3f} GHz {t.P_dBm - va_P_probe:7.1f} dBc  {t.kind:9s} {t.origin}')
print()
va_opt.table(12)


# %% Setup 2 interactive: laser spectrum after the EOM vs. AWG / pump settings
# Same chain as the "Setup 2" notebook cell (LO 9.4 GHz, SDG7102A I/Q, ZMIQ-143H-S+,
# ZVBP-10R5G-S+, ZLSS-11G-S+, SynthHD ch2 3.1 GHz + VLF-3800+, ZDSS-5G6G-S+ -> EOM),
# wrapped in five sliders. Needs ipywidgets and anywidget (Jupyter / VS Code).
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display


def setup2_spectra(f_if=0.8, awg_q_phase_shift=-90, awg_pow_i=2, awg_pow_q=3,
                   windfreak_pump_power=10):
    """RF spectrum at the EOM and the optical spectrum after it."""
    harm = -40 if f_if > 0.5 else -55                    # SDG7102A datasheet
    lo = pll_synthesizer(P_dBm=-1.3, f_GHz=9.4, preset='SynthHD v2', name='SynthHD Ch1')
    mw_1 = pll_synthesizer(P_dBm=windfreak_pump_power, f_GHz=3.1, preset='SynthHD v2',
                           name='SynthHD Ch2')
    mw_2_i = source(P_dBm=awg_pow_i, f_GHz=f_if, harmonic_dBc=harm, n_harmonics=5,
                    f_s_GHz=5, name='SDG7102A')
    mw_2_q = source(P_dBm=awg_pow_q, f_GHz=f_if, harmonic_dBc=harm, n_harmonics=5,
                    f_s_GHz=5, phase_deg=awg_q_phase_shift, name='SDG7102A')

    lo = amplifier(lo, gain_dB=lambda f: 26.5 if f < 18 else 0, p1db_dBm=25.3,
                   oip3_dBm=33, label='ZVA-183-S+')
    lo = attenuator(lo, 6, label='BW-S6W2+')
    mw_2_i = filter_from_table(mw_2_i, part='VLF-1200+', label='VLF-1200+ (I)')
    mw_2_q = filter_from_table(mw_2_q, part='VLF-1200+', label='VLF-1200+ (Q)')

    mw_3 = iq_mixer(lo, mw_2_i, mw_2_q, conv_loss_dB=8.5, amp_imbalance_dB=0.2,
                    phase_imbalance_deg=4.5, lo_rf_isolation_dB=40, if_rf_isolation_dB=25,
                    label='ZMIQ-143H-S+')
    mw_3_lp = filter_from_table(filter_from_table(mw_3, part='ZVBP-10R5G-S+'),
                                part='ZLSS-11G-S+')
    mw_1_lp = filter_from_table(mw_1, part='VLF-3800+')
    out = diplexer_combine(mw_1_lp, mw_3_lp, part='ZDSS-5G6G-S+', label='Output (ZDSS-5G6G-S+)')
    opt = eom_laser_output(out, laser_power_mW=10, vpi_V=3.0, insertion_loss_dB=3.5)
    return out, opt


def _stems(spectrum, floor):
    """Per tone kind: x, y, hover text, marker sizes for a stem plot (None-separated)."""
    data = {}
    for t in spectrum:
        if t.P_dBm <= floor:
            continue
        x, y, text, size = data.setdefault(t.kind, ([], [], [], []))
        label = f'{t.origin}<br>{t.f_GHz:+.3f} GHz<br>{t.P_dBm:.1f} dBm'
        x += [t.f_GHz, t.f_GHz, None]
        y += [floor, t.P_dBm, None]
        text += [label, label, None]
        size += [0, 7, 0]
    return data


def setup2_widget(floor=-60, vpi_V=3.0):
    """Five sliders -> optical spectrum after the EOM (updates live while dragging)."""
    style = dict(description_width='190px')
    layout = widgets.Layout(width='620px')
    sliders = {
        'f_if': widgets.FloatSlider(value=0.8, min=0.5, max=1.0, step=0.01,
                                    description='f_if (GHz)', readout_format='.2f'),
        'awg_q_phase_shift': widgets.FloatSlider(value=-90, min=-180, max=180, step=1,
                                                 description='Q phase vs. I (°)'),
        'awg_pow_i': widgets.FloatSlider(value=2, min=-20, max=10, step=0.5,
                                         description='AWG I power (dBm)'),
        'awg_pow_q': widgets.FloatSlider(value=3, min=-20, max=10, step=0.5,
                                         description='AWG Q power (dBm)'),
        'windfreak_pump_power': widgets.FloatSlider(value=10, min=-10, max=17, step=0.5,
                                                    description='SynthHD ch2 power (dBm)'),
    }
    for s in sliders.values():
        s.style, s.layout, s.continuous_update = style, layout, True   # ~10 ms per update

    fig = go.FigureWidget()
    for kind, color in KIND_COLORS.items():
        fig.add_scatter(name=kind, mode='lines+markers', line=dict(color=color, width=2),
                        marker=dict(color=color), hovertemplate='%{text}<extra></extra>',
                        x=[], y=[])
    fig.update_layout(height=520, template='plotly_white', hovermode='closest',
                      xaxis_title='Offset from laser (GHz)', yaxis_title='Optical power (dBm)',
                      yaxis_range=[floor, 10], legend=dict(orientation='h', y=1.12))
    info = widgets.HTML()

    def update(_=None):
        values = {k: s.value for k, s in sliders.items()}
        out, opt = setup2_spectra(**values)
        data = _stems(opt, floor)
        sig = sorted((t for t in out if t.kind == 'signal'), key=lambda t: t.f_GHz)
        with fig.batch_update():
            for tr in fig.data:
                x, y, text, size = data.get(tr.name, ([], [], [], []))
                tr.x, tr.y, tr.text, tr.marker.size = x, y, text, size
            fig.layout.title.text = 'Laser after the EOM (10 mW, V_pi 3 V, IL 3.5 dB)'
        line = lambda f: max((t.P_dBm for t in opt if abs(t.f_GHz - f) < 1e-6), default=-np.inf)
        rows = ''.join(
            f'<tr><td>{t.origin}</td><td>{t.f_GHz:.3f} GHz</td><td>{t.P_dBm:.1f} dBm</td>'
            f'<td>β = {modulation_depth(t.P_dBm, vpi_V):.2f}</td>'
            f'<td>±1 sideband: {line(t.f_GHz):.1f} dBm</td></tr>' for t in sig)
        info.value = (f'<table style="font-family:monospace">{rows}'
                      f'<tr><td>laser carrier</td><td></td><td></td><td></td>'
                      f'<td>{line(0.0):.1f} dBm</td></tr></table>')

    for s in sliders.values():
        s.observe(update, names='value')
    update()
    display(widgets.VBox(list(sliders.values()) + [info, fig]))


setup2_widget()
