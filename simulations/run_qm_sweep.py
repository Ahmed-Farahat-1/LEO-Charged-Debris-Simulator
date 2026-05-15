"""
Charge-to-mass ratio sensitivity analysis and convergence study.
Generates q/m sweep and RK4 convergence figures.

Author: Ahmed S. Farahat
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.constants import MU_EARTH, R_EARTH
from src.propagator import propagate

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                     'figure.dpi': 200, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})

print("=" * 60)
print("  DebriSolver v2 -- Sensitivity & Convergence")
print("=" * 60)

mass = 50.0
r_debris = 0.5
alt0 = 500e3
inc0 = np.radians(51.6)
v_circ = np.sqrt(MU_EARTH / (R_EARTH + alt0))
r0 = np.array([R_EARTH + alt0, 0.0, 0.0])
v0 = np.array([0.0, v_circ * np.cos(inc0), v_circ * np.sin(inc0)])

# --- q/m sweep ---
print("\n[1/2] Running q/m sensitivity sweep (10 days)...")
qm_sweep = [1e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
t_span = (0.0, 10 * 86400.0)
save_every = 100

# J2-only reference
ref = propagate(r0.copy(), v0.copy(), 0.0, mass, r_debris, t_span,
                10.0, include_j2=True, save_every=save_every, fixed_charge=True)

sweep = {}
for qm in qm_sweep:
    res = propagate(r0.copy(), v0.copy(), qm * mass, mass, r_debris, t_span,
                    10.0, include_j2=True, save_every=save_every, fixed_charge=True)
    sweep[qm] = res
    n = min(len(res['elements']['sma']), len(ref['elements']['sma']))
    da = res['elements']['sma'][:n] - ref['elements']['sma'][:n]
    print(f"   q/m = {qm:.0e}: Da_Lorentz = {da[-1]:.4f} m")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(qm_sweep)))
for (qm, res), c in zip(sweep.items(), colors):
    n = min(len(res['t']), len(ref['t']))
    td = res['t'][:n] / 86400.0
    label = f'q/m={qm:.0e}'
    axes[0].plot(td, res['elements']['sma'][:n] - ref['elements']['sma'][:n],
                 color=c, lw=1.5, label=label)
    axes[1].plot(td, res['elements']['ecc'][:n] - ref['elements']['ecc'][:n],
                 color=c, lw=1.5, label=label)
    axes[2].plot(td, res['elements']['raan'][:n] - ref['elements']['raan'][:n],
                 color=c, lw=1.5, label=label)

axes[0].set_ylabel('Da [m]'); axes[0].set_title('SMA Perturbation')
axes[1].set_ylabel('De'); axes[1].set_title('Eccentricity Perturbation')
axes[2].set_ylabel('DRAAN [deg]'); axes[2].set_title('RAAN Perturbation')
for ax in axes:
    ax.set_xlabel('Time [days]'); ax.legend(fontsize=7, ncol=2)
fig.suptitle('Lorentz Perturbation vs q/m (10-day, 500 km)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_05_qm_sensitivity.png'))
plt.close()

# --- Convergence study ---
print("\n[2/2] Running convergence study (2 days)...")
dt_tests = [120, 60, 30, 15, 10, 5]
conv_sma, conv_ecc = [], []
q_conv = 1e-5 * mass

for dt_c in dt_tests:
    res = propagate(r0.copy(), v0.copy(), q_conv, mass, r_debris,
                    (0.0, 2 * 86400.0), dt_c,
                    include_j2=True, save_every=max(1, int(100/dt_c)),
                    fixed_charge=True)
    conv_sma.append(res['elements']['sma'][-1])
    conv_ecc.append(res['elements']['ecc'][-1])
    print(f"   dt = {dt_c:>4d} s: a_final = {res['elements']['sma'][-1]/1e3:.4f} km")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].semilogx(dt_tests, np.array(conv_sma)/1e3, 'bo-', lw=2, ms=8)
axes[0].set_xlabel('dt [s]'); axes[0].set_ylabel('Final SMA [km]')
axes[0].set_title('SMA Convergence')
axes[1].semilogx(dt_tests, conv_ecc, 'ro-', lw=2, ms=8)
axes[1].set_xlabel('dt [s]'); axes[1].set_ylabel('Final Eccentricity')
axes[1].set_title('Eccentricity Convergence')
fig.suptitle('Numerical Convergence (RK4, 2-day)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_06_convergence.png'))
plt.close()

print("\nSensitivity and convergence analysis complete.")
