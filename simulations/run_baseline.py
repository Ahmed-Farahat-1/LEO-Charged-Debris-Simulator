"""
Baseline coupled simulation: 10-day propagation at 500 km, 51.6 deg.
Generates orbital element and charge evolution figures.

Author: Ahmed S. Farahat
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.constants import MU_EARTH, R_EARTH
from src.charging import sphere_capacitance, equilibrium_potential, \
    electron_density, electron_temperature, debye_length, net_current
from src.magnetic_field import field_magnitude_at_altitude, tilted_dipole_field
from src.propagator import propagate

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                     'figure.dpi': 200, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})

print("=" * 60)
print("  DebriSolver v2 -- Baseline Simulation")
print("=" * 60)

# --- Fig 1: Plasma environment ---
print("\n[1/6] Plasma environment profiles...")
alts = np.linspace(200, 1000, 500)
ne = [electron_density(h) for h in alts]
Te = [electron_temperature(h) for h in alts]
lD = [debye_length(h) * 100 for h in alts]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].semilogy(alts, ne, 'b-', lw=2)
axes[0].set_xlabel('Altitude [km]'); axes[0].set_ylabel('$n_e$ [m$^{-3}$]')
axes[0].set_title('Electron Density')
axes[1].plot(alts, Te, 'r-', lw=2)
axes[1].set_xlabel('Altitude [km]'); axes[1].set_ylabel('$T_e$ [K]')
axes[1].set_title('Electron Temperature')
axes[2].plot(alts, lD, 'g-', lw=2)
axes[2].set_xlabel('Altitude [km]'); axes[2].set_ylabel('$\\lambda_D$ [cm]')
axes[2].set_title('Debye Length')
for ax in axes: ax.set_xlim(200, 1000)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_01_plasma_environment.png'))
plt.close()

# --- Fig 2: OML equilibrium ---
print("[2/6] OML equilibrium potentials...")
r_debris = 0.5
alt_sample = np.linspace(300, 900, 80)
V_sun = [equilibrium_potential(r_debris, h, sunlit=True) for h in alt_sample]
V_ecl = [equilibrium_potential(r_debris, h, sunlit=False) for h in alt_sample]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(alt_sample, V_sun, 'orange', lw=2, label='Sunlit')
ax.plot(alt_sample, V_ecl, 'navy', lw=2, label='Eclipse')
ax.set_xlabel('Altitude [km]'); ax.set_ylabel('$V_{eq}$ [V]')
ax.set_title('OML Equilibrium Surface Potential (r = 0.5 m)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_02_equilibrium_charging.png'))
plt.close()

# --- Propagation ---
print("[3/6] Running 10-day coupled propagation...")
mass = 50.0
alt0 = 500e3
inc0 = np.radians(51.6)
r0 = np.array([R_EARTH + alt0, 0.0, 0.0])
v_circ = np.sqrt(MU_EARTH / (R_EARTH + alt0))
v0 = np.array([0.0, v_circ * np.cos(inc0), v_circ * np.sin(inc0)])

V_eq = equilibrium_potential(r_debris, 500.0, sunlit=True)
q0 = sphere_capacitance(r_debris) * V_eq
print(f"   q_eq = {q0*1e9:.4f} nC, V_eq = {V_eq:.2f} V, q/m = {q0/mass:.2e} C/kg")

results = propagate(r0, v0, q0, mass, r_debris,
                    (0.0, 10 * 86400.0), 10.0,
                    include_j2=True, save_every=50, fixed_charge=True)

t_days = results['t'] / 86400.0
el = results['elements']
print(f"   Done. {len(results['t'])} points.")

# --- Fig 3: Orbital elements ---
print("[4/6] Plotting orbital elements...")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0,0].plot(t_days, el['h_perigee'], 'b-', lw=1, alpha=0.7)
axes[0,0].set_ylabel('Perigee Alt [km]'); axes[0,0].set_title('Perigee')
axes[0,1].plot(t_days, el['ecc'], 'r-', lw=1, alpha=0.7)
axes[0,1].set_ylabel('Eccentricity'); axes[0,1].set_title('Eccentricity')
axes[1,0].plot(t_days, el['inc'], 'g-', lw=1, alpha=0.7)
axes[1,0].set_ylabel('Inclination [deg]'); axes[1,0].set_title('Inclination')
axes[1,0].set_xlabel('Time [days]')
axes[1,1].plot(t_days, el['raan'], 'm-', lw=1, alpha=0.7)
axes[1,1].set_ylabel('RAAN [deg]'); axes[1,1].set_title('RAAN')
axes[1,1].set_xlabel('Time [days]')
fig.suptitle('Orbital Elements (500 km, 51.6 deg, natural charge)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_03_orbital_elements.png'))
plt.close()

# --- Fig 4: Trajectory ---
print("[5/6] Plotting trajectory...")
mask = t_days <= 2.0
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(results['r'][mask, 0]/1e6, results['r'][mask, 1]/1e6, 'b-', lw=0.3, alpha=0.6)
theta_e = np.linspace(0, 2*np.pi, 200)
ax.fill(R_EARTH/1e6*np.cos(theta_e), R_EARTH/1e6*np.sin(theta_e), color='lightblue', alpha=0.5)
ax.plot(R_EARTH/1e6*np.cos(theta_e), R_EARTH/1e6*np.sin(theta_e), 'b-', lw=1.5)
ax.set_xlabel('X [Mm]'); ax.set_ylabel('Y [Mm]')
ax.set_title('Orbital Trajectory (XY, 2 days)'); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig_04_trajectory.png'))
plt.close()

# --- Save data ---
print("[6/6] Saving results...")
datadir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(datadir, exist_ok=True)
np.savez(os.path.join(datadir, 'baseline_results.npz'),
         t=results['t'], r=results['r'], v=results['v'], q=results['q'],
         sma=el['sma'], ecc=el['ecc'], inc=el['inc'], raan=el['raan'],
         h_perigee=el['h_perigee'], h_apogee=el['h_apogee'])

print("\nBaseline simulation complete.")
print(f"Figures saved to: {os.path.abspath(FIGDIR)}")
