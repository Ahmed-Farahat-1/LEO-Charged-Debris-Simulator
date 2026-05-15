"""
Quick start: 3-orbit propagation with console summary.

Author: Ahmed S. Farahat
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.constants import MU_EARTH, R_EARTH
from src.charging import equilibrium_potential, sphere_capacitance
from src.magnetic_field import field_magnitude_at_altitude
from src.propagator import propagate

mass = 50.0
r_debris = 0.5
alt_km = 500.0

r0 = np.array([R_EARTH + alt_km * 1e3, 0.0, 0.0])
v_circ = np.sqrt(MU_EARTH / (R_EARTH + alt_km * 1e3))
inc = np.radians(51.6)
v0 = np.array([0.0, v_circ * np.cos(inc), v_circ * np.sin(inc)])

V_eq = equilibrium_potential(r_debris, alt_km, sunlit=True)
q0 = sphere_capacitance(r_debris) * V_eq

T_orbit = 2 * np.pi * np.sqrt((R_EARTH + alt_km * 1e3)**3 / MU_EARTH)

print(f"Orbit:  {alt_km:.0f} km, {np.degrees(inc):.1f} deg")
print(f"V_eq:   {V_eq:.2f} V, q = {q0*1e9:.4f} nC, q/m = {q0/mass:.2e} C/kg")
print(f"B(500): {field_magnitude_at_altitude(alt_km)*1e6:.1f} uT")
print(f"Period: {T_orbit:.0f} s\n")

print("Propagating 3 orbits...")
res = propagate(r0, v0, q0, mass, r_debris,
                (0.0, 3 * T_orbit), 10.0,
                include_j2=True, save_every=50, fixed_charge=True)

el = res['elements']
print(f"Done. {len(res['t'])} points.\n")
print(f"SMA:       {el['sma'][0]/1e3:.3f} -> {el['sma'][-1]/1e3:.3f} km")
print(f"Ecc:       {el['ecc'][0]:.2e} -> {el['ecc'][-1]:.2e}")
print(f"Inc:       {el['inc'][0]:.4f} -> {el['inc'][-1]:.4f} deg")
print(f"h_perigee: {el['h_perigee'][0]:.2f} -> {el['h_perigee'][-1]:.2f} km")

dE = (el['energy'][-1] - el['energy'][0]) / abs(el['energy'][0])
print(f"\ndE/E = {dE:.2e}")
