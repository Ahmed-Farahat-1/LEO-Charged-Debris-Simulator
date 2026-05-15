"""
Physical constants for DebriSolver v2.
All values in SI units.

Author: Ahmed S. Farahat
"""
import numpy as np

# Earth
MU_EARTH = 3.986004418e14      # Gravitational parameter [m^3/s^2]
R_EARTH = 6371.2e3             # Mean equatorial radius [m]
OMEGA_EARTH = 7.2921159e-5     # Earth rotation rate [rad/s]
J2 = 1.08263e-3                # J2 oblateness coefficient

# Geomagnetic field
B0_EQUATOR = 3.12e-5           # Equatorial surface field [T]
DIPOLE_TILT_DEG = 11.5         # Magnetic dipole tilt [deg]
DIPOLE_TILT_RAD = np.radians(DIPOLE_TILT_DEG)

# Plasma / charging
E_CHARGE = 1.602176634e-19     # Elementary charge [C]
M_ELECTRON = 9.1093837015e-31  # Electron mass [kg]
M_PROTON = 1.67262192e-27      # Proton mass [kg]
M_OXYGEN = 16.0 * M_PROTON     # Oxygen ion mass [kg]
K_BOLTZMANN = 1.380649e-23     # Boltzmann constant [J/K]
EPSILON_0 = 8.854187817e-12    # Vacuum permittivity [F/m]
PI = np.pi

# Photoelectron parameters
J_PHOTO = 2.0e-5               # Photoelectron current density [A/m^2]
