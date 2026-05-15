"""
Geomagnetic field models (aligned and tilted dipole).

All coordinates in ECI. Returns B-field in Tesla.

Author: Ahmed S. Farahat
"""
import numpy as np
from .constants import B0_EQUATOR, R_EARTH, DIPOLE_TILT_RAD, OMEGA_EARTH


def aligned_dipole_field(r_eci):
    """Simple z-aligned dipole field."""
    x, y, z = r_eci
    r = np.sqrt(x*x + y*y + z*z)
    coeff = B0_EQUATOR * R_EARTH**3 / r**5
    return np.array([coeff * 3.0 * z * x,
                     coeff * 3.0 * z * y,
                     coeff * (3.0 * z * z - r * r)])


def tilted_dipole_field(r_eci, t=0.0):
    """
    Tilted dipole (11.5 deg tilt, co-rotating with Earth).
    At t=0 the tilt is in the x-z plane.
    """
    phi_rot = OMEGA_EARTH * t
    m_hat = np.array([
        np.sin(DIPOLE_TILT_RAD) * np.cos(phi_rot),
        np.sin(DIPOLE_TILT_RAD) * np.sin(phi_rot),
        np.cos(DIPOLE_TILT_RAD)
    ])

    r_vec = np.asarray(r_eci, dtype=float)
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / r

    m_dot_r = np.dot(m_hat, r_hat)
    coeff = B0_EQUATOR * (R_EARTH / r)**3
    return coeff * (3.0 * m_dot_r * r_hat - m_hat)


def field_magnitude_at_altitude(h_km):
    """Scalar dipole field magnitude at equatorial altitude h [km]."""
    r = R_EARTH + h_km * 1e3
    return B0_EQUATOR * (R_EARTH / r)**3
