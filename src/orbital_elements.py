"""
Osculating Keplerian element computation from Cartesian state vectors.

Author: Ahmed S. Farahat
"""
import numpy as np
from .constants import MU_EARTH, R_EARTH


def compute_elements(r_arr, v_arr):
    """Compute osculating elements from position/velocity arrays."""
    n = len(r_arr)
    sma = np.empty(n)
    ecc = np.empty(n)
    inc = np.empty(n)
    raan = np.empty(n)
    h_perigee = np.empty(n)
    h_apogee = np.empty(n)
    energy = np.empty(n)

    for i in range(n):
        r_vec, v_vec = r_arr[i], v_arr[i]
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        eps = 0.5 * v**2 - MU_EARTH / r
        energy[i] = eps
        a = -MU_EARTH / (2.0 * eps) if eps != 0 else np.inf
        sma[i] = a

        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        inc[i] = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))

        n_vec = np.cross([0, 0, 1], h_vec)
        n_mag = np.linalg.norm(n_vec)
        if n_mag > 1e-10:
            raan_val = np.arccos(np.clip(n_vec[0] / n_mag, -1, 1))
            if n_vec[1] < 0:
                raan_val = 2 * np.pi - raan_val
            raan[i] = np.degrees(raan_val)
        else:
            raan[i] = 0.0

        e_vec = (np.cross(v_vec, h_vec) / MU_EARTH) - (r_vec / r)
        ecc[i] = np.linalg.norm(e_vec)

        if a > 0 and ecc[i] < 1.0:
            h_perigee[i] = (a * (1.0 - ecc[i]) - R_EARTH) / 1e3
            h_apogee[i] = (a * (1.0 + ecc[i]) - R_EARTH) / 1e3
        else:
            h_perigee[i] = (r - R_EARTH) / 1e3
            h_apogee[i] = h_perigee[i]

    return {'sma': sma, 'ecc': ecc, 'inc': inc, 'raan': raan,
            'h_perigee': h_perigee, 'h_apogee': h_apogee, 'energy': energy}
