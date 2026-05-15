"""
OML charging model and LEO plasma environment profiles.

Implements altitude-dependent plasma parameters (IRI-2016 representative)
and Orbital Motion Limited current collection on a conducting sphere.

Author: Ahmed S. Farahat
"""
import numpy as np
from .constants import (
    E_CHARGE, M_ELECTRON, M_OXYGEN, K_BOLTZMANN,
    EPSILON_0, PI, R_EARTH, J_PHOTO
)


# --- Plasma profiles (IRI-2016 representative, mid-latitude dayside) ---

def electron_density(h_km):
    """Electron density [m^-3] vs altitude [km]."""
    h = float(h_km)
    if h < 200:
        return 1.0e11
    elif h < 350:
        return 1.0e11 * 10.0**((h - 200.0) / 150.0 * 0.6)
    elif h < 500:
        return 4.0e11 * 10.0**(-(h - 350.0) / 150.0 * 0.3)
    elif h < 800:
        return 2.0e11 * 10.0**(-(h - 500.0) / 300.0 * 1.0)
    else:
        return 2.0e10 * 10.0**(-(h - 800.0) / 400.0 * 0.5)


def electron_temperature(h_km):
    """Electron temperature [K] vs altitude [km]."""
    h = float(h_km)
    if h < 300:
        return 1500.0 + (h - 200.0) * 5.0
    elif h < 600:
        return 2000.0 + (h - 300.0) * 3.33
    else:
        return 3000.0 + (h - 600.0) * 2.5


def ion_temperature(h_km):
    """Ion temperature [K] — approximately 0.7 * T_e in LEO."""
    return 0.7 * electron_temperature(h_km)


def debye_length(h_km):
    """Debye length [m]."""
    ne = electron_density(h_km)
    Te = electron_temperature(h_km)
    return np.sqrt(EPSILON_0 * K_BOLTZMANN * Te / (ne * E_CHARGE**2))


# --- OML current collection ---

def sphere_capacitance(radius_m):
    """Capacitance of an isolated sphere: C = 4*pi*eps0*r [F]."""
    return 4.0 * PI * EPSILON_0 * radius_m


def oml_electron_current(V_s, n_e, T_e, A_surf):
    """OML electron current [A]. Negative = deposits electrons."""
    kTe = K_BOLTZMANN * T_e
    v_th_e = np.sqrt(kTe / (2.0 * PI * M_ELECTRON))
    J0 = n_e * E_CHARGE * v_th_e

    if V_s <= 0.0:
        arg = max(E_CHARGE * V_s / kTe, -500.0)
        return -J0 * A_surf * np.exp(arg)
    else:
        return -J0 * A_surf * (1.0 + E_CHARGE * V_s / kTe)


def oml_ion_current(V_s, n_e, T_i, A_surf):
    """OML O+ ion current [A]. Positive = deposits positive charge."""
    kTi = K_BOLTZMANN * T_i
    v_th_i = np.sqrt(kTi / (2.0 * PI * M_OXYGEN))
    J0 = n_e * E_CHARGE * v_th_i

    if V_s >= 0.0:
        arg = min(-E_CHARGE * V_s / kTi, 500.0)
        return J0 * A_surf * np.exp(arg)
    else:
        enhancement = min(1.0 - E_CHARGE * V_s / kTi, 1e6)
        return J0 * A_surf * enhancement


def photoelectron_current(A_illuminated, sunlit=True):
    """Photoelectron current [A]."""
    if not sunlit:
        return 0.0
    return J_PHOTO * A_illuminated


def net_current(q, radius_m, h_km, sunlit=True, I_beam=0.0):
    """Net current onto the debris sphere [C/s]."""
    C = sphere_capacitance(radius_m)
    V_s = max(min(q / C, 1e5), -1e5)

    n_e = electron_density(h_km)
    T_e = electron_temperature(h_km)
    T_i = ion_temperature(h_km)

    A_surf = 4.0 * PI * radius_m**2
    A_illum = PI * radius_m**2

    I_e = oml_electron_current(V_s, n_e, T_e, A_surf)
    I_i = oml_ion_current(V_s, n_e, T_i, A_surf)
    I_ph = photoelectron_current(A_illum, sunlit)

    return I_e + I_i + I_ph - I_beam


def equilibrium_potential(radius_m, h_km, sunlit=True, I_beam=0.0):
    """Find equilibrium surface potential where net current = 0 (bisection)."""
    C = sphere_capacitance(radius_m)

    def func(V):
        return net_current(C * V, radius_m, h_km, sunlit, I_beam)

    V_lo, V_hi = -100e3, 100.0
    f_lo, f_hi = func(V_lo), func(V_hi)

    if f_lo * f_hi > 0:
        return V_lo if abs(f_lo) < abs(f_hi) else V_hi

    for _ in range(200):
        V_mid = 0.5 * (V_lo + V_hi)
        f_mid = func(V_mid)
        if abs(f_mid) < 1e-20 or (V_hi - V_lo) < 0.01:
            return V_mid
        if f_mid * f_lo < 0:
            V_hi, f_hi = V_mid, f_mid
        else:
            V_lo, f_lo = V_mid, f_mid
    return 0.5 * (V_lo + V_hi)
