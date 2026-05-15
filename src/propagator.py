"""
Coupled orbit + charging propagator (RK4, fixed timestep).

State vector: [x, y, z, vx, vy, vz, q]
Accelerations: two-body gravity, J2, Lorentz force.

Author: Ahmed S. Farahat
"""
import numpy as np
from .constants import MU_EARTH, R_EARTH, J2
from .magnetic_field import tilted_dipole_field
from .charging import net_current
from .orbital_elements import compute_elements


def equations_of_motion(t, state, mass, radius_m, I_beam=0.0,
                        include_j2=True, fixed_charge=False):
    """Right-hand side for the coupled 7-DOF system."""
    r_vec = state[0:3]
    v_vec = state[3:6]
    q = state[6]

    r = np.linalg.norm(r_vec)
    h_km = (r - R_EARTH) / 1e3

    # Gravity
    a_grav = -MU_EARTH / r**3 * r_vec

    # J2
    if include_j2:
        x, y, z = r_vec
        fac = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r**5
        zr2 = (z / r)**2
        a_j2 = np.array([fac * x * (5.0 * zr2 - 1.0),
                         fac * y * (5.0 * zr2 - 1.0),
                         fac * z * (5.0 * zr2 - 3.0)])
    else:
        a_j2 = np.zeros(3)

    # Lorentz
    B_vec = tilted_dipole_field(r_vec, t)
    a_lorentz = (q / mass) * np.cross(v_vec, B_vec)

    a_total = a_grav + a_j2 + a_lorentz

    # Charging
    if fixed_charge:
        dqdt = 0.0
    else:
        sunlit = (r_vec[0] > 0)
        dqdt = net_current(q, radius_m, h_km, sunlit, I_beam)

    return np.concatenate([v_vec, a_total, [dqdt]])


def rk4_step(f, t, y, dt, *args):
    """Single RK4 step."""
    k1 = f(t, y, *args)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, *args)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, *args)
    k4 = f(t + dt, y + dt*k3, *args)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate(r0, v0, q0, mass, radius_m, t_span, dt,
              I_beam=0.0, include_j2=True, save_every=1, fixed_charge=False):
    """Propagate the coupled orbit + charging system."""
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    state = np.concatenate([r0, v0, [q0]])

    max_save = n_steps // save_every + 2
    t_arr = np.empty(max_save)
    r_arr = np.empty((max_save, 3))
    v_arr = np.empty((max_save, 3))
    q_arr = np.empty(max_save)
    idx = 0

    def save(t_val, s):
        nonlocal idx
        t_arr[idx] = t_val
        r_arr[idx] = s[0:3]
        v_arr[idx] = s[3:6]
        q_arr[idx] = s[6]
        idx += 1

    t = t0
    save(t, state)

    for step_i in range(n_steps):
        state = rk4_step(equations_of_motion, t, state, dt,
                         mass, radius_m, I_beam, include_j2, fixed_charge)
        t += dt
        if (step_i + 1) % save_every == 0:
            save(t, state)

    t_arr = t_arr[:idx]
    r_arr = r_arr[:idx]
    v_arr = v_arr[:idx]
    q_arr = q_arr[:idx]

    elements = compute_elements(r_arr, v_arr)

    return {'t': t_arr, 'r': r_arr, 'v': v_arr, 'q': q_arr, 'elements': elements}
