"""
Validation checks: OML equilibrium, energy conservation, J2 RAAN rate.

Author: Ahmed S. Farahat
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.constants import MU_EARTH, R_EARTH, J2
from src.charging import equilibrium_potential
from src.propagator import propagate


def run_validation():
    print("=" * 50)
    print("  DebriSolver v2 -- Validation Suite")
    print("=" * 50)
    passed = 0

    # Test 1: OML equilibrium
    print("\n  [1] OML equilibrium at 500 km")
    V_s = equilibrium_potential(0.5, 500.0, sunlit=True)
    V_e = equilibrium_potential(0.5, 500.0, sunlit=False)
    print(f"      V_sunlit = {V_s:.2f} V, V_eclipse = {V_e:.2f} V")
    ok = -5.0 < V_s < 5.0 and -5.0 < V_e < 0.0
    print(f"      {'PASS' if ok else 'FAIL'}")
    passed += ok

    # Test 2: Energy conservation (Keplerian, no perturbation)
    print("\n  [2] Energy conservation (1-day Keplerian)")
    alt0 = 500e3
    r0 = np.array([R_EARTH + alt0, 0.0, 0.0])
    vc = np.sqrt(MU_EARTH / (R_EARTH + alt0))
    v0 = np.array([0.0, vc, 0.0])
    res = propagate(r0, v0, 0.0, 50.0, 0.5, (0.0, 86400.0), 10.0,
                    include_j2=False, save_every=100, fixed_charge=True)
    E = res['elements']['energy']
    dE = abs((E[-1] - E[0]) / E[0])
    print(f"      dE/E = {dE:.2e}")
    ok = dE < 1e-9
    print(f"      {'PASS' if ok else 'FAIL'}")
    passed += ok

    # Test 3: J2 RAAN rate
    print("\n  [3] J2 RAAN rate vs analytical")
    inc0 = np.radians(51.6)
    v0 = np.array([0.0, vc * np.cos(inc0), vc * np.sin(inc0)])
    res = propagate(r0.copy(), v0, 0.0, 50.0, 0.5, (0.0, 5*86400.0), 10.0,
                    include_j2=True, save_every=200, fixed_charge=True)
    raan_uw = np.unwrap(np.radians(res['elements']['raan']))
    rate_num = np.degrees(raan_uw[-1] - raan_uw[0]) / 5.0
    a = R_EARTH + alt0
    n = np.sqrt(MU_EARTH / a**3)
    rate_ana = np.degrees(-1.5 * n * J2 * (R_EARTH/a)**2 * np.cos(inc0)) * 86400
    err = abs(rate_num - rate_ana) / abs(rate_ana)
    print(f"      Numerical = {rate_num:.4f} deg/day")
    print(f"      Analytical = {rate_ana:.4f} deg/day ({err*100:.2f}% error)")
    ok = err < 0.01
    print(f"      {'PASS' if ok else 'FAIL'}")
    passed += ok

    print(f"\n  Result: {passed}/3 passed\n")


if __name__ == '__main__':
    run_validation()
