# DebriSolver v2

Coupled electrostatic charging and orbital perturbation analysis of charged debris in low Earth orbit.

**Author:** Ahmed S. Farahat

## About

This repository contains the simulation code and data associated with the study of Lorentz-force perturbations on electrostatically charged debris in LEO. The framework couples Orbital Motion Limited (OML) charging physics with numerical orbit propagation to quantify the magnitude of electromagnetic perturbations across realistic charge-to-mass ratio regimes.

The main findings:
- Natural equilibrium charging produces q/m ~ 10⁻¹² C/kg — far too small for meaningful orbital perturbation
- The Lorentz force does no work (F·v = 0) and cannot reduce orbital energy
- Even optimistic active charging (q/m ~ 10⁻⁴ C/kg) yields RAAN drift of only ~0.01 deg/day
- Secular perturbation is in orbital plane orientation, not orbit size

## Structure

```
src/              Core simulation modules
simulations/      Executable simulation cases
figures/          Publication figures
data/             Supporting datasets
examples/         Usage example
```

## Requirements

Python 3.9+, NumPy, Matplotlib. Install with:

```
pip install -r requirements.txt
```

## Running

Generate all results and figures:

```
python simulations/run_baseline.py
python simulations/run_qm_sweep.py
```

Quick test (3 orbits, console output only):

```
python examples/quick_start.py
```

Validation checks:

```
python simulations/run_validation.py
```

## Key Files

| File | Description |
|------|-------------|
| `src/charging.py` | OML current collection and plasma profiles |
| `src/magnetic_field.py` | Tilted-dipole geomagnetic field model |
| `src/propagator.py` | Coupled 7-DOF RK4 orbit integrator |
| `src/constants.py` | Physical constants (SI) |
| `simulations/run_baseline.py` | 10-day coupled propagation at 500 km |
| `simulations/run_qm_sweep.py` | q/m sensitivity analysis |
| `simulations/run_validation.py` | Energy conservation and J2 rate checks |

## Figures

Figures are generated deterministically by the simulation scripts. Key outputs include plasma environment profiles, OML equilibrium charging, orbital element evolution, Lorentz perturbation sensitivity to q/m, and numerical convergence verification.

## Citation

```bibtex
@software{farahat2026debrisolver,
  author  = {Farahat, Ahmed S.},
  title   = {{DebriSolver v2: Coupled Electrostatic Charging and Orbital Perturbation Framework}},
  year    = {2026},
  version = {2.0.0}
}
```

## License

MIT License. See [LICENSE](LICENSE).
