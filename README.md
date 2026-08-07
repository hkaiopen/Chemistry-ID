# Chemistry-ID: Information Dynamics for Chemical Reaction Simulations

**Information dynamics provides a unified computational architecture for chemical reaction simulations**—integrating non-adiabatic dissociation (NaI), adiabatic dissociation (Cl₂), and formation (H₂) into a single abstract framework.


---

## Overview

**Information Dynamics** reformulates chemical reactions as a three-element system:

| Element | Description | Chemical Equivalent |
|:---|:---|:---|
| **Virtual Space** | Absolute rules defining allowed states | Potential Energy Surfaces (PESs) |
| **Real Space** | Observational data with uncertainty | Initial wave packet distribution |
| **Coupling Matrix** | Projection mechanism driving evolution | Gradient flow / Landau-Zener transitions |

The framework demonstrates that three distinct reaction types—non-adiabatic dissociation (NaI), adiabatic dissociation (Cl₂), and formation (H₂)—can be described with the same abstract architecture, enabling consistent parameter calibration and cross-reaction comparison.

**Key features**:
- Single code framework for three reaction types
- Velocity Verlet integrator with energy conservation
- Landau-Zener surface hopping for non-adiabatic transitions
- Multi-ensemble statistics for robust error estimation
- Parallel grid scanning for parameter calibration

---

## Quick Start

### Run all three simulations

```bash
# NaI photodissociation (default parameters: De=3.30 eV, V12=0.034 eV)
cd NaI_photodissociation
python nai_photodissociation_v2.py

# Cl₂ photodissociation
cd ../Cl2_photodissociation
python cl2_photodissociation.py

# H₂ formation (threshold behavior)
cd ../H2_formation
python h2_formation.py
```

### Run the NaI parameter scan

```bash
cd NaI_photodissociation
python nai_photodissociation_sensitivityscan_v3\(终稿\).py
```

This performs a 2D grid scan over:
- **De_ionic**: 2.8–3.4 eV (step 0.1 eV)
- **V12 (coupling)**: 0.020–0.034 eV (step 0.002 eV)
- Each point: 3 ensembles × 300 trajectories (coarse) or 5 × 2000 (refined)

---

## Results

### 1. NaI Photodissociation

**Model**: Non-adiabatic, two crossing PESs
- **Virtual space**: Ionic state (Morse well, $D_e=3.30$ eV, $R_0=2.5$ Å) + covalent state (repulsive wall + linear descent)
- **Real space**: Gaussian wavepacket ($\sigma_R=0.15$ Å), initial outward velocity $\bar{v}=5.0$ Å/ps, **initial state: covalent (repulsive branch)**
- **Coupling matrix**: Landau-Zener probability ($V_{12}=0.034$ eV)

**Result**:
- Dissociation yield: **65.3% ± 1.5%**
- Matches Zewail's experimental value: **65%**

| Parameter | Value |
|:---|:---|
| $D_e$ (well depth) | **3.30 eV** |
| $V_{12}$ (electronic coupling) | **0.034 eV** |
| Simulated yield | **65.3% ± 1.5%** |
| Experimental yield (Zewail) | ~65% |

**Figures**:

| Figure | Description |
|:---|:---|
| `nai_potentials.png` | Potential energy surfaces (ionic in red, covalent in blue) |
| `nai_trajectories.png` | Example trajectories (red=ionic, blue=covalent) |
| `2d_scan_heatmap_extended.png` | 2D parameter scan: yield vs. $D_e$ and $V_{12}$ |
| `2d_scan_contour_extended.png` | Contour plot of the same scan |

The system is initially prepared on the covalent (repulsive) state, corresponding to laser excitation. At the crossing ($R_c=6.9$ Å), a Landau-Zener transition may populate the ionic (bound) state. Trajectories that hop to ionic are trapped; those remaining on covalent dissociate.

---

### 2. Cl₂ Photodissociation

**Model**: Adiabatic, single repulsive PES
- **Virtual space**: Exponential repulsive potential $V(R) = 4.0\exp[-2.0(R-1.98)]$ eV
- **Real space**: Gaussian wavepacket ($\sigma_R=0.05$ Å), zero initial velocity
- **Coupling matrix**: Gradient flow (Velocity Verlet)

**Result**:
- Dissociation yield: **100%**
- Mean relative kinetic energy: **3.84 ± 0.39 eV** (1.92 eV per fragment)

| Quantity | Value |
|:---|:---|
| Dissociation yield | **100%** |
| Mean relative kinetic energy | **3.84 ± 0.39 eV** |
| Mean kinetic energy per fragment | **1.92 ± 0.20 eV** |
| Literature range | 3.5–4.0 eV (total) |

**Figures**:

| Figure | Description |
|:---|:---|
| `cl2_potential.png` | Exponential repulsive potential |
| `cl2_kinetic_energy.png` | Fragment kinetic energy distribution |

---

### 3. H₂ Formation

**Model**: Attractive potential, bound state formation
- **Virtual space**: Morse potential ($D_e=4.746$ eV, $\beta=1.942$ Å⁻¹, $R_0=0.741$ Å)
- **Real space**: Fixed bond length at equilibrium, **scanning initial outward velocity** (0–600 Å/ps)
- **Coupling matrix**: Gradient flow (Velocity Verlet)

**Result**:
- $E_{\text{tot}} < 0$ eV: binding probability **= 1.00** (trapped in Morse well)
- $E_{\text{tot}} > 0$ eV: binding probability **drops to 0** (dissociation)
- Sharp transition at the dissociation threshold

| Condition | Binding Probability |
|:---|:---|
| $E_{\text{tot}} < D_e$ | **1.00** (bound) |
| $E_{\text{tot}} > D_e$ | **0.00** (dissociated) |

**Figures**:

| Figure | Description |
|:---|:---|
| `h2_formation_prob.png` | Binding probability vs. initial velocity (threshold behavior) |

---

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** .

**You are free to**:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

**Under the following terms**:
- **Attribution** — You must give appropriate credit, provide a link to the license
- **NonCommercial** — You may not use the material for commercial purposes

For more information: https://creativecommons.org/licenses/by-nc/4.0/
