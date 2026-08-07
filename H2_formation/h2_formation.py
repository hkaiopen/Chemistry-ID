#!/usr/bin/env python3
"""
Information Dynamics Simulation: H₂ Formation (2H → H₂)
========================================================
- Virtual space: Morse potential (attractive, bound state)
- Real space: fixed initial bond length, scanning outward velocity
- Coupling matrix: gradient flow (Velocity Verlet)

Demonstrates the threshold behaviour: when total energy exceeds dissociation
energy, binding probability drops to zero.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Virtual space: Morse potential for H₂ ground state
# ----------------------------------------------------------------------
De = 4.746      # Dissociation energy (eV)
beta = 1.942    # Width parameter (Å⁻¹)
R0 = 0.741      # Equilibrium bond length (Å)

def V_morse(r):
    x = np.exp(-beta * (r - R0))
    return De * (x*x - 2*x)

def dV_dr(r):
    x = np.exp(-beta * (r - R0))
    return 2.0 * De * beta * (x - x*x)

# ----------------------------------------------------------------------
# Real space and coupling matrix (gradient flow)
# ----------------------------------------------------------------------
mu = 1.00784 / 2.0          # Reduced mass (amu)
conv = 9648.5               # eV/Å -> Å/ps² for mass in amu

def acceleration(r):
    return -dV_dr(r) * conv / mu

def trajectory(r0, v0, dt=0.0002, tmax=0.5):
    """
    Propagate a single trajectory.
    Returns True if bound (r < 3.0 at tmax), False if dissociated.
    """
    # Reduce time step for high velocities
    if v0 > 100.0:
        dt = 0.00005
    elif v0 > 50.0:
        dt = 0.0001
    r = r0
    v = v0
    t = 0.0
    while t < tmax and r > 0.2:
        a = acceleration(r)
        v_half = v + 0.5 * a * dt
        r_new = r + v_half * dt
        a_new = acceleration(r_new)
        v_new = v_half + 0.5 * a_new * dt
        r, v = r_new, v_new
        t += dt
        if r > 3.0 and v > 0:
            return False
    return r < 3.0

def main():
    np.random.seed(2024)
    print("Information Dynamics Simulation: H₂ Formation (2H → H₂)")
    print("Virtual space: Morse potential (attractive)")
    print("Real space: fixed bond length, scan initial outward velocity")
    print("Coupling matrix: gradient flow (Velocity Verlet)\n")

    # Fixed bond length at equilibrium; scan velocity from 0 to 600 Å/ps
    r0_fixed = R0
    # Use more points near threshold for smooth curve
    v0_list = np.concatenate([
        np.linspace(0, 300, 15, endpoint=False),
        np.linspace(300, 600, 20)
    ])
    prob_list = []
    n_traj = 200

    print(" v0 (Å/ps)   E_total (eV)   binding prob")
    print("-----------------------------------------")
    for v0 in v0_list:
        E_kin = 0.5 * mu * v0**2 * 1.0364e-4   # eV
        E_pot = V_morse(r0_fixed)               # eV (≈ -De at equilibrium)
        E_total = E_kin + E_pot

        n_success = 0
        for _ in range(n_traj):
            # Small perturbation in initial bond length
            r0_pert = r0_fixed + np.random.normal(0, 0.02)
            success = trajectory(r0_pert, v0, dt=0.0002, tmax=0.5)
            if success:
                n_success += 1
        prob = n_success / n_traj
        prob_list.append(prob)
        # Print only every few points or when significant change
        if abs(prob - 0.5) < 0.1 or v0 % 50 < 5:
            print(f" {v0:6.1f}     {E_total:8.3f}      {prob:5.2f}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(v0_list, prob_list, 'o-', color='blue', linewidth=2)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    plt.xlabel('Initial outward velocity (Å/ps)')
    plt.ylabel('Binding probability')
    plt.title('H₂ formation: binding probability vs. initial kinetic energy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('h2_formation_prob.png', dpi=150)
    plt.show()
    print("\nBinding probability plot saved as 'h2_formation_prob.png'")

    print("\n=== Information Dynamics Interpretation ===")
    print("Virtual space: Morse potential well (attractive)")
    print("Real space: fixed bond length, varying outward velocity")
    print("Coupling matrix: gradient flow (Newtonian dynamics)")
    print("Result: Binding probability drops to zero when total energy exceeds De.")
    threshold_v = np.sqrt(2 * (De - V_morse(r0_fixed)) / (mu * 1.0364e-4))
    print(f"Estimated threshold velocity: {threshold_v:.1f} Å/ps (kinetic energy = De - V(r0))")

if __name__ == "__main__":
    main()
