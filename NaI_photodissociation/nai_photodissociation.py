#!/usr/bin/env python3
"""
Information Dynamics Simulation: NaI Photodissociation
=======================================================
- Virtual space: two potential energy surfaces (ionic Morse + covalent linear)
- Real space: Gaussian wavepacket in position; initial outward velocity
              (approximating momentum from laser excitation)
- Coupling matrix: Landau-Zener transition probability at crossing

Physical model: System is initially excited to the covalent (repulsive) state.
It moves outward, and at the crossing (Rc) there is a probability to hop to
the ionic (bound) state. Dissociation occurs if it remains on the covalent state.
"""

import numpy as np
import matplotlib.pyplot as plt

class NaIDissociationGIP:
    def __init__(self):
        # Virtual space
        self.R0_ionic = 2.5          # equilibrium bond length (Angstrom)
        self.De_ionic = 3.30         # well depth (eV) – to be calibrated
        self.beta_ionic = 1.4        # Morse width (1/Angstrom)
        self.Rc = 6.9                # crossing position (Angstrom)
        self.Vc = 0.0                # potential at crossing (eV)
        self.slope_cov = -2.0        # covalent slope (eV/Angstrom)
        self.wall_height = 10.0      # hard wall for short range

        # Coupling matrix
        self.coupling = 0.025        # electronic coupling (eV) – to be calibrated

        # Physical constants
        self.mass = 23.0 * 126.9 / (23.0 + 126.9) * 1.66054e-27  # reduced mass (kg)
        self.eV_to_J = 1.602e-19
        self.A_to_m = 1e-10
        self.conv = 9648.5           # (Angstrom/ps^2) per (eV/Angstrom) for mass in amu

        # Simulation settings
        self.n_traj = 2000
        self.dt = 0.5e-15            # 0.5 fs
        self.tmax = 5.0e-12          # 5 ps

        # Real space sampling: initial position Gaussian, initial velocity positive
        self.sigma_R = 0.15          # width of wavepacket (Angstrom)
        self.mean_v = 5.0            # initial outward velocity (Angstrom/ps)
        self.sigma_v = 0.2           # small spread in velocity

    def V_ionic(self, R):
        """Morse potential for ionic state."""
        if np.isscalar(R):
            if R < 1.0:
                return self.wall_height
            x = np.exp(-self.beta_ionic * (R - self.R0_ionic))
            return self.De_ionic * (x*x - 2*x)
        else:
            R = np.asarray(R)
            V = np.full_like(R, self.wall_height)
            mask = (R >= 1.0)
            Rclip = R[mask]
            x = np.exp(-self.beta_ionic * (Rclip - self.R0_ionic))
            V[mask] = self.De_ionic * (x*x - 2*x)
            return V

    def V_covalent(self, R):
        """Covalent state: hard wall below Rc, linear descending above."""
        if np.isscalar(R):
            return self.wall_height if R < self.Rc else self.Vc + self.slope_cov * (R - self.Rc)
        else:
            R = np.asarray(R)
            V = np.full_like(R, self.wall_height)
            mask = (R >= self.Rc)
            V[mask] = self.Vc + self.slope_cov * (R[mask] - self.Rc)
            return V

    def dV_ionic(self, R):
        """Derivative of Morse potential: dV/dr = 2*De*beta*(x - x^2)."""
        if np.isscalar(R):
            if R < 1.0:
                return 0.0
            x = np.exp(-self.beta_ionic * (R - self.R0_ionic))
            return 2.0 * self.De_ionic * self.beta_ionic * (x - x*x)
        else:
            R = np.asarray(R)
            grad = np.zeros_like(R)
            mask = (R >= 1.0)
            Rclip = R[mask]
            x = np.exp(-self.beta_ionic * (Rclip - self.R0_ionic))
            grad[mask] = 2.0 * self.De_ionic * self.beta_ionic * (x - x*x)
            return grad

    def dV_covalent(self, R):
        """Derivative of covalent potential."""
        if np.isscalar(R):
            return 0.0 if R < self.Rc else self.slope_cov
        else:
            grad = np.zeros_like(R)
            grad[R >= self.Rc] = self.slope_cov
            return grad

    def landau_zener_prob(self, v):
        """Landau-Zener hopping probability at the crossing."""
        slope_ion = self.dV_ionic(self.Rc)
        slope_cov = self.dV_covalent(self.Rc)
        dF = abs(slope_ion - slope_cov)
        dF_SI = dF * self.eV_to_J / self.A_to_m
        hbar = 1.0545718e-34
        V12 = self.coupling * self.eV_to_J
        if v < 1e-3:
            return 0.0
        exponent = -2.0 * np.pi * V12**2 / (hbar * v * dF_SI)
        exponent = min(0.0, max(-50.0, exponent))
        return np.exp(exponent)

    def initial_conditions(self, n_traj):
        """Sample initial positions from Gaussian, velocities fixed positive."""
        R0 = np.random.normal(loc=self.R0_ionic, scale=self.sigma_R, size=n_traj)
        v0 = np.abs(np.random.normal(loc=self.mean_v, scale=self.sigma_v, size=n_traj))
        return R0, v0

    def propagate(self, R0, v0):
        """
        Propagate a single trajectory.
        Initial state is set to covalent (repulsive branch).
        Returns (dissociated, final_R, final_v, history).
        """
        R = R0 * self.A_to_m          # convert to meters
        v = v0 * 100.0                # Angstrom/ps -> m/s
        state = 'covalent'            # INITIAL STATE: covalent (repulsive)
        t = 0.0
        R_hist = [R / self.A_to_m]
        state_hist = [state]
        t_hist = [t * 1e12]           # ps

        while t < self.tmax:
            R_ang = R / self.A_to_m
            if state == 'ionic':
                force = -self.dV_ionic(R_ang) * self.eV_to_J / self.A_to_m
            else:
                force = -self.dV_covalent(R_ang) * self.eV_to_J / self.A_to_m
            a = force / self.mass

            # Velocity Verlet step 1
            v_half = v + 0.5 * a * self.dt
            R_new = R + v_half * self.dt
            R_ang_new = R_new / self.A_to_m

            # New force
            if state == 'ionic':
                force_new = -self.dV_ionic(R_ang_new) * self.eV_to_J / self.A_to_m
            else:
                force_new = -self.dV_covalent(R_ang_new) * self.eV_to_J / self.A_to_m
            a_new = force_new / self.mass
            v_new = v_half + 0.5 * a_new * self.dt

            # Crossing detection: covalent -> ionic only
            if state == 'covalent' and (R_ang - self.Rc) * (R_ang_new - self.Rc) < 0:
                frac = (self.Rc - R_ang) / (R_ang_new - R_ang)
                v_cross = v + frac * (v_new - v)
                P = self.landau_zener_prob(abs(v_cross))
                if np.random.rand() < P:
                    state = 'ionic'   # Hop to ionic (bound) state
                    # Recompute force after hopping
                    force_new = -self.dV_ionic(R_ang_new) * self.eV_to_J / self.A_to_m
                    a_new = force_new / self.mass
                    v_new = v_half + 0.5 * a_new * self.dt

            R, v, t = R_new, v_new, t + self.dt

            if len(R_hist) < 1000:
                R_hist.append(R / self.A_to_m)
                state_hist.append(state)
                t_hist.append(t * 1e12)

            # If still on covalent and bond length > 15 Å, dissociation is complete
            if state == 'covalent' and (R / self.A_to_m) > 15.0:
                return True, R / self.A_to_m, v / 100.0, (R_hist, state_hist, t_hist)

        # If time runs out without dissociating (likely trapped in ionic well)
        return False, R / self.A_to_m, v / 100.0, (R_hist, state_hist, t_hist)

    def run_ensemble(self, n_traj=None, verbose=False):
        if n_traj is None:
            n_traj = self.n_traj
        R0_arr, v0_arr = self.initial_conditions(n_traj)
        diss_count = 0
        example_trajs = []
        for i in range(n_traj):
            diss, _, _, hist = self.propagate(R0_arr[i], v0_arr[i])
            if diss:
                diss_count += 1
            if i < 5:
                example_trajs.append(hist)
        yield_pct = diss_count / n_traj * 100.0
        if verbose:
            print(f"Ensemble yield: {yield_pct:.1f}%")
        return yield_pct, example_trajs

    def plot_potentials(self, filename='nai_potentials.png'):
        R_vals = np.linspace(1.5, 12.0, 500)
        V_ion = [self.V_ionic(R) for R in R_vals]
        V_cov = [self.V_covalent(R) for R in R_vals]
        plt.figure(figsize=(8,5))
        plt.plot(R_vals, V_ion, 'r-', lw=2, label='Ionic state')
        plt.plot(R_vals, V_cov, 'b-', lw=2, label='Covalent state')
        plt.axvline(x=self.Rc, color='k', linestyle='--', label=f'Crossing at {self.Rc} Å')
        plt.xlabel('Internuclear distance (Å)')
        plt.ylabel('Potential energy (eV)')
        plt.title('Virtual Space: NaI Potential Energy Surfaces')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Potential curves saved as '{filename}'")

    def plot_trajectories(self, example_trajs, filename='nai_trajectories.png'):
        plt.figure(figsize=(10,6))
        for R_hist, state_hist, t_hist in example_trajs:
            R_ion, R_cov = [], []
            t_ion, t_cov = [], []
            for j, s in enumerate(state_hist):
                if s == 'ionic':
                    R_ion.append(R_hist[j])
                    t_ion.append(t_hist[j])
                else:
                    R_cov.append(R_hist[j])
                    t_cov.append(t_hist[j])
            plt.plot(t_ion, R_ion, 'r-', alpha=0.7, lw=1)
            plt.plot(t_cov, R_cov, 'b-', alpha=0.7, lw=1)
        plt.xlabel('Time (ps)')
        plt.ylabel('Internuclear distance (Å)')
        plt.title('Real Space Trajectories (Red=ionic, Blue=covalent)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Trajectory plot saved as '{filename}'")


def multi_run(n_ensembles=5, n_traj=2000):
    yields = []
    example_trajs = None
    for i in range(n_ensembles):
        np.random.seed(2024 + i * 37)
        sim = NaIDissociationGIP()
        sim.n_traj = n_traj
        y, ex = sim.run_ensemble(verbose=False)
        yields.append(y)
        if i == 0:
            example_trajs = ex
        print(f"Run {i+1}: yield = {y:.1f}%")
    mean_yield = np.mean(yields)
    std_yield = np.std(yields, ddof=1)
    ci95 = 1.96 * std_yield / np.sqrt(n_ensembles)
    print(f"\nMean yield: {mean_yield:.1f}% ± {std_yield:.1f}% (std dev)")
    print(f"95% CI: [{mean_yield - ci95:.1f}, {mean_yield + ci95:.1f}]%")
    return mean_yield, std_yield, example_trajs


def main():
    print("Information Dynamics Simulation: NaI Photodissociation")
    print("Virtual space: Morse well + dissociative slope")
    print("Real space: Gaussian wavepacket, mean outward velocity = 5.0 Å/ps")
    print("Coupling matrix: Landau-Zener (hopping covalent -> ionic)")
    print("Initial state: covalent (repulsive branch)\n")

    sim = NaIDissociationGIP()
    sim.plot_potentials('nai_potentials.png')

    mean_yield, std_yield, example_trajs = multi_run(n_ensembles=5, n_traj=2000)

    sim.plot_trajectories(example_trajs, 'nai_trajectories.png')

    print("\n=== Information Dynamics Interpretation ===")
    print(f"Virtual space: Morse well (De={sim.De_ionic} eV) + dissociative slope")
    print(f"Real space: Gaussian wavepacket, mean_v={sim.mean_v} Å/ps")
    print(f"Coupling matrix: Landau-Zener (V12={sim.coupling} eV)")
    print(f"Result: Dissociation yield = {mean_yield:.1f}% ± {std_yield:.1f}%")
    print("(Calibrate V12 to match experimental ~65% dissociation)")


if __name__ == "__main__":
    main()
