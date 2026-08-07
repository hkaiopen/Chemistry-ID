#!/usr/bin/env python3
"""
2D grid scan for NaI photodissociation calibration
============================================================================
Initial state: covalent (repulsive branch).
Extended coupling up to 0.032 eV to find the 65% yield region.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class NaIDissociationGIP:
    def __init__(self, De_ionic=3.30, coupling=0.025):
        self.R0_ionic = 2.5
        self.De_ionic = De_ionic
        self.beta_ionic = 1.4
        self.Rc = 6.9
        self.Vc = 0.0
        self.slope_cov = -2.0
        self.wall_height = 10.0

        self.coupling = coupling

        self.mass = 23.0 * 126.9 / (23.0 + 126.9) * 1.66054e-27
        self.eV_to_J = 1.602e-19
        self.A_to_m = 1e-10
        self.conv = 9648.5

        self.n_traj = 1000
        self.dt = 0.5e-15
        self.tmax = 5.0e-12

        self.sigma_R = 0.15
        self.mean_v = 5.0
        self.sigma_v = 0.2

    def V_ionic(self, R):
        if np.isscalar(R):
            if R < 1.0:
                return self.wall_height
            x = np.exp(-self.beta_ionic * (R - self.R0_ionic))
            return self.De_ionic * (x*x - 2*x)
        else:
            R = np.asarray(R)
            V = np.full_like(R, self.wall_height)
            mask = (R >= 1.0)
            Rc = R[mask]
            x = np.exp(-self.beta_ionic * (Rc - self.R0_ionic))
            V[mask] = self.De_ionic * (x*x - 2*x)
            return V

    def V_covalent(self, R):
        if np.isscalar(R):
            return self.wall_height if R < self.Rc else self.Vc + self.slope_cov * (R - self.Rc)
        else:
            R = np.asarray(R)
            V = np.full_like(R, self.wall_height)
            mask = (R >= self.Rc)
            V[mask] = self.Vc + self.slope_cov * (R[mask] - self.Rc)
            return V

    def dV_ionic(self, R):
        if np.isscalar(R):
            if R < 1.0:
                return 0.0
            x = np.exp(-self.beta_ionic * (R - self.R0_ionic))
            return 2.0 * self.De_ionic * self.beta_ionic * (x - x*x)
        else:
            R = np.asarray(R)
            grad = np.zeros_like(R)
            mask = (R >= 1.0)
            Rc = R[mask]
            x = np.exp(-self.beta_ionic * (Rc - self.R0_ionic))
            grad[mask] = 2.0 * self.De_ionic * self.beta_ionic * (x - x*x)
            return grad

    def dV_covalent(self, R):
        if np.isscalar(R):
            return 0.0 if R < self.Rc else self.slope_cov
        else:
            grad = np.zeros_like(R)
            grad[R >= self.Rc] = self.slope_cov
            return grad

    def landau_zener_prob(self, v):
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
        R0 = np.random.normal(loc=self.R0_ionic, scale=self.sigma_R, size=n_traj)
        v0 = np.abs(np.random.normal(loc=self.mean_v, scale=self.sigma_v, size=n_traj))
        return R0, v0

    def propagate(self, R0, v0):
        R = R0 * self.A_to_m
        v = v0 * 100.0
        state = 'covalent'            # INITIAL STATE: covalent (repulsive)
        t = 0.0
        while t < self.tmax:
            R_ang = R / self.A_to_m
            if state == 'ionic':
                force = -self.dV_ionic(R_ang) * self.eV_to_J / self.A_to_m
            else:
                force = -self.dV_covalent(R_ang) * self.eV_to_J / self.A_to_m
            a = force / self.mass

            v_half = v + 0.5 * a * self.dt
            R_new = R + v_half * self.dt
            R_ang_new = R_new / self.A_to_m

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
                    state = 'ionic'
                    force_new = -self.dV_ionic(R_ang_new) * self.eV_to_J / self.A_to_m
                    a_new = force_new / self.mass
                    v_new = v_half + 0.5 * a_new * self.dt

            R, v, t = R_new, v_new, t + self.dt

            if state == 'covalent' and (R / self.A_to_m) > 15.0:
                return True
        return False

    def run_ensemble(self, n_traj=None):
        if n_traj is None:
            n_traj = self.n_traj
        R0_arr, v0_arr = self.initial_conditions(n_traj)
        diss = 0
        for i in range(n_traj):
            if self.propagate(R0_arr[i], v0_arr[i]):
                diss += 1
        return diss / n_traj * 100.0


def run_grid_point(args):
    De, coup, n_traj, n_ensembles, base_seed = args
    yields = []
    for i in range(n_ensembles):
        seed = base_seed + i * 37 + int(De*100) + int(coup*1000)
        np.random.seed(seed)
        sim = NaIDissociationGIP(De_ionic=De, coupling=coup)
        sim.n_traj = n_traj
        y = sim.run_ensemble()
        yields.append(y)
    mean_y = np.mean(yields)
    std_y = np.std(yields, ddof=1) if n_ensembles > 1 else 0.0
    return (De, coup, mean_y, std_y)


def main():
    # ---------- CONFIGURATION (FINE SCAN) ----------
    N_TRAJ = 2000
    N_ENSEMBLES = 5
    BASE_SEED = 2024

    De_vals = [3.10, 3.20, 3.30]
    coup_vals = [0.030, 0.031, 0.032, 0.033, 0.034]
    # -----------------------------------------------

    print("2D grid scan for NaI photodissociation (Extended coupling)")
    print(f"Trajectories per ensemble: {N_TRAJ}")
    print(f"Ensembles per point: {N_ENSEMBLES}")
    print(f"Using multi-processing on {cpu_count()} cores.")
    print(f"De range: {De_vals}")
    print(f"Coupling range: {coup_vals}\n")

    args_list = []
    for De, coup in product(De_vals, coup_vals):
        args_list.append((De, coup, N_TRAJ, N_ENSEMBLES, BASE_SEED))

    start = time.time()

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_grid_point, args_list), total=len(args_list)))

    elapsed = time.time() - start
    print(f"\nScan completed in {elapsed:.0f} seconds.")

    results_dict = {}
    for De, coup, mean_y, std_y in results:
        results_dict[(De, coup)] = (mean_y, std_y)
        print(f"De = {De:.2f} eV, coupling = {coup:.3f} eV → yield = {mean_y:.1f}% ± {std_y:.2f}%")

    best = min(results_dict.items(), key=lambda x: abs(x[1][0] - 65.0))
    (De_best, coup_best), (y_best, std_best) = best
    print(f"\n>>> Best combination: De = {De_best:.2f} eV, coupling = {coup_best:.3f} eV")
    print(f"    yield = {y_best:.1f}% ± {std_best:.2f}% (target ~65%)")

    # Heatmap
    X, Y = np.meshgrid(De_vals, coup_vals)
    Z_mean = np.zeros_like(X, dtype=float)
    Z_std = np.zeros_like(X, dtype=float)
    for i, De in enumerate(De_vals):
        for j, coup in enumerate(coup_vals):
            mean_y, std_y = results_dict[(De, coup)]
            Z_mean[j, i] = mean_y
            Z_std[j, i] = std_y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(Z_mean, extent=[De_vals[0], De_vals[-1], coup_vals[0], coup_vals[-1]],
                     origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
    ax1.set_xlabel('De_ionic (eV)')
    ax1.set_ylabel('Coupling V12 (eV)')
    ax1.set_title('Mean dissociation yield (%)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(Z_std, extent=[De_vals[0], De_vals[-1], coup_vals[0], coup_vals[-1]],
                     origin='lower', aspect='auto', cmap='plasma', interpolation='bilinear')
    ax2.set_xlabel('De_ionic (eV)')
    ax2.set_ylabel('Coupling V12 (eV)')
    ax2.set_title('Standard deviation (%)')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('2d_scan_heatmap_extended.png', dpi=150)
    plt.close()
    print("Heatmap saved as '2d_scan_heatmap_extended.png'")

    # Contour
    plt.figure(figsize=(8,6))
    contour = plt.contour(X, Y, Z_mean, levels=8, cmap='coolwarm')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('De_ionic (eV)')
    plt.ylabel('Coupling V12 (eV)')
    plt.title('Contour plot of dissociation yield (%)')
    plt.colorbar(contour)
    plt.savefig('2d_scan_contour_extended.png', dpi=150)
    plt.close()
    print("Contour plot saved as '2d_scan_contour_extended.png'")

    with open('2d_scan_results_extended.txt', 'w') as f:
        f.write("De_ionic (eV), coupling (eV), mean_yield (%), std (%)\n")
        for (De, coup), (mean_y, std_y) in results_dict.items():
            f.write(f"{De:.2f}, {coup:.3f}, {mean_y:.2f}, {std_y:.2f}\n")
    print("Numerical results saved as '2d_scan_results_extended.txt'.")


if __name__ == "__main__":
    main()
