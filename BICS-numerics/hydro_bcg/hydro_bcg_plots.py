import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import kn
from scipy.sparse import diags, bmat
import time
import os

print("--- Initializing Hydrodynamic Model ---")

# ==========================================
# Physical parameters and grid
# ==========================================
T0 = 1.5    
Tvac = 0.13    
tau0 = 1.0     

dr = 0.05
rmax = 40.0    # computations up to rmax = 40 fm for boundary conditions safety
N = int(rmax / dr) + 1
r = np.linspace(0, rmax, N)

def get_initial_conditions(r_grid):
    pparam = 7.0
    sigma = 0.54
    # Woods-Saxon density profile
    rho = 1.0 / (1.0 + np.exp((r_grid - pparam) / sigma))
    
    T = T0 * rho**(1/3) + Tvac
    mu = 0.15 * rho  # 30 MeV
    theta = 0.01 * np.ones_like(r_grid)
    theta[0] = 0.0 
    return np.concatenate([T, mu, theta])

y0 = get_initial_conditions(r)

mu0 = y0[N] # mu(r=0)

# ==========================================
# Helper functions and sparse matrix
# ==========================================
def fast_gradient(arr, dr_step):
    darr = np.zeros_like(arr)
    darr[1:-1] = (arr[2:] - arr[:-2]) / (2 * dr_step)
    darr[0] = (arr[1] - arr[0]) / dr_step
    darr[-1] = (arr[-1] - arr[-2]) / dr_step
    return darr

def fast_laplacian(arr, dr_step):
    d2arr = np.zeros_like(arr)
    d2arr[1:-1] = (arr[2:] - 2*arr[1:-1] + arr[:-2]) / (dr_step**2)
    return d2arr

# Jacobian sparsity pattern
ones_diag = np.ones(N)
ones_off = np.ones(N-1)
block = diags([ones_off, ones_diag, ones_off], [-1, 0, 1])
jac_sparsity = bmat([[block, block, block],
                     [block, block, block],
                     [block, block, block]], format='csc')

# ==========================================
# Hydrodynamic simulation function
# ==========================================
def run_hydro_simulation(m_mass, tau_max_val, tau_eval_arr):
    print(f"\n--- Preparing simulation for mass in code m={m_mass} ---")
    
    # Thermodynamic tables and interpolation (dependent on the given mass)
    T_table = np.linspace(0.01, 3.5, 5000)
    
    e0_arr = m_mass**2 * T_table**2 * (3 * kn(2, m_mass/T_table) + (m_mass/T_table) * kn(1, m_mass/T_table))
    p0_arr = m_mass**2 * T_table**2 * kn(2, m_mass/T_table)
    n0_arr = m_mass**2 * T_table * kn(2, m_mass/T_table)
    
    de0_arr = np.gradient(e0_arr, T_table)
    dp0_arr = np.gradient(p0_arr, T_table)
    dn0_arr = np.gradient(n0_arr, T_table)
    
    interp_e0 = CubicSpline(T_table, e0_arr)
    interp_p0 = CubicSpline(T_table, p0_arr)
    interp_n0 = CubicSpline(T_table, n0_arr)
    interp_de0 = CubicSpline(T_table, de0_arr)
    interp_dp0 = CubicSpline(T_table, dp0_arr)
    interp_dn0 = CubicSpline(T_table, dn0_arr)

    # Hydrodynamic equations
    def hydro_derivatives(tau, y):
        T = y[0:N]
        mu = y[N:2*N]
        theta = y[2*N:3*N]
        
        T_safe = np.clip(T, T_table[0], T_table[-1])
        mu_T = np.clip(mu / T_safe, -100, 100)
        theta_safe = np.clip(theta, -20, 20)
        
        dT_dr = fast_gradient(T, dr)
        dmu_dr = fast_gradient(mu, dr)
        dtheta_dr = fast_gradient(theta, dr)
        
        dT_dr[0] = 0.0
        dmu_dr[0] = 0.0
        
        e0, p0, n0 = interp_e0(T_safe), interp_p0(T_safe), interp_n0(T_safe)
        de0, dp0, dn0 = interp_de0(T_safe), interp_dp0(T_safe), interp_dn0(T_safe)
        
        ch, sh = np.cosh(mu_T), np.sinh(mu_T)
        
        eps = 2 * ch * e0
        P = 2 * ch * p0
        n = 2 * sh * n0
        
        deps_dT = 2 * ch * de0 - 2 * (mu_T / T_safe) * sh * e0
        deps_dmu = (2 / T_safe) * sh * e0
        dP_dT = 2 * ch * dp0 - 2 * (mu_T / T_safe) * sh * p0
        dP_dmu = (2 / T_safe) * sh * p0
        dn_dT = 2 * sh * dn0 - 2 * (mu_T / T_safe) * ch * n0
        dn_dmu = (2 / T_safe) * ch * n0
        
        deps_dr = fast_gradient(eps, dr)
        dP_dr = fast_gradient(P, dr)
        dn_dr = fast_gradient(n, dr)
        
        cosh_t, sinh_t = np.cosh(theta_safe), np.sinh(theta_safe)
        eps_P = eps + P
        
        M = np.zeros((N, 3, 3))
        M[:, 0, 0] = cosh_t * deps_dT
        M[:, 0, 1] = cosh_t * deps_dmu
        M[:, 0, 2] = eps_P * sinh_t
        M[:, 1, 0] = sinh_t * dP_dT
        M[:, 1, 1] = sinh_t * dP_dmu
        M[:, 1, 2] = eps_P * cosh_t
        M[:, 2, 0] = cosh_t * dn_dT
        M[:, 2, 1] = cosh_t * dn_dmu
        M[:, 2, 2] = n * sinh_t
        
        RHS = np.zeros((N, 3))
        RHS[:, 0] = -sinh_t * deps_dr - eps_P * cosh_t * (1/tau + dtheta_dr) - eps_P * sinh_t / (r+1e-15)
        RHS[:, 1] = -cosh_t * dP_dr - eps_P * sinh_t * dtheta_dr
        RHS[:, 2] = -sinh_t * dn_dr - n * cosh_t * (1/tau + dtheta_dr) - n * sinh_t / (r+1e-15)
        
        dydt_mat = np.linalg.solve(M, RHS[..., np.newaxis]).squeeze(-1) 
        
        dT_dtau = dydt_mat[:, 0]
        dmu_dtau = dydt_mat[:, 1]
        dtheta_dtau = dydt_mat[:, 2]
        
        # Numerical viscosity
        nu = 0.05 
        dT_dtau += nu * fast_laplacian(T_safe, dr)
        dmu_dtau += nu * fast_laplacian(mu, dr)
        dtheta_dtau += nu * fast_laplacian(theta_safe, dr)
        
        dtheta_dtau[0] = 0.0 
        
        return np.concatenate([dT_dtau, dmu_dtau, dtheta_dtau])

    # Solving the system
    print(f"Solving up to time tau = {tau_max_val} fm/c...")
    t0 = time.time()
    sol = solve_ivp(
        hydro_derivatives, 
        (tau0, tau_max_val), 
        y0, 
        method='BDF', 
        t_eval=tau_eval_arr, 
        rtol=1e-3, 
        atol=1e-3,
        jac_sparsity=jac_sparsity 
    )
    print(f"Finish. Time: {time.time() - t0:.2f} s.")
    
    return sol

# ==========================================
# Simulation for both masses
# ==========================================
tau_eval_times = [1.0, 4.0, 7.0, 10.0]
tau_max = 10.0

# 1000 MeV 
sol_1000 = run_hydro_simulation(m_mass=5.0, tau_max_val=tau_max, tau_eval_arr=tau_eval_times)

# 300 MeV 
sol_300  = run_hydro_simulation(m_mass=1.5, tau_max_val=tau_max, tau_eval_arr=tau_eval_times)

# ==========================================
# Plotting results
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("output_plots", exist_ok=True)

line_styles = [
    {'tau': 1.0,  'color': 'black', 'ls': '-',  'lw': 3.0}, 
    {'tau': 4.0,  'color': 'blue',  'ls': '--', 'lw': 3.0}, 
    {'tau': 7.0,  'color': 'red',   'ls': '-.', 'lw': 3.0}, 
    {'tau': 10.0, 'color': 'black', 'ls': ':',  'lw': 3.5}  
]

label_x = [2.0, 4.5, 7.5, 10.5]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(wspace=0, hspace=0)

solutions = [sol_1000, sol_300]

for col, sol in enumerate(solutions):
    for i, config in enumerate(line_styles):
        tau = config['tau']
        color = config['color']
        ls = config['ls']
        lw = config['lw']
        
        T_res = sol.y[0:N, i] / T0
        mu_res = sol.y[N:2*N, i] / mu0
        
        ax_T = axes[0, col]
        ax_T.plot(r, T_res, color=color, linestyle=ls, linewidth=lw)
        
        idx_T = np.searchsorted(r, label_x[i])
        if idx_T < len(r):
            ax_T.text(r[idx_T], T_res[idx_T] + 0.05, f'{tau}', 
                      color=color, fontsize=20, fontweight='bold', ha='center')
        
        ax_mu = axes[1, col]
        ax_mu.plot(r, mu_res, color=color, linestyle=ls, linewidth=lw)
        
        idx_mu = np.searchsorted(r, label_x[i])
        if idx_mu < len(r):
            ax_mu.text(r[idx_mu], mu_res[idx_mu] + 0.05, f'{tau}', 
                       color=color, fontsize=20, fontweight='bold', ha='center')

axes[0, 0].text(0.95, 0.93, '(a)', transform=axes[0, 0].transAxes, fontsize=32, fontweight='normal', ha='right', va='top')
axes[0, 1].text(0.95, 0.93, '(b)', transform=axes[0, 1].transAxes, fontsize=32, fontweight='normal', ha='right', va='top')
axes[1, 0].text(0.95, 0.93, '(c)', transform=axes[1, 0].transAxes, fontsize=32, fontweight='normal', ha='right', va='top')
axes[1, 1].text(0.95, 0.93, '(d)', transform=axes[1, 1].transAxes, fontsize=32, fontweight='normal', ha='right', va='top')

for ax in axes.flat:
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis='both', direction='in', top=True, right=True, labelsize=24)
    
    ax.set_ylim(-0.05, 1.25)
    ax.set_yticks([0.0, 0.4, 0.8, 1.2])

axes[0, 0].set_xticklabels([])
axes[0, 1].set_xticklabels([])

axes[1, 0].set_xticklabels(['0', '5', '10', '15', ''])
axes[1, 1].set_xticklabels(['0', '5', '10', '15', '20'])


axes[0, 0].set_yticklabels(['', '0.4', '0.8', '1.2'])
axes[0, 1].set_yticklabels([])

axes[1, 0].set_yticklabels(['0.0', '0.4', '0.8', '1.2'])
axes[1, 1].set_yticklabels([])

axes[0, 0].set_ylabel(r'$T/T_0$', fontsize=28)
axes[1, 0].set_ylabel(r'$\mu/\mu_0$', fontsize=28)
axes[1, 0].set_xlabel('$r$ [fm]', fontsize=28)
axes[1, 1].set_xlabel('$r$ [fm]', fontsize=28)

file_path_pdf = 'output_plots/Hydro_2x2_comparison_v3.pdf'
fig.savefig(file_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
