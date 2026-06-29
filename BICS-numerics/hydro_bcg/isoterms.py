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
# 1. Physical parameters and grid
# ==========================================
T0 = 1.5       # 300 MeV
Tvac = 0.13    
tau0 = 1.0     

dr = 0.05
rmax = 40.0    # max radius for the grid to capture the full evolution of the isotherm
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
# 2. Helper functions and sparse matrix
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
# 3. Function to run the hydrodynamic simulation for a given mass and extract the isotherm
# ==========================================
def run_hydro_simulation(m_mass, tau_max_val, tau_eval_arr):
    print(f"\n--- Preparing simulation for mass in code m={m_mass} ---")
    
    # Thermodynamic tables and interpolation
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
    print(f"final tau = {tau_max_val} fm...")
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
    print(f"Simulation completed in: {time.time() - t0:.2f} s.")
    
    return sol

# ==========================================
# 4. Running simulations for both masses and preparing the grid for contour extraction
# ==========================================
tau_max = 23.0
# grid for contour extraction
tau_eval_times = np.linspace(tau0, tau_max, 250)

# 1000 MeV -> 5.0
sol_1000 = run_hydro_simulation(m_mass=5.0, tau_max_val=tau_max, tau_eval_arr=tau_eval_times)

# 300 MeV -> 1.5
sol_300  = run_hydro_simulation(m_mass=1.5, tau_max_val=tau_max, tau_eval_arr=tau_eval_times)

# ==========================================
# 5. Function to extract the isotherm (T=150 MeV) from the simulation results
# ==========================================
def extract_isotherm(sol_obj, tau_array, target_T):
    # T_mat shape: (num_tau, N) where each row corresponds to T(r) at a given tau
    T_mat = sol_obj.y[0:N, :].T 
    R_grid, TAU_grid = np.meshgrid(r, tau_array)
    
    fig_temp, ax_temp = plt.subplots()
    cs = ax_temp.contour(R_grid, TAU_grid, T_mat, levels=[target_T])
    paths = cs.get_paths()
    plt.close(fig_temp)
    
    if not paths:
        return [], []
        
    main_path = max(paths, key=lambda p: len(p.vertices))
    vertices = main_path.vertices
    return vertices[:, 0], vertices[:, 1]

# Extracting the isotherm for T=150 MeV
T_freeze = 0.75 
r_1000, tau_1000 = extract_isotherm(sol_1000, tau_eval_times, T_freeze)
r_300, tau_300   = extract_isotherm(sol_300, tau_eval_times, T_freeze)

# ==========================================
# 6. Plotting the isotherms for both masses
# ==========================================
os.makedirs("output_plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 7))

if len(r_1000) > 0:
    ax.plot(r_1000, tau_1000, color='red', linestyle='-', linewidth=3.5, label=r'$m = 1$ GeV')
if len(r_300) > 0:
    ax.plot(r_300, tau_300, color='blue', linestyle='--', linewidth=3.5, label=r'$m = 0.3$ GeV')


ax.set_xlabel(r'$r$ [fm]', fontsize=28)
ax.set_ylabel(r'$\tau$ [fm]', fontsize=28)

ax.set_xlim(0, 15)
ax.set_ylim(tau0, tau_max)

ax.tick_params(axis='both', direction='in', top=True, right=True, labelsize=24, length=6, width=1.5)

ax.legend(fontsize=20, loc='upper right', frameon=False, handlelength=2.5)

fig.tight_layout()
file_path = 'output_plots/Freezeout.pdf'
fig.savefig(file_path, format='pdf', dpi=300)
