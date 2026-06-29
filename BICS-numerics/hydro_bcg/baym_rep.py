import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import kn
from scipy.sparse import diags, bmat
import time
import os

print("--- Testing the massless limit: Standard Approach vs WNM ---")

# ==========================================
# 1. Physical parameters and grid
# ==========================================
T0 = 1.5           # 300 MeV
Tvac = 0.13        
tau0 = 1.0     
tau_max = 10.0

# Parameters for the initial density profile (Woods-Saxon)
m_mass = 0.01      # 2 MeV
mu0 = 0.00001      # 0.002 MeV

rmax = 40.0
dr = 0.05
N = int(rmax / dr) + 1
r = np.linspace(0, rmax, N)

# Baym grid (symmetric around r=0)
r_baym = np.linspace(-rmax, rmax, 2 * N - 1)
dr_baym = r_baym[1] - r_baym[0]
cs = 1.0 / np.sqrt(3.0)  

pparam = 7.0
sigma = 0.54

def get_rho(r_grid):
    return 1.0 / (1.0 + np.exp((np.abs(r_grid) - pparam) / sigma))

# ==========================================
# 2. Setup (Thermodynamics & Equations)
# ==========================================
print("preparing tables for m =", m_mass, "...")
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

def hydro_derivatives_massive(tau, y):
    T = y[0:N]; mu = y[N:2*N]; theta = y[2*N:3*N]
    
    T_safe = np.clip(T, T_table[0], T_table[-1])
    mu_T = np.clip(mu / T_safe, -100, 100)
    theta_safe = np.clip(theta, -20, 20)
    
    dT_dr = fast_gradient(T, dr); dmu_dr = fast_gradient(mu, dr); dtheta_dr = fast_gradient(theta, dr)
    dT_dr[0] = 0.0; dmu_dr[0] = 0.0
    
    e0, p0, n0 = interp_e0(T_safe), interp_p0(T_safe), interp_n0(T_safe)
    de0, dp0, dn0 = interp_de0(T_safe), interp_dp0(T_safe), interp_dn0(T_safe)
    
    ch, sh = np.cosh(mu_T), np.sinh(mu_T)
    eps = 2 * ch * e0; P = 2 * ch * p0; n = 2 * sh * n0
    
    deps_dT = 2 * ch * de0 - 2 * (mu_T / T_safe) * sh * e0
    deps_dmu = (2 / T_safe) * sh * e0
    dP_dT = 2 * ch * dp0 - 2 * (mu_T / T_safe) * sh * p0
    dP_dmu = (2 / T_safe) * sh * p0
    dn_dT = 2 * sh * dn0 - 2 * (mu_T / T_safe) * ch * n0
    dn_dmu = (2 / T_safe) * ch * n0
    
    deps_dr = fast_gradient(eps, dr); dP_dr = fast_gradient(P, dr); dn_dr = fast_gradient(n, dr)
    
    cosh_t, sinh_t = np.cosh(theta_safe), np.sinh(theta_safe)
    eps_P = eps + P
    
    M = np.zeros((N, 3, 3))
    M[:, 0, 0] = cosh_t * deps_dT; M[:, 0, 1] = cosh_t * deps_dmu; M[:, 0, 2] = eps_P * sinh_t
    M[:, 1, 0] = sinh_t * dP_dT; M[:, 1, 1] = sinh_t * dP_dmu; M[:, 1, 2] = eps_P * cosh_t
    M[:, 2, 0] = cosh_t * dn_dT; M[:, 2, 1] = cosh_t * dn_dmu; M[:, 2, 2] = n * sinh_t
    
    RHS = np.zeros((N, 3))
    RHS[:, 0] = -sinh_t * deps_dr - eps_P * cosh_t * (1/tau + dtheta_dr) - eps_P * sinh_t / (r+1e-15)
    RHS[:, 1] = -cosh_t * dP_dr - eps_P * sinh_t * dtheta_dr
    RHS[:, 2] = -sinh_t * dn_dr - n * cosh_t * (1/tau + dtheta_dr) - n * sinh_t / (r+1e-15)
    
    dydt_mat = np.linalg.solve(M, RHS[..., np.newaxis]).squeeze(-1) 
    
    dT_dtau = dydt_mat[:, 0]; dmu_dtau = dydt_mat[:, 1]; dtheta_dtau = dydt_mat[:, 2]
    
    nu = 0.05 
    dT_dtau += nu * fast_laplacian(T_safe, dr)
    dmu_dtau += nu * fast_laplacian(mu, dr)
    dtheta_dtau += nu * fast_laplacian(theta_safe, dr)
    dtheta_dtau[0] = 0.0 
    
    return np.concatenate([dT_dtau, dmu_dtau, dtheta_dtau])

def hydro_derivatives_baym(tau, b):
    b_inv = b[::-1]
    v_r = np.tanh((b - b_inv) / 2.0)
    v_char = (v_r + cs) / (1.0 + v_r * cs)
    
    db_dr = np.gradient(b, dr_baym)
    d2b_dr2 = np.gradient(db_dr, dr_baym)
    
    vr_over_r = np.zeros_like(r_baym)
    non_zero = (np.abs(r_baym) > 1e-8)
    vr_over_r[non_zero] = v_r[non_zero] / r_baym[non_zero]
    
    mid_idx = N - 1
    dv_r_dr = np.gradient(v_r, dr_baym) 
    vr_over_r[mid_idx] = dv_r_dr[mid_idx]
    
    source_term = (cs / (1.0 + v_r * cs)) * (vr_over_r + 1.0 / tau)
    
    nu_baym = 0.05 
    db_dtau = -v_char * db_dr - source_term + nu_baym * d2b_dr2
    
    return db_dtau

# ==========================================
# 3. Solving both scenarios
# ==========================================
tau_eval_arr = [1.0, 4.0, 7.0, 10.0]
rho_massive = get_rho(r)
rho_baym = get_rho(r_baym)

ones_diag = np.ones(N); ones_off = np.ones(N-1)
block = diags([ones_off, ones_diag, ones_off], [-1, 0, 1])
jac_sparsity = bmat([[block, block, block], [block, block, block], [block, block, block]], format='csc')

results = []

scenarios = [
    {"name": "Previous Approach", "power": 1.0},    # T proportional to rho
    {"name": "WNM Approach", "power": 1/3.0}        # T proportional to rho^(1/3)
]

for sc in scenarios:
    print(f"\n--- Solving scenario: {sc['name']} ---")
    p = sc["power"]
    
    # Massive IC
    T_init_mass = T0 * rho_massive**p + Tvac
    mu_init_mass = mu0 * rho_massive
    theta_init_mass = 0.01 * np.ones_like(r); theta_init_mass[0] = 0.0
    y0_mass = np.concatenate([T_init_mass, mu_init_mass, theta_init_mass])
    
    print("Massive solver...")
    sol_mass = solve_ivp(hydro_derivatives_massive, (tau0, tau_max), y0_mass, 
                         method='BDF', t_eval=tau_eval_arr, rtol=1e-3, atol=1e-3, jac_sparsity=jac_sparsity)
    
    # Baym IC
    T_init_baym = T0 * rho_baym**p + Tvac
    b_init_baym = np.sqrt(3.0) * np.log(T_init_baym / T0)
    
    print("Baym solver...")
    sol_baym = solve_ivp(hydro_derivatives_baym, (tau0, tau_max), b_init_baym, 
                         method='BDF', t_eval=tau_eval_arr, rtol=1e-3, atol=1e-3)
    
    results.append({"mass": sol_mass, "baym": sol_baym})

# ==========================================
# 4. Preparing the grid for contour extraction and plotting
# ==========================================
os.makedirs("output_plots", exist_ok=True)

import matplotlib.lines as mlines

line_styles = [
    {'tau': 1.0,  'color': 'black', 'ls': '-',  'lw': 2.5},
    {'tau': 4.0,  'color': 'blue',  'ls': '--', 'lw': 2.5},
    {'tau': 7.0,  'color': 'red',   'ls': '-.', 'lw': 2.5},
    {'tau': 10.0, 'color': 'black', 'ls': ':',  'lw': 3.5}
]

label_x = [2.0, 4.0, 7.5, 10.5]

baym_lw = 7.5
baym_alpha = 0.25

baym_proxy = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=baym_lw, alpha=baym_alpha, label="$m=0$ GeV")
massive_proxy = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2.5, label="$m=0.002$ GeV")

# ---------------------------------------------------------
# plot 1: single plot for WNM
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 7))

sol_mass_wnm = results[1]["mass"]
sol_baym_wnm = results[1]["baym"]

for i, config in enumerate(line_styles):
    tau = config['tau']; color = config['color']; ls = config['ls']; lw = config['lw']
    
    # Baym
    b_res = sol_baym_wnm.y[:, i]
    b_inv_res = b_res[::-1]
    T_res_baym_full = T0 * np.exp((b_res + b_inv_res) / (2.0 * np.sqrt(3.0)))
    T_res_baym = T_res_baym_full[N-1:] / T0
    ax1.plot(r, T_res_baym, color=color, linestyle='-', linewidth=baym_lw, alpha=baym_alpha)
    
    # Our model
    T_res_mass = sol_mass_wnm.y[0:N, i] / T0
    ax1.plot(r, T_res_mass, color=color, linestyle=ls, linewidth=lw)

    idx_T = np.searchsorted(r, label_x[i])
    if idx_T < len(r):
        ax1.text(r[idx_T], T_res_mass[idx_T] + 0.05, f'{tau}', color=color, fontsize=16, fontweight='bold', ha='center')

ax1.legend(handles=[baym_proxy, massive_proxy], fontsize=16, loc='upper right', frameon=False)
ax1.set_xlim(0, 20)
ax1.set_ylim(-0.05, 1.25)
ax1.set_xlabel('$r$ [fm]', fontsize=22)
ax1.set_ylabel(r'$T/T_0$', fontsize=22)
ax1.tick_params(axis='both', direction='in', top=True, right=True, labelsize=18)

fig1.tight_layout()
fig1.savefig('output_plots/Massless_Limit_Single_WNM.pdf', format='pdf', dpi=300)

# ---------------------------------------------------------
# plot 2: double plot (Standard vs WNM)
# ---------------------------------------------------------
fig2, axes = plt.subplots(2, 1, figsize=(10, 14), sharex=True, gridspec_kw={'hspace': 0})

panel_labels = ['(a) $T \propto \\rho$', '(b) $T \propto \\rho^{1/3}$']

for col, ax in enumerate(axes):
    sol_mass = results[col]["mass"]
    sol_baym = results[col]["baym"]
    
    for i, config in enumerate(line_styles):
        tau = config['tau']; color = config['color']; ls = config['ls']; lw = config['lw']
        
        # Baym
        b_res = sol_baym.y[:, i]
        b_inv_res = b_res[::-1]
        T_res_baym_full = T0 * np.exp((b_res + b_inv_res) / (2.0 * np.sqrt(3.0)))
        T_res_baym = T_res_baym_full[N-1:] / T0
        ax.plot(r, T_res_baym, color=color, linestyle='-', linewidth=baym_lw, alpha=baym_alpha)
        
        # Our model
        T_res_mass = sol_mass.y[0:N, i] / T0
        ax.plot(r, T_res_mass, color=color, linestyle=ls, linewidth=lw)

        idx_T = np.searchsorted(r, label_x[i])
        if idx_T < len(r):
            ax.text(r[idx_T], T_res_mass[idx_T] + 0.05, f'{tau}', color=color, fontsize=20, fontweight='bold', ha='center')
    
    ax.text(0.98, 0.97, panel_labels[col], transform=ax.transAxes, fontsize=32, ha='right', va='top')
    
    ax.legend(handles=[baym_proxy, massive_proxy], fontsize=20, loc='upper right', bbox_to_anchor=(0.98, 0.88), frameon=False)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.05, 1.25)
    ax.set_ylabel(r'$T/T_0$', fontsize=28)
    
    if col == 1:
        ax.set_xlabel('$r$ [fm]', fontsize=28)
        ax.set_xticks([0, 5, 10, 15, 20])
        
    ax.tick_params(axis='both', direction='in', top=True, right=True, labelsize=24)

fig2.savefig('output_plots/Massless_Limit_2x1.pdf', format='pdf', dpi=300, bbox_inches='tight')
