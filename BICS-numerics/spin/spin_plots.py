import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import kn
from scipy.sparse import diags, bmat
import time
import os

print("--- Initializing Hydrodynamic Background Model ---")

# ==========================================
# 1. Physical Parameters and Grid
# ==========================================
T0 = 1.5        # Initial temperature (300 MeV)
Tvac = 0.13     # Vacuum temperature
tau0 = 1.0      # Initial time
tau_max = 10  # Final time

masses = [1.5]
mass_labels = ['300 MeV']
m = masses[0]

N = 900        
r0 = 0.0      
rmax = 20.0
r = np.linspace(r0, rmax, N)
dr = r[1] - r[0]

# Initial conditions (Woods-Saxon)
pparam = 7.0
sigma = 0.54
rho = 1.0 / (1.0 + np.exp((r - pparam) / sigma))

T_init = T0 * rho**(1/3) + Tvac
mu_init = 0.15 * rho # small chemical potential (rho_0 = 0.15 is about 30 MeV)

# Dirichlet boundary condition: velocity at r=0, so theta(0) = 0
theta_init = 0.01 * np.ones_like(r)
theta_init[0] = 0.0 

y0_bg = np.concatenate([T_init, mu_init, theta_init])
tau_eval = np.linspace(tau0, tau_max, 50)

# ==========================================
# 2. Auxiliary Functions & Sparsity
# ==========================================
def calc_A(T, mu, m):
    z = m / T
    return - (1.0 / (3.0 * np.pi**2)) * m * T**2 * np.cosh(mu/T) * kn(3, z)

def calc_A1(T, mu, m):
    z = m / T
    return (1.0 / (6.0 * np.pi**2)) * m * T**2 * np.cosh(mu/T) * (z * kn(2, z) + 2.0 * kn(3, z))
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

# Sparsity for background (3 variables: T, mu, theta)
ones_diag = np.ones(N)
ones_off = np.ones(N-1)
block = diags([ones_off, ones_diag, ones_off], offsets=[-1, 0, 1])

jac_sparsity_bg = bmat([
    [block, block, block],
    [block, block, block],
    [block, block, block]
], format='csc')

# Sparsity for spin (6 variables)
jac_sparsity_spin = bmat([[block]*6]*6, format='csc')

# ==========================================
# 3. Solving Background Hydrodynamics
# ==========================================
total_start_time = time.time()

print("\n--- Interpolating Background Equation of State ---")
T_table = np.linspace(0.01, 3.5, 5000)
e0_arr = m**2 * T_table**2 * (3 * kn(2, m/T_table) + (m/T_table) * kn(1, m/T_table))
p0_arr = m**2 * T_table**2 * kn(2, m/T_table)
n0_arr = m**2 * T_table * kn(2, m/T_table)

de0_arr = np.gradient(e0_arr, T_table)
dp0_arr = np.gradient(p0_arr, T_table)
dn0_arr = np.gradient(n0_arr, T_table)

interp_e0 = CubicSpline(T_table, e0_arr)
interp_p0 = CubicSpline(T_table, p0_arr)
interp_n0 = CubicSpline(T_table, n0_arr)
interp_de0 = CubicSpline(T_table, de0_arr)
interp_dp0 = CubicSpline(T_table, dp0_arr)
interp_dn0 = CubicSpline(T_table, dn0_arr)

def hydro_derivatives_massive(tau, y):
    T = y[0:N]
    mu = y[N:2*N]
    theta = y[2*N:3*N]
    
    # numerical safeguards for interpolation (clipping)
    T_safe = np.clip(T, T_table[0], T_table[-1])
    mu_T = np.clip(mu / T_safe, -100, 100)
    theta_safe = np.clip(theta, -20, 20)
    
    # numerical derivatives
    dT_dr = fast_gradient(T_safe, dr)
    dmu_dr = fast_gradient(mu, dr)
    dtheta_dr = fast_gradient(theta_safe, dr)
    
    # numerical safeguards for boundary conditions (clipping)
    dT_dr[0] = 0.0
    dmu_dr[0] = 0.0
    
    e0, p0, n0 = interp_e0(T_safe), interp_p0(T_safe), interp_n0(T_safe)
    de0, dp0, dn0 = interp_de0(T_safe), interp_dp0(T_safe), interp_dn0(T_safe)
    
    ch = np.cosh(mu_T)
    sh = np.sinh(mu_T)
    
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
    
    cosh_t = np.cosh(theta_safe)
    sinh_t = np.sinh(theta_safe)
    eps_P = eps + P
    
    # numerical safeguards for geometry (sinh(theta)/r) at r=0
    sinh_over_r = np.zeros_like(r)
    sinh_over_r[1:] = sinh_t[1:] / r[1:]
    sinh_over_r[0] = dtheta_dr[0] * cosh_t[0] # from l'Hospital's rule
    
    # constructing the M matrix and RHS vector
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
    
    # constructing the RHS vector with numerical safeguards for geometry
    RHS = np.zeros((N, 3))
    RHS[:, 0] = -sinh_t * deps_dr - eps_P * cosh_t * (1/tau + dtheta_dr) - eps_P * sinh_over_r
    RHS[:, 1] = -cosh_t * dP_dr - eps_P * sinh_t * dtheta_dr
    RHS[:, 2] = -sinh_t * dn_dr - n * cosh_t * (1/tau + dtheta_dr) - n * sinh_over_r
    
    # solving the linear system M * dydt = RHS for dydt
    dydt_mat = np.linalg.solve(M, RHS[..., np.newaxis]).squeeze(-1) 
    
    dT_dtau = dydt_mat[:, 0]
    dmu_dtau = dydt_mat[:, 1]
    dtheta_dtau = dydt_mat[:, 2]
    
    # --- small numerical viscosity for stability ---
    nu = 0.05
    dT_dtau += nu * fast_laplacian(T_safe, dr)
    dmu_dtau += nu * fast_laplacian(mu, dr)
    dtheta_dtau += nu * fast_laplacian(theta_safe, dr)
    
    # enforce boundary condition for theta at r=0 (velocity must be zero)
    dtheta_dtau[0] = 0.0 
    
    return np.concatenate([dT_dtau, dmu_dtau, dtheta_dtau])

start_time = time.time()
print("Starting BDF Solver for Background...")
solution_bg = solve_ivp(
    hydro_derivatives_massive, 
    (tau0, tau_max), 
    y0_bg, 
    method='BDF', 
    t_eval=tau_eval, 
    dense_output=True, # to allow interpolation for spin solver
    rtol=1e-3, 
    atol=1e-3,
    jac_sparsity=jac_sparsity_bg 
)
print(f"Background hydrodynamics completed! Time: {time.time() - start_time:.2f} s.")

# ==========================================
# 4. Solving Spin Evolution via Interpolation
# ==========================================
print("\n--- Initializing Spin Evolution based on Background Interpolation ---")

Ckx_init = 0.1 * rho
Cwx_init = 0.1 * rho
Cky_init = 0.0 * (r / rmax) * rho 
Cwz_init = 0.1 * rho
Ckz_init = 0.1 * rho
Cwy_init = 0.0 * (r / rmax) * rho

y0_spin = np.concatenate([Ckx_init, Cwx_init, Cky_init, Cwz_init, Ckz_init, Cwy_init])

def spin_derivatives(tau, y_spin):
    Ckx = y_spin[0:N]
    Cwx = y_spin[N:2*N]
    Cky = y_spin[2*N:3*N]
    Cwz = y_spin[3*N:4*N]
    Ckz = y_spin[4*N:5*N]
    Cwy = y_spin[5*N:6*N]

    y_bg = solution_bg.sol(tau)
    T = y_bg[0:N]
    mu = y_bg[N:2*N]
    theta = y_bg[2*N:3*N]
    
    # numerical safeguards for interpolation (clipping)
    T_safe = np.clip(T, 0.05, 3.5)
    mu_safe = np.clip(mu, -100*T_safe, 100*T_safe)
    theta_safe = np.clip(theta, -20, 20)

    # numerical derivatives for background 
    dy_bg = hydro_derivatives_massive(tau, y_bg)
    dT_dtau = dy_bg[0:N]
    dmu_dtau = dy_bg[N:2*N]
    dtheta_dtau = dy_bg[2*N:3*N]
    
    dtheta_dr = fast_gradient(theta_safe, dr)

    A_arr = calc_A(T_safe, mu_safe, m)
    A1_arr = calc_A1(T_safe, mu_safe, m)

    eps_diff = 1e-5
    dA_dT = (calc_A(T_safe + eps_diff, mu_safe, m) - calc_A(T_safe - eps_diff, mu_safe, m)) / (2 * eps_diff)
    dA_dmu = (calc_A(T_safe, mu_safe + eps_diff, m) - calc_A(T_safe, mu_safe - eps_diff, m)) / (2 * eps_diff)
    dA_dr = fast_gradient(A_arr, dr)
    dA_dtau = dA_dT * dT_dtau + dA_dmu * dmu_dtau

    dA1_dT = (calc_A1(T_safe + eps_diff, mu_safe, m) - calc_A1(T_safe - eps_diff, mu_safe, m)) / (2 * eps_diff)
    dA1_dmu = (calc_A1(T_safe, mu_safe + eps_diff, m) - calc_A1(T_safe, mu_safe - eps_diff, m)) / (2 * eps_diff)
    dA1_dr = fast_gradient(A1_arr, dr)
    dA1_dtau = dA1_dT * dT_dtau + dA1_dmu * dmu_dtau

    dCkx_dr = fast_gradient(Ckx, dr); dCkx_dr[0] = 0.0
    dCwx_dr = fast_gradient(Cwx, dr); dCwx_dr[0] = 0.0
    
    dCky_dr = fast_gradient(Cky, dr)
    dCwz_dr = fast_gradient(Cwz, dr)
    dCkz_dr = fast_gradient(Ckz, dr)
    dCwy_dr = fast_gradient(Cwy, dr)

    cosh_t = np.cosh(theta_safe)
    sinh_t = np.sinh(theta_safe)
    tanh_t = np.tanh(theta_safe)
    
    tanh_over_r = np.zeros_like(r)
    tanh_over_r[1:] = tanh_t[1:] / r[1:]
    tanh_over_r[0] = dtheta_dr[0] / (cosh_t[0]**2)
    
    sinh_over_r = np.zeros_like(r)
    sinh_over_r[1:] = sinh_t[1:] / r[1:]
    sinh_over_r[0] = dtheta_dr[0] * cosh_t[0]

    Cky_over_r = np.zeros_like(r)
    Cky_over_r[1:] = Cky[1:] / r[1:]
    Cky_over_r[0] = dCky_dr[0]

    Cwz_over_r = np.zeros_like(r)
    Cwz_over_r[1:] = Cwz[1:] / r[1:]
    Cwz_over_r[0] = dCwz_dr[0]

    Ckz_over_r = np.zeros_like(r)
    Ckz_over_r[1:] = Ckz[1:] / r[1:]
    Ckz_over_r[0] = dCkz_dr[0]

    Cwy_over_r = np.zeros_like(r)
    Cwy_over_r[1:] = Cwy[1:] / r[1:]
    Cwy_over_r[0] = dCwy_dr[0]

    term_geom = 1.0/tau + tanh_over_r

    # --- 1. System for X components ---
    dCwx_dtau = (A_arr * Cwx / (2.0 * A1_arr)) * term_geom \
                - (dA1_dtau / A1_arr) * Cwx \
                - (dA1_dr / A1_arr) * Cwx * tanh_t \
                - Cwx * dtheta_dr \
                - Cwx * dtheta_dtau * tanh_t \
                - Cwx * term_geom \
                - dCwx_dr * tanh_t

    dCkx_dtau = - (dA_dtau / A_arr) * Ckx \
                - (dA_dr / A_arr) * Ckx * tanh_t \
                - Ckx * dtheta_dr \
                - Ckx * dtheta_dtau * tanh_t \
                - 1.5 * Ckx * term_geom \
                - dCkx_dr * tanh_t

    # --- 2. System for Y and Z components ---
    M_Y1 = np.zeros((N, 2, 2))
    M_Y1[:, 0, 0] = - (A_arr / 2.0) * sinh_t
    M_Y1[:, 0, 1] = - A1_arr * cosh_t
    M_Y1[:, 1, 0] = A_arr * cosh_t
    M_Y1[:, 1, 1] = - (A_arr / 2.0) * sinh_t
    
    RHS_Y1 = np.zeros((N, 2))
    RHS_Y1[:, 0] = -(
        - (dA_dtau * Cky * sinh_t) / 2.0 
        - (dA_dr * Cky * cosh_t) / 2.0 
        + (A_arr / 2.0) * (
            - 3.0 * Cky * dtheta_dr * sinh_t 
            - 3.0 * Cky * dtheta_dtau * cosh_t 
            - Cky_over_r * cosh_t 
            - dCky_dr * cosh_t 
            + Cwz * dtheta_dr * cosh_t 
            + Cwz * dtheta_dtau * sinh_t 
            + Cwz_over_r * sinh_t
        ) 
        - dA1_dtau * Cwz * cosh_t 
        - dA1_dr * Cwz * sinh_t 
        - A1_arr * (
            Cwz * dtheta_dr * cosh_t 
            + Cwz * dtheta_dtau * sinh_t 
            + Cwz * cosh_t / tau 
            + Cwz_over_r * sinh_t 
            + dCwz_dr * sinh_t
        )
    )
    
    RHS_Y1[:, 1] = -(
        dA_dtau * (Cky * cosh_t - 0.5 * Cwz * sinh_t)
        + dA_dr * (Cky * sinh_t - 0.5 * Cwz * cosh_t)
        + A_arr * (
            1.5 * Cky * dtheta_dr * cosh_t
            + 1.5 * Cky * dtheta_dtau * sinh_t
            + 1.5 * Cky * cosh_t / tau
            + Cky_over_r * sinh_t
            + dCky_dr * sinh_t
            - 0.5 * Cwz * dtheta_dr * sinh_t
            - 0.5 * Cwz * dtheta_dtau * cosh_t
            - 0.5 * Cwz * sinh_t / tau
            - 0.5 * dCwz_dr * cosh_t
        )
        + A1_arr * Cwz * (dtheta_dr * sinh_t + dtheta_dtau * cosh_t)
    )

    sol_Y1 = np.linalg.solve(M_Y1, RHS_Y1[..., np.newaxis]).squeeze(-1)
    dCky_dtau = sol_Y1[:, 0]
    dCwz_dtau = sol_Y1[:, 1]

    # --- 3. System for Z and Y components ---
    M_Z1 = np.zeros((N, 2, 2))
    M_Z1[:, 0, 0] = (A_arr / 2.0) * sinh_t
    M_Z1[:, 0, 1] = - A1_arr * cosh_t
    M_Z1[:, 1, 0] = A_arr * cosh_t
    M_Z1[:, 1, 1] = (A_arr / 2.0) * sinh_t
    
    RHS_Z1 = np.zeros((N, 2))
    RHS_Z1[:, 0] = -(
        (dA_dtau * Ckz * sinh_t) / 2.0
        + (dA_dr * Ckz * cosh_t) / 2.0
        + (A_arr / 2.0) * (
            3.0 * Ckz * dtheta_dr * sinh_t
            + 3.0 * Ckz * dtheta_dtau * cosh_t
            + Ckz * sinh_t / tau
            + dCkz_dr * cosh_t
            + Cwy * dtheta_dr * cosh_t
            + Cwy * dtheta_dtau * sinh_t
            + Cwy * cosh_t / tau
        )
        - dA1_dtau * Cwy * cosh_t
        - dA1_dr * Cwy * sinh_t
        - A1_arr * (
            Cwy * dtheta_dr * cosh_t
            + Cwy * dtheta_dtau * sinh_t
            + Cwy * cosh_t / tau
            + Cwy_over_r * sinh_t
            + dCwy_dr * sinh_t
        )
    )
    
    RHS_Z1[:, 1] = -(
        dA_dtau * (Ckz * cosh_t + 0.5 * Cwy * sinh_t)
        + dA_dr * (Ckz * sinh_t + 0.5 * Cwy * cosh_t)
        + A_arr * (
            1.5 * Ckz * dtheta_dr * cosh_t
            + 1.5 * Ckz * dtheta_dtau * sinh_t
            + Ckz * cosh_t / tau
            + 1.5 * Ckz_over_r * sinh_t
            + dCkz_dr * sinh_t
            + 0.5 * Cwy * dtheta_dr * sinh_t
            + 0.5 * Cwy * dtheta_dtau * cosh_t
            + 0.5 * Cwy_over_r * cosh_t
            + 0.5 * dCwy_dr * cosh_t
        )
        - A1_arr * Cwy * (dtheta_dr * sinh_t + dtheta_dtau * cosh_t)
    )

    sol_Z1 = np.linalg.solve(M_Z1, RHS_Z1[..., np.newaxis]).squeeze(-1)
    dCkz_dtau = sol_Z1[:, 0]
    dCwy_dtau = sol_Z1[:, 1]

    # --- Numerical viscosity ---
    nu_spin = 0.02 
    dCkx_dtau += nu_spin * fast_laplacian(Ckx, dr)
    dCwx_dtau += nu_spin * fast_laplacian(Cwx, dr)
    dCky_dtau += nu_spin * fast_laplacian(Cky, dr)
    dCwz_dtau += nu_spin * fast_laplacian(Cwz, dr)
    dCkz_dtau += nu_spin * fast_laplacian(Ckz, dr)
    dCwy_dtau += nu_spin * fast_laplacian(Cwy, dr)

    dCky_dtau[0] = 0.0
    dCwy_dtau[0] = 0.0

    return np.concatenate([dCkx_dtau, dCwx_dtau, dCky_dtau, dCwz_dtau, dCkz_dtau, dCwy_dtau])

start_time = time.time()
solution_spin = solve_ivp(
    spin_derivatives, 
    (tau0, tau_max), 
    y0_spin, 
    method='BDF', 
    t_eval=tau_eval, 
    rtol=1e-3, 
    atol=1e-3,
    jac_sparsity=jac_sparsity_spin # jacobian sparsity for spin solver
)
print(f"Spin evolution completed! Time: {time.time() - start_time:.2f} s.")

os.makedirs("wykresy_spin1", exist_ok=True)

# ==========================================
# 5. Plots
# ==========================================
import matplotlib.lines as mlines
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

print("\nGenerating plotss...")
os.makedirs("wykresy_spin_final", exist_ok=True)

Ckx_mat = solution_spin.y[0:N, :]
Cwx_mat = solution_spin.y[N:2*N, :]
Cky_mat = solution_spin.y[2*N:3*N, :]
Cwz_mat = solution_spin.y[3*N:4*N, :]
Ckz_mat = solution_spin.y[4*N:5*N, :]
Cwy_mat = solution_spin.y[5*N:6*N, :]

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(wspace=0.0, hspace=0.0)

tau_eval_plot = [1.0, 4.0, 7.0, 10.0]

styles = [
    {'color': 'black', 'ls': '-',  'lw': 3.5}, 
    {'color': 'blue',  'ls': '--', 'lw': 3.5}, 
    {'color': 'red',   'ls': '-.', 'lw': 3.5}, 
    {'color': 'black', 'ls': ':',  'lw': 4.0}  
]

def plot_panel(ax, mat1, mat2, name1, name2, y_label, 
               label_x, track_mat2, y_offsets, panel_label,
               is_bottom=False, is_right=False):
    
    tau_indices = [np.argmin(np.abs(solution_spin.t - t)) for t in tau_eval_plot]

    for i, (idx, tau, lx, track, y_off_factor) in enumerate(zip(tau_indices, tau_eval_plot, label_x, track_mat2, y_offsets)):
        c = styles[i]['color']
        ls = styles[i]['ls']
        lw_thick = styles[i]['lw']
        lw_sec = 3.0   
        alpha_sec = 0.5
        
        ax.plot(r, mat1[:, idx], color=c, linestyle=ls, lw=lw_thick)
        
        if mat2 is not None:
            ax.plot(r, mat2[:, idx], color=c, linestyle=ls, lw=lw_sec, alpha=alpha_sec)
        
        lbl_idx = np.searchsorted(r, lx)
        if lbl_idx < len(r):
            if mat2 is not None and track:
                y_val = mat2[lbl_idx, idx]
            else:
                y_val = mat1[lbl_idx, idx]
            
            y_span = np.max(mat1) - np.min(mat1)
            if mat2 is not None:
                y_span = max(y_span, np.max(mat2) - np.min(mat2))
            
            y_offset_actual = y_span * y_off_factor
  
            ax.text(r[lbl_idx], y_val + y_offset_actual, f'{tau}', color=c, fontsize=20, fontweight='bold',
                    ha='center', va='center')
    
    lines = [mlines.Line2D([], [], color='gray', linestyle='-', lw=3.5, label=name1)]
    if mat2 is not None:
        lines.append(mlines.Line2D([], [], color='gray', linestyle='-', lw=3.0, alpha=0.5, label=name2))
    
    ax.legend(handles=lines, loc='upper right', bbox_to_anchor=(0.98, 0.85), fontsize=20, frameon=False)
    
    ax.set_xlim(0, 20)
    ax.grid(False) 
    
    ax.tick_params(axis='both', which='both', direction='in', 
                   top=True, right=True, bottom=True, left=True, labelsize=24)
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    if is_bottom:
        ax.set_xlabel('r [fm]', fontsize=28)
        
    if is_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        
    ax.set_ylabel(y_label, fontsize=28)
    
    ax.text(0.95, 0.95, panel_label, transform=ax.transAxes, 
            fontsize=28, fontweight='normal', ha='right', va='top')

# ==================== Panels ====================

lx_1 = [2.0, 5.5, 10.0, 14.0] 
tr_1 = [False, False, False, False] 
yo_1 = [0.06, 0.06, 0.06, 0.06]    

plot_panel(axs[0, 0], Ckx_mat, None, r'$C_{kr}$', None, r'$C_{kr}$', 
           label_x=lx_1, track_mat2=tr_1, y_offsets=yo_1, panel_label=r'(a) $C_{kr}=0.1$', is_bottom=False, is_right=False)

lx_2 = [2.5, 6.0, 10.5, 15.0]
tr_2 = [False, False, False, False]
yo_2 = [0.06, 0.06, 0.06, 0.06]

plot_panel(axs[0, 1], Cwx_mat, None, r'$C_{\omega r}$', None, r'$C_{\omega r}$', 
           label_x=lx_2, track_mat2=tr_2, y_offsets=yo_2, panel_label=r'(b) $C_{\omega r}=0.1$', is_bottom=False, is_right=True)

lx_3 = [2.0, 5.0, 11.0, 16.5]
tr_3 = [False, False, True, True] 
yo_3 = [0.08, 0.08, -0.08, 0.08]  

plot_panel(axs[1, 0], Cwz_mat, Cky_mat, r'$C_{\omega z}$', r'$C_{k\phi}$', r'$C_{\omega z}$, $C_{k\phi}$', 
           label_x=lx_3, track_mat2=tr_3, y_offsets=yo_3, panel_label=r'(c) $C_{\omega z}=0.1$', is_bottom=True, is_right=False)

lx_4 = [2.0, 6.0, 11.0, 16.0]
tr_4 = [False, False, True, True]
yo_4 = [0.08, 0.08, 0.08, 0.08]

plot_panel(axs[1, 1], Ckz_mat, Cwy_mat, r'$C_{kz}$', r'$C_{\omega \phi}$', r'$C_{kz}$, $C_{\omega \phi}$', 
           label_x=lx_4, track_mat2=tr_4, y_offsets=yo_4, panel_label=r'(d) $C_{kz}=0.1$', is_bottom=True, is_right=True)

for ax in axs.flat:
    ax.set_xticks([0, 5, 10, 15, 20])

axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticklabels([])

axs[1, 0].set_xticklabels(['0', '5', '10', '15', '']) 
axs[1, 1].set_xticklabels(['0', '5', '10', '15', '20'])  

filename = 'wykresy_spin_final/Spin_300MeV_all.pdf'
fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')

