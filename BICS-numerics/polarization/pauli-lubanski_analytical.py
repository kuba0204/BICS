import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.special import kn, iv
from scipy.sparse import diags, bmat
import time
import os

print("--- Initialization and Hydro Evolution ---")
start_time_total = time.time()

# ==========================================
# 1. Parameters and Initial Conditions
# ==========================================
T0 = 1.5        # 300 MeV
Tvac = 0.13     # Vacuum temperature
tau0 = 1.0     
tau_max = 22    
m = 5         # setting the mass to 300 MeV (1.5) or 1000 MeV (5.0)

N = 900      
r0 = 0.0     
rmax = 40.0    
r = np.linspace(r0, rmax, N)
dr = r[1] - r[0]

# Initial conditions (Woods-Saxon)
pparam, sigma = 7.0, 0.54
rho = 1.0 / (1.0 + np.exp((r - pparam) / sigma))

T_init = T0 * rho**(1/3) + Tvac
mu_init = 0.15 * rho 

# Initial velocity profile (theta) - small initial flow
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

# Sparsity Pattern
ones_diag = np.ones(N)
ones_off = np.ones(N-1)
block = diags([ones_off, ones_diag, ones_off], offsets=[-1, 0, 1])

jac_sparsity_bg = bmat([
    [block, block, block],
    [block, block, block],
    [block, block, block]
], format='csc')

jac_sparsity_spin = bmat([[block]*6]*6, format='csc')

# ==========================================
# 3. EoS i Hydro Evolution for Background
# ==========================================
print("--- Interpolating Background Equation of State ---")
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
    T, mu, theta = y[0:N], y[N:2*N], y[2*N:3*N]
    
    # safeguards
    T_safe = np.clip(T, T_table[0], T_table[-1])
    mu_T = np.clip(mu / T_safe, -100, 100)
    theta_safe = np.clip(theta, -20, 20)
    
    dT_dr = fast_gradient(T_safe, dr)
    dmu_dr = fast_gradient(mu, dr)
    dtheta_dr = fast_gradient(theta_safe, dr)
    
    dT_dr[0] = 0.0
    dmu_dr[0] = 0.0
    
    e0, p0, n0 = interp_e0(T_safe), interp_p0(T_safe), interp_n0(T_safe)
    de0, dp0, dn0 = interp_de0(T_safe), interp_dp0(T_safe), interp_dn0(T_safe)
    
    ch, sh = np.cosh(mu_T), np.sinh(mu_T)
    eps, P, n_dens = 2 * ch * e0, 2 * ch * p0, 2 * sh * n0
    
    deps_dT = 2 * ch * de0 - 2 * (mu_T / T_safe) * sh * e0
    deps_dmu = (2 / T_safe) * sh * e0
    dP_dT = 2 * ch * dp0 - 2 * (mu_T / T_safe) * sh * p0
    dP_dmu = (2 / T_safe) * sh * p0
    dn_dT = 2 * sh * dn0 - 2 * (mu_T / T_safe) * ch * n0
    dn_dmu = (2 / T_safe) * ch * n0
    
    deps_dr = fast_gradient(eps, dr)
    dP_dr = fast_gradient(P, dr)
    dn_dr = fast_gradient(n_dens, dr)
    
    cosh_t, sinh_t = np.cosh(theta_safe), np.sinh(theta_safe)
    eps_P = eps + P
    
    sinh_over_r = np.zeros_like(r)
    sinh_over_r[1:] = sinh_t[1:] / r[1:]
    sinh_over_r[0] = dtheta_dr[0] * cosh_t[0] 
    
    M = np.zeros((N, 3, 3))
    M[:, 0, 0] = cosh_t * deps_dT; M[:, 0, 1] = cosh_t * deps_dmu; M[:, 0, 2] = eps_P * sinh_t
    M[:, 1, 0] = sinh_t * dP_dT;   M[:, 1, 1] = sinh_t * dP_dmu;   M[:, 1, 2] = eps_P * cosh_t
    M[:, 2, 0] = cosh_t * dn_dT;   M[:, 2, 1] = cosh_t * dn_dmu;   M[:, 2, 2] = n_dens * sinh_t
    
    RHS = np.zeros((N, 3))
    RHS[:, 0] = -sinh_t * deps_dr - eps_P * cosh_t * (1/tau + dtheta_dr) - eps_P * sinh_over_r
    RHS[:, 1] = -cosh_t * dP_dr - eps_P * sinh_t * dtheta_dr
    RHS[:, 2] = -sinh_t * dn_dr - n_dens * cosh_t * (1/tau + dtheta_dr) - n_dens * sinh_over_r
    
    dydt_mat = np.linalg.solve(M, RHS[..., np.newaxis]).squeeze(-1) 
    
    dT_dtau = dydt_mat[:, 0]
    dmu_dtau = dydt_mat[:, 1]
    dtheta_dtau = dydt_mat[:, 2]
    
    nu = 0.05  
    dT_dtau += nu * fast_laplacian(T_safe, dr)
    dmu_dtau += nu * fast_laplacian(mu, dr)
    dtheta_dtau += nu * fast_laplacian(theta_safe, dr)
    
    dtheta_dtau[0] = 0.0
    
    return np.concatenate([dT_dtau, dmu_dtau, dtheta_dtau])

print("Starting BDF Solver for Background...")
solution_bg = solve_ivp(
    hydro_derivatives_massive, 
    (tau0, tau_max), 
    y0_bg, 
    method='BDF', 
    t_eval=tau_eval, 
    dense_output=True, 
    rtol=1e-3, 
    atol=1e-3,
    jac_sparsity=jac_sparsity_bg
)

if solution_bg.success:
    print(f"Hydro background evolution completed. Time: {time.time() - start_time_total:.2f} s")
else:
    print(f"Hydro STOPPED EARLY at tau = {solution_bg.t[-1]:.2f}. Reason: {solution_bg.message}")

# ==========================================
# 4. Thermal Freeze-out Isotherm Extraction
# ==========================================
print("--- Thermal Freeze-out Isotherm ---")
T_matrix = solution_bg.y[0:N, :].T / T0 
R_mesh, TAU_mesh = np.meshgrid(r, tau_eval)

fig_tmp, ax_tmp = plt.subplots()
cs_iso = ax_tmp.contour(R_mesh, TAU_mesh, T_matrix, levels=[0.65])
plt.close(fig_tmp)

paths = cs_iso.get_paths()
if len(paths) > 0:
    main_path = max(paths, key=lambda p: len(p.vertices))
    r_iso = main_path.vertices[:, 0]
    tau_iso = main_path.vertices[:, 1]
    print(f"Extracted {len(r_iso)} points for the isotherm curve.")
else:
    print("WARNING: Isotherm contour not found! Adjust levels or tau_max.")
    r_iso, tau_iso = np.array([]), np.array([])

# ==========================================
# 5. spin derivatives
# ==========================================
print("--- Spin bcg derivatives ---")

def spin_derivatives(tau, y_spin):
    Ckx, Cwx, Cky, Cwz, Ckz, Cwy = y_spin[0:N], y_spin[N:2*N], y_spin[2*N:3*N], y_spin[3*N:4*N], y_spin[4*N:5*N], y_spin[5*N:6*N]
    
    y_bg = solution_bg.sol(tau)
    T, mu, theta = y_bg[0:N], y_bg[N:2*N], y_bg[2*N:3*N]
    
    T_safe = np.clip(T, 0.05, 3.5)
    mu_safe = np.clip(mu, -100*T_safe, 100*T_safe)
    theta_safe = np.clip(theta, -20, 20)
    
    dy_bg = hydro_derivatives_massive(tau, y_bg)
    dT_dtau, dmu_dtau, dtheta_dtau = dy_bg[0:N], dy_bg[N:2*N], dy_bg[2*N:3*N]
    
    dtheta_dr = fast_gradient(theta_safe, dr)
    
    A_arr, A1_arr = calc_A(T_safe, mu_safe, m), calc_A1(T_safe, mu_safe, m)
    eps_diff = 1e-5
    
    dA_dT = (calc_A(T_safe+eps_diff, mu_safe, m) - calc_A(T_safe-eps_diff, mu_safe, m))/(2*eps_diff)
    dA_dmu = (calc_A(T_safe, mu_safe+eps_diff, m) - calc_A(T_safe, mu_safe-eps_diff, m))/(2*eps_diff)
    dA_dr = fast_gradient(A_arr, dr)
    dA_dtau = dA_dT * dT_dtau + dA_dmu * dmu_dtau

    dA1_dT = (calc_A1(T_safe+eps_diff, mu_safe, m) - calc_A1(T_safe-eps_diff, mu_safe, m))/(2*eps_diff)
    dA1_dmu = (calc_A1(T_safe, mu_safe+eps_diff, m) - calc_A1(T_safe, mu_safe-eps_diff, m))/(2*eps_diff)
    dA1_dr = fast_gradient(A1_arr, dr)
    dA1_dtau = dA1_dT * dT_dtau + dA1_dmu * dmu_dtau

    dCkx_dr, dCwx_dr = fast_gradient(Ckx, dr), fast_gradient(Cwx, dr)
    dCkx_dr[0] = 0.0; dCwx_dr[0] = 0.0
    
    dCky_dr, dCwy_dr = fast_gradient(Cky, dr), fast_gradient(Cwy, dr)
    
    dCwz_dr, dCkz_dr = fast_gradient(Cwz, dr), fast_gradient(Ckz, dr)
    dCwz_dr[0] = 0.0; dCkz_dr[0] = 0.0

    cosh_t, sinh_t, tanh_t = np.cosh(theta_safe), np.sinh(theta_safe), np.tanh(theta_safe)
    
    tanh_over_r = np.zeros_like(r)
    tanh_over_r[1:] = tanh_t[1:]/r[1:]
    tanh_over_r[0] = dtheta_dr[0]/(cosh_t[0]**2)
    
    sinh_over_r = np.zeros_like(r)
    sinh_over_r[1:] = sinh_t[1:] / r[1:]
    sinh_over_r[0] = dtheta_dr[0] * cosh_t[0]

    Cky_over_r = np.zeros_like(r);  Cky_over_r[1:] = Cky[1:]/r[1:];   Cky_over_r[0] = dCky_dr[0]
    Cwz_over_r = np.zeros_like(r);  Cwz_over_r[1:] = Cwz[1:]/r[1:];   Cwz_over_r[0] = dCwz_dr[0]
    Ckz_over_r = np.zeros_like(r);  Ckz_over_r[1:] = Ckz[1:]/r[1:];   Ckz_over_r[0] = dCkz_dr[0]
    Cwy_over_r = np.zeros_like(r);  Cwy_over_r[1:] = Cwy[1:]/r[1:];   Cwy_over_r[0] = dCwy_dr[0]

    term_geom = 1.0/tau + tanh_over_r

    dCwx_dtau = (A_arr*Cwx/(2.0*A1_arr))*term_geom - (dA1_dtau/A1_arr)*Cwx - (dA1_dr/A1_arr)*Cwx*tanh_t - Cwx*dtheta_dr - Cwx*dtheta_dtau*tanh_t - Cwx*term_geom - dCwx_dr*tanh_t
    dCkx_dtau = - (dA_dtau/A_arr)*Ckx - (dA_dr/A_arr)*Ckx*tanh_t - Ckx*dtheta_dr - Ckx*dtheta_dtau*tanh_t - 1.5*Ckx*term_geom - dCkx_dr*tanh_t

    M_Y1 = np.zeros((N, 2, 2))
    M_Y1[:, 0, 0] = - (A_arr / 2.0) * sinh_t; M_Y1[:, 0, 1] = - A1_arr * cosh_t
    M_Y1[:, 1, 0] = A_arr * cosh_t;           M_Y1[:, 1, 1] = - (A_arr / 2.0) * sinh_t
    RHS_Y1 = np.zeros((N, 2))
    
    RHS_Y1[:, 0] = -( -(dA_dtau*Cky*sinh_t)/2.0 - (dA_dr*Cky*cosh_t)/2.0 + (A_arr/2.0)*(-3.0*Cky*dtheta_dr*sinh_t - 3.0*Cky*dtheta_dtau*cosh_t - Cky_over_r*cosh_t - dCky_dr*cosh_t + Cwz*dtheta_dr*cosh_t + Cwz*dtheta_dtau*sinh_t + Cwz*sinh_over_r) - dA1_dtau*Cwz*cosh_t - dA1_dr*Cwz*sinh_t - A1_arr*(Cwz*dtheta_dr*cosh_t + Cwz*dtheta_dtau*sinh_t + Cwz*cosh_t/tau + Cwz*sinh_over_r + dCwz_dr*sinh_t) )
    RHS_Y1[:, 1] = -( dA_dtau*(Cky*cosh_t - 0.5*Cwz*sinh_t) + dA_dr*(Cky*sinh_t - 0.5*Cwz*cosh_t) + A_arr*(1.5*Cky*dtheta_dr*cosh_t + 1.5*Cky*dtheta_dtau*sinh_t + 1.5*Cky*cosh_t/tau + Cky_over_r*sinh_t + dCky_dr*sinh_t - 0.5*Cwz*dtheta_dr*sinh_t - 0.5*Cwz*dtheta_dtau*cosh_t - 0.5*Cwz*sinh_t/tau - 0.5*dCwz_dr*cosh_t) + A1_arr*Cwz*(dtheta_dr*sinh_t + dtheta_dtau*cosh_t) )
    sol_Y1 = np.linalg.solve(M_Y1, RHS_Y1[..., np.newaxis]).squeeze(-1)
    dCky_dtau = sol_Y1[:, 0]
    dCwz_dtau = sol_Y1[:, 1]

    M_Z1 = np.zeros((N, 2, 2))
    M_Z1[:, 0, 0] = (A_arr / 2.0) * sinh_t; M_Z1[:, 0, 1] = - A1_arr * cosh_t
    M_Z1[:, 1, 0] = A_arr * cosh_t;         M_Z1[:, 1, 1] = (A_arr / 2.0) * sinh_t
    RHS_Z1 = np.zeros((N, 2))
    RHS_Z1[:, 0] = -( (dA_dtau*Ckz*sinh_t)/2.0 + (dA_dr*Ckz*cosh_t)/2.0 + (A_arr/2.0)*(3.0*Ckz*dtheta_dr*sinh_t + 3.0*Ckz*dtheta_dtau*cosh_t + Ckz*sinh_t/tau + dCkz_dr*cosh_t + Cwy*dtheta_dr*cosh_t + Cwy*dtheta_dtau*sinh_t + Cwy*cosh_t/tau) - dA1_dtau*Cwy*cosh_t - dA1_dr*Cwy*sinh_t - A1_arr*(Cwy*dtheta_dr*cosh_t + Cwy*dtheta_dtau*sinh_t + Cwy*cosh_t/tau + Cwy_over_r*sinh_t + dCwy_dr*sinh_t) )
    RHS_Z1[:, 1] = -( dA_dtau*(Ckz*cosh_t + 0.5*Cwy*sinh_t) + dA_dr*(Ckz*sinh_t + 0.5*Cwy*cosh_t) + A_arr*(1.5*Ckz*dtheta_dr*cosh_t + 1.5*Ckz*dtheta_dtau*sinh_t + Ckz*cosh_t/tau + 1.5*Ckz*sinh_over_r + dCkz_dr*sinh_t + 0.5*Cwy*dtheta_dr*sinh_t + 0.5*Cwy*dtheta_dtau*cosh_t + 0.5*Cwy_over_r*cosh_t + 0.5*dCwy_dr*cosh_t) - A1_arr*Cwy*(dtheta_dr*sinh_t + dtheta_dtau*cosh_t) )
    sol_Z1 = np.linalg.solve(M_Z1, RHS_Z1[..., np.newaxis]).squeeze(-1)
    dCkz_dtau = sol_Z1[:, 0]
    dCwy_dtau = sol_Z1[:, 1]

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


# ==========================================
# 6. Pauli-Lubański vector (Analytical Expressions)
# ==========================================
def compute_polarization_maps(px_arr, py_arr, C_curves, T_curve, theta_curve, r_iso_c, tau_iso_c, m_particle=1.116):
    C_kx_c, C_wx_c, C_ky_c, C_wz_c, C_kz_c, C_wy_c = C_curves
    
    # Derivatives over a parameter (index i) along the isotherm curve
    dr_curve, dtau_curve = np.gradient(r_iso_c), np.gradient(tau_iso_c)
    
    pi_PRF_x_map = np.zeros((len(py_arr), len(px_arr)))
    pi_PRF_y_map = np.zeros((len(py_arr), len(px_arr)))
    pi_PRF_z_map = np.zeros((len(py_arr), len(px_arr)))
    
    # Mask: taking only r >= 1e-3
    valid = r_iso_c >= 1e-3
    r_val = r_iso_c[valid]
    tau_val = tau_iso_c[valid]
    dr_val = dr_curve[valid]
    dtau_val = dtau_curve[valid]
    vol_element = r_val * tau_val
    
    # protection against zero temperature (which would cause singularities in the Bessel functions)
    T = np.maximum(T_curve[valid], 1e-3)
    theta_p = theta_curve[valid]
    
    # protection against zero theta (which would cause singularities in coth and csch)
    theta_p_safe = np.where(np.abs(theta_p) > 1e-10, theta_p, 
                            np.where(theta_p >= 0, 1e-10, -1e-10))
    
    ch_theta = np.cosh(theta_p_safe)
    sh_theta = np.sinh(theta_p_safe)
    coth_theta = 1.0 / np.tanh(theta_p_safe)
    csch_theta = 1.0 / sh_theta
    
    # mapping C_curves to the valid domain
    C_kr = C_kx_c[valid]
    C_wr = C_wx_c[valid]
    C_kp = C_ky_c[valid]
    C_wz = C_wz_c[valid]
    C_kz = C_kz_c[valid]
    C_wp = C_wy_c[valid]
    
    for i_y, py in enumerate(py_arr):
        for i_x, px in enumerate(px_arr):
            p_T = np.sqrt(px**2 + py**2)
            m_T = np.sqrt(m_particle**2 + p_T**2)
            phi_p = np.arctan2(py, px)
            
            # safeguard against zero transverse momentum 
            p_T_safe = max(p_T, 1e-12)
            
            sin_pp = np.sin(phi_p)
            cos_pp = np.cos(phi_p)
            
            # arguments for Bessel functions
            z = m_T * ch_theta / T
            w = p_T_safe * sh_theta / T
            
            K0, K1, K2 = kn(0, z), kn(1, z), kn(2, z)
            I0, I1, I2 = iv(0, w), iv(1, w), iv(2, w)
            kappa = K0 + K2
            
            # components F_mu
            F0 = 2 * np.pi * C_wr * p_T_safe * kappa * I1
            
            F1 = 4 * np.pi * p_T_safe * sin_pp * (C_kz * ch_theta - C_wp * sh_theta) * K1 * I0 \
               - 2 * np.pi * m_T * kappa * I1 * (C_kz * sh_theta * sin_pp - C_wp * ch_theta * sin_pp + C_wr * cos_pp)
               
            F2 = 4 * np.pi * p_T_safe * cos_pp * (C_wp * sh_theta - C_kz * ch_theta) * K1 * I0 \
               - 2 * np.pi * m_T * kappa * I1 * (-C_kz * sh_theta * cos_pp + C_wp * ch_theta * cos_pp + C_wr * sin_pp)
               
            F3 = 2 * np.pi * p_T_safe * (C_kp * ch_theta + C_wz * sh_theta) * kappa * I1 \
               - 4 * np.pi * m_T * (C_kp * sh_theta + C_wz * ch_theta) * K1 * I0
               
            # components G_mu
            G0 = 4 * np.pi * C_wr * K1 * (p_T_safe * I2 + T * csch_theta * I1)
            
            G1 = (4 * np.pi / p_T_safe) * I1 * (
                p_T_safe**2 * sin_pp * (C_kz * ch_theta - C_wp * sh_theta) * K0 
                - m_T * T * K1 * (sin_pp * (C_kz - C_wp * coth_theta) + C_wr * csch_theta * cos_pp)
            ) - 4 * np.pi * m_T * K1 * I2 * (
                C_kz * sh_theta * sin_pp - C_wp * ch_theta * sin_pp + C_wr * cos_pp
            )
            
            G2 = (4 * np.pi / p_T_safe) * I1 * (
                p_T_safe**2 * cos_pp * (C_wp * sh_theta - C_kz * ch_theta) * K0
                + m_T * T * K1 * (cos_pp * (C_kz - C_wp * coth_theta) - C_wr * csch_theta * sin_pp)
            ) - 4 * np.pi * m_T * K1 * I2 * (
                -C_kz * sh_theta * cos_pp + C_wp * ch_theta * cos_pp + C_wr * sin_pp
            )
            
            G3 = 4 * np.pi * sh_theta * (
                (C_kp * coth_theta + C_wz) * K1 * (p_T_safe * I2 + T * csch_theta * I1)
                - m_T * (C_kp + C_wz * coth_theta) * K0 * I1
            )
            
            # analytical (denominator Cooper-Frye)
            F_den = 4 * np.pi * K1 * I0
            G_den = 4 * np.pi * K0 * I1
            
            # integration over the variable zeta (in this case just a sum over the index i on the hypersurface)
            # Vol_element is a vector for the entire valid domain, we use numpy vectorization with sum()
            num0 = np.sum((m_T * F0 * dr_val - p_T_safe * G0 * dtau_val) * vol_element)
            num1 = np.sum((m_T * F1 * dr_val - p_T_safe * G1 * dtau_val) * vol_element)
            num2 = np.sum((m_T * F2 * dr_val - p_T_safe * G2 * dtau_val) * vol_element)
            num3 = np.sum((m_T * F3 * dr_val - p_T_safe * G3 * dtau_val) * vol_element)
            
            den = np.sum((m_T * F_den * dr_val - p_T_safe * G_den * dtau_val) * vol_element)
            
            # Pauli-Lubański vector in the LAB frame (covariant components)
            pi_LAB_cov = - np.array([num0, num1, num2, num3]) / (4.0 * m_particle * den + 1e-15)
            
            # transformation pi^i = -pi_i
            pi_LAB_contra = np.array([pi_LAB_cov[0], -pi_LAB_cov[1], -pi_LAB_cov[2], -pi_LAB_cov[3]])
            
            # PRF boost
            gamma = m_T / m_particle
            v_x, v_y = px / m_T, py / m_T
            
            # scalar product v_i * pi^i
            v_dot_pi = v_x * pi_LAB_contra[1] + v_y * pi_LAB_contra[2]
            
            boost_factor = gamma**2 / (gamma + 1.0)
            
            pi_PRF_x_map[i_y, i_x] = pi_LAB_contra[1] + boost_factor * v_x * v_dot_pi - gamma * v_x * pi_LAB_contra[0]
            pi_PRF_y_map[i_y, i_x] = pi_LAB_contra[2] + boost_factor * v_y * v_dot_pi - gamma * v_y * pi_LAB_contra[0]
            pi_PRF_z_map[i_y, i_x] = pi_LAB_contra[3] 
            
    return pi_PRF_x_map, pi_PRF_y_map, pi_PRF_z_map


# ==========================================
# 7. 4 cases of spin evolution + Pauli-Lubański integration
# ==========================================
print("--- Simulations of spin evolution (4 cases) ---")

cases = [
    {"label": r"$C_{kr} = 0.1$", "idx": 0},  # Ckx
    {"label": r"$C_{\omega r} = 0.1$", "idx": 1},  # Cwx
    {"label": r"$C_{\omega z} = 0.1$", "idx": 3},  # Cwz
    {"label": r"$C_{kz} = 0.1$", "idx": 4}   # Ckz
]

p_grid = np.linspace(-4.0, 4.0, 41)
all_polarization_maps = []

for case in cases:
    print(f"\nComputing the case: {case['label']}")
    
    y0_spin_case = np.zeros(6 * N)
    y0_spin_case[case['idx'] * N : (case['idx'] + 1) * N] = 0.1 * rho

    solution_spin = solve_ivp(
        spin_derivatives, 
        (tau0, tau_max), 
        y0_spin_case, 
        method='BDF', 
        t_eval=solution_bg.t,  
        rtol=1e-3, 
        atol=1e-3,
        jac_sparsity=jac_sparsity_spin
    )
    
    if len(r_iso) > 0:
        curve_points = np.column_stack((tau_iso, r_iso))

        interp_Ckx = RegularGridInterpolator((solution_bg.t, r), solution_spin.y[0:N, :].T)
        interp_Cwx = RegularGridInterpolator((solution_bg.t, r), solution_spin.y[N:2*N, :].T)
        interp_Cky = RegularGridInterpolator((solution_bg.t, r), solution_spin.y[2*N:3*N, :].T)
        interp_Cwz = RegularGridInterpolator((solution_bg.t, r), solution_spin.y[3*N:4*N, :].T)
        interp_Ckz = RegularGridInterpolator((solution_bg.t, r), solution_spin.y[4*N:5*N, :].T)
        interp_Cwy = RegularGridInterpolator((solution_bg.t, r), solution_spin.y[5*N:6*N, :].T)

        interp_T = RegularGridInterpolator((solution_bg.t, r), solution_bg.y[0:N, :].T / T0)
        interp_theta = RegularGridInterpolator((solution_bg.t, r), solution_bg.y[2*N:3*N, :].T)

        C_curves = (
            interp_Ckx(curve_points), interp_Cwx(curve_points), 
            interp_Cky(curve_points), interp_Cwz(curve_points), 
            interp_Ckz(curve_points), interp_Cwy(curve_points)
        )
        T_curve = interp_T(curve_points)
        theta_curve = interp_theta(curve_points)

        print("integrating Pauli-Lubański vector...")
        pi_x, pi_y, pi_z = compute_polarization_maps(p_grid, p_grid, C_curves, T_curve, theta_curve, r_iso, tau_iso)
        all_polarization_maps.append([pi_x, pi_y, pi_z])
    else:
        print("no isotherm.")
        all_polarization_maps.append([np.zeros((41,41)), np.zeros((41,41)), np.zeros((41,41))])


# ==========================================
# 8. Final Visualization of Polarization Maps
# ==========================================
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter 
from matplotlib.ticker import FuncFormatter, MaxNLocator

print("\n--- generating polarization maps ---")

fig, axs = plt.subplots(4, 3, figsize=(16, 20))
fig.subplots_adjust(wspace=0.0, hspace=0.0)

titles = [r'$\langle \pi_x^* \rangle$', r'$\langle \pi_y^* \rangle$', r'$\langle \pi_z^* \rangle$']
P_X, P_Y = np.meshgrid(p_grid, p_grid)
threshold = 1e-15

for row in range(4):
    maps_case = all_polarization_maps[row]
    maps_clean = [np.where(np.abs(m) < threshold, 0.0, m) for m in maps_case]
    
    vmax = max([np.max(np.abs(m)) for m in maps_clean])
    if vmax < 1e-5: 
        vmax = 0.01 
        
    levels_filled = np.linspace(-vmax, vmax, 21)
    levels_lines = np.linspace(-vmax, vmax, 11)

    for col in range(3):
        ax = axs[row, col]
        cf = ax.contourf(P_X, P_Y, maps_clean[col], levels=levels_filled, cmap='RdYlBu_r', extend='both')
        
        if np.max(np.abs(maps_clean[col])) > 0:
            ax.contour(P_X, P_Y, maps_clean[col], levels=levels_lines, colors='black', linewidths=0.5, alpha=0.5)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        
        # axis description
        ax.tick_params(axis='both', which='both', direction='in', 
                       top=True, right=True, bottom=True, left=True, labelsize=24)

        ax.set_xticks([-4, -2, 0, 2, 4])
        if row == 3:
            # x axis description
            ax.set_xlabel(r'$p_x$ [GeV]', fontsize=28)
            if col == 0:
                ax.set_xticklabels(['-4', '-2', '0', '2', '4'])
            else:
                ax.set_xticklabels(['', '-2', '0', '2', '4'])
        else:
            ax.set_xticklabels([])

        ax.set_yticks([-4, -2, 0, 2, 4])
        if col == 0:
            # y axis description 
            ax.set_ylabel(r'$p_y$ [GeV]', fontsize=28)
        else:
            ax.set_yticklabels([])

        if row == 0:
            # titles 
            ax.set_title(titles[col], fontsize=32, fontweight='bold', pad=20)

        if col == 0:
            # inside plot label for the case 
            ax.text(-3.6, 3.2, cases[row]['label'], fontsize=28, fontweight='bold')

    cbar = fig.colorbar(cf, ax=axs[row, :], location='right', fraction=0.02, pad=0.03, shrink=0.85, aspect=15)
    
    cbar.ax.tick_params(labelsize=24)
    
    ticks = [-vmax, -vmax/2, 0.0, vmax/2, vmax]
    cbar.set_ticks(ticks)
    
    def custom_formatter(x, pos, current_vmax=vmax):
        decimals = 3 if current_vmax < 0.05 else 2
        val_str = f"{x:.{decimals}f}"
        
        if val_str in ["-0.00", "-0.000"]:
            val_str = val_str.replace("-", "")
        return val_str

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

os.makedirs("output_plots", exist_ok=True)
filename = "output_plots/spin_polarization_mass_1000MeV-analytical.pdf"
plt.savefig(filename, dpi=300, bbox_inches='tight')
