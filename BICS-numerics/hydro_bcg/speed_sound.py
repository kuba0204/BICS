import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kve
import os

# ==========================================
# 1. Function to calculate c_s^2
# ==========================================
def calc_cs2(T_grid, mu_grid, m):
    T = np.clip(T_grid, 1e-3, None)
    z = m / T
    xi = mu_grid / T
    
    K1_K2_ratio = kve(1, z) / kve(2, z)
    A = 4.0 + z * K1_K2_ratio
    
    cs2 = np.zeros_like(T)
    
    mask_zero = (xi < 1e-6)
    mask_pos = ~mask_zero
    
    A_z = A[mask_zero]
    z_z = z[mask_zero]
    cs2[mask_zero] = A_z / (3.0 * A_z + z_z**2)
    
    A_p = A[mask_pos]
    z_p = z[mask_pos]
    xi_p = xi[mask_pos]
    
    coth2 = 1.0 / (np.tanh(xi_p)**2)
    numerator = A_p**2 * coth2 - 2*A_p**2 + 5*A_p + z_p**2
    denominator = A_p * ((3*A_p + z_p**2)*coth2 - (A_p - 1)**2)
    
    cs2[mask_pos] = numerator / denominator
    
    return cs2

# ==========================================
# 2. Function to prepare the grid and target contour levels
# ==========================================
T_vals = np.linspace(30, 450, 400)
mu_vals = np.linspace(0, 300, 400)

T_grid, mu_grid = np.meshgrid(T_vals, mu_vals)

cs2_1000 = calc_cs2(T_grid, mu_grid, m=1000.0)
cs2_300  = calc_cs2(T_grid, mu_grid, m=300.0)

# values of c_s^2 at the target point (T=300 MeV, mu=30 MeV) for both masses
target_T = 300.0
target_mu = 30.0
val_1000 = calc_cs2(np.array([target_T]), np.array([target_mu]), 1000.0)[0]
val_300  = calc_cs2(np.array([target_T]), np.array([target_mu]), 300.0)[0]

def get_clean_levels(target_val, vmin=0.0, vmax=0.33, num=10):
    base_levels = list(np.linspace(vmin, vmax, num))
    final_levels = [lvl for lvl in base_levels if abs(lvl - target_val) > 0.02]
    final_levels.append(target_val)
    return np.sort(final_levels)

levels_line_1000 = get_clean_levels(val_1000)
levels_line_300  = get_clean_levels(val_300)

# ==========================================
# 3. Function to extract exact contour coordinates
# ==========================================
def get_exact_contour_positions(clines, base_mu=25.0, stagger=25.0):

    pos = []
    paths = clines.get_paths()
    
    valid_idx = 0 
    for path in paths:
        v = path.vertices
        if len(v) == 0:
            continue 
            
        target_mu = base_mu if valid_idx % 2 == 0 else base_mu + stagger
        
        # looking for the point on the contour line closest to the target_mu
        idx = np.argmin(np.abs(v[:, 1] - target_mu))
        pos.append((v[idx, 0], v[idx, 1]))
        
        valid_idx += 1  
        
    return pos

# ==========================================
# 4. plots
# ==========================================
os.makedirs("output_plots", exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 14), sharex=True, gridspec_kw={'hspace': 0})

vmin, vmax = 0.0, 0.335 
levels_fill = np.linspace(vmin, vmax, 40)


levels_line_a = [0.04, 0.08, 0.11, 0.14, 0.18, 0.21]
levels_line_b = [0.11, 0.15, 0.18, 0.23, 0.26, 0.28, 0.30, 0.31]

datasets = [
    {"data": cs2_1000, "title": "(a)", "ax": axes[0], "levels_line": levels_line_a},
    {"data": cs2_300,  "title": "(b)", "ax": axes[1], "levels_line": levels_line_b}
]

for i, item in enumerate(datasets):
    ax = item["ax"]
    data = item["data"]
    l_line = item["levels_line"]
    
    cf = ax.contourf(T_grid / 1000, mu_grid / 1000, data, levels=levels_fill, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, extend='max')
    
    clines = ax.contour(T_grid / 1000, mu_grid / 1000, data, levels=l_line, colors='black', linewidths=0.6, alpha=1.0)
    
    positions = get_exact_contour_positions(clines, base_mu=0.045, stagger=0.030)
    
    fmt_dict = {lvl: f"{lvl:.2f}" for lvl in clines.levels}
    
    ax.clabel(clines, inline=True, manual=positions, fontsize=18, fmt=fmt_dict, colors='black')
    
    ax.set_xlim(T_vals[0] / 1000, T_vals[-1] / 1000)
    ax.set_ylim(mu_vals[0] / 1000, mu_vals[-1] / 1000)
    
    ax.set_ylabel(r'$\mu$ [GeV]', fontsize=28, color='black')
    
    if i == 1:
        ax.set_xlabel(r'$T$ [GeV]', fontsize=28)
        
    ax.tick_params(axis='both', direction='in', top=True, right=True, labelsize=24, color='black', labelcolor='black')
    
    ax.text(0.95, 0.95, item["title"], transform=ax.transAxes, 
            fontsize=32, fontweight='normal', color='black', va='top', ha='right')

cbar = fig.colorbar(cf, ax=axes, fraction=0.04, pad=0.03)
cbar.set_label(r'$c_s^2$', fontsize=28, rotation=0, labelpad=30, color='black')
cbar.ax.tick_params(labelsize=24, color='black', labelcolor='black')
cbar.set_ticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.333])
cbar.set_ticklabels(['0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '1/3'])

file_path_pdf = 'output_plots/Sound_speed_contours.pdf'
fig.savefig(file_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
