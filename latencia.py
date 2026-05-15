import numpy as np
import csv
import time
import os
import math

# =======================================================================
# FUNCIONES DE MEMBRESÍA MANUALES
# =======================================================================
def triangular(x, a, b, c):
    if x <= a or x >= c: 
        return 0.0
    elif a < x < b: 
        return (x - a) / (b - a)
    else: 
        return (c - x) / (c - b)

def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d: 
        return 0.0
    elif a < x < b: 
        return (x - a) / (b - a)
    elif b <= x <= c: 
        return 1.0
    else: 
        return (d - x) / (d - c)

def defusificar(memberships, singletons):
    num = sum(mu * s for mu, s in zip(memberships, singletons))
    den = sum(memberships)
    return num / den if den > 1e-9 else 0.0

# =======================================================================
# GENERACIÓN DE TRAYECTORIAS (1000 waypoints)
# =======================================================================
N = 1000
t = np.linspace(0, 2 * np.pi, N)

# 1. Circular
xd_c = 5 * np.cos(t)
yd_c = 5 * np.sin(t)
phid_c = np.arctan2(np.gradient(yd_c, t), np.gradient(xd_c, t))

# 2. Infinito (Lemniscata)
xd_i = 5 * np.cos(t) / (1 + np.sin(t)**2)
yd_i = 5 * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
phid_i = np.arctan2(np.gradient(yd_i, t), np.gradient(xd_i, t))

# 3. Trébol (Rosa de 3 pétalos)
r = 4.2 * np.cos(3 * t)
xd_t = r * np.cos(t)
yd_t = r * np.sin(t)
phid_t = np.arctan2(np.gradient(yd_t, t), np.gradient(xd_t, t))

# Parámetros optimizados GWO (Tabla VI, IEEE)
# z1 a z10
trajectories = {
    "Circular": {
        "x": xd_c, "y": yd_c, "phi": phid_c,
        "z": [0.4031, 0.1202, 0.3888, 0.6100, 1.0699, 1.3146, -1.4374, -0.1158, 0.4153, 2.3124]
    },
    "Infinito": {
        "x": xd_i, "y": yd_i, "phi": phid_i,
        "z": [0.3700, 0.2375, 1.2525, 1.5686, 1.5989, 3.2296, -3.1995, -1.2209, 1.5619, 2.5464]
    },
    "Trebol": {
        "x": xd_t, "y": yd_t, "phi": phid_t,
        "z": [0.8321, 0.1046, 1.6999, 1.7194, 1.8500, 2.5607, -3.8419, -1.7393, 0.8489, 1.8500]
    }
}

# =======================================================================
# PROCEDIMIENTO DE MEDICIÓN DE LATENCIA
# =======================================================================
dt = 0.01
trials = 5
all_results = []

print("Iniciando medición de latencia computacional...\n")

for name, data in trajectories.items():
    z = data["z"]
    
    # Extraer variables z (notación IEEE)
    # Internamente soporta la "Notación dual" mapeando z1->z10 correctamente
    z1 = z[0] # x1
    z2 = z[1] # x2
    z3 = z[2] # x3
    z4 = z[3] # x4
    z5 = z[4] # x5
    z6 = z[5] # x12
    RG = z[6] # z7
    RP = z[7] # z8
    ZE = 0.0  # Fijo
    LP = z[8] # z9
    LG = z[9] # z10
    
    # Pre-cargar singletons para optimizar
    s_v = [z2, z3, z4, z5, z6]
    r2_s = [RP, RP, ZE, LP, LP] # Fila 2 de la FAM
    
    times_ms = []
    
    for trial in range(trials):
        x, y, phi = 0.0, 0.0, 0.0
        
        for k in range(N):
            xd, yd, phid = data["x"][k], data["y"][k], data["phi"][k]
            
            # --- INICIO DE MEDICIÓN ---
            t0 = time.perf_counter()
            
            # 1. Cinemática de Error Kanayama
            dx = xd - x
            dy = yd - y
            rho = math.sqrt(dx**2 + dy**2)
            
            alpha = math.atan2(dy, dx) - phi
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            
            beta = phid - phi - alpha
            beta = math.atan2(math.sin(beta), math.cos(beta))
            
            # 2. FS1 (Velocidad Lineal)
            # Clip manual optimizado
            rho_z = z1 * rho
            rho_n = 1.0 if rho_z > 1.0 else (0.0 if rho_z < 0.0 else rho_z)
            
            mu_rho = [
                trapezoidal(rho_n, 0, 0, 0.05, 0.15),
                triangular(rho_n, 0.05, 0.20, 0.35),
                triangular(rho_n, 0.25, 0.45, 0.65),
                triangular(rho_n, 0.55, 0.75, 0.95),
                trapezoidal(rho_n, 0.85, 0.95, 1.0, 1.0)
            ]
            
            # Defuzzificación manual inline (sin bucles Python) para < 5ms
            num_v = mu_rho[0]*s_v[0] + mu_rho[1]*s_v[1] + mu_rho[2]*s_v[2] + mu_rho[3]*s_v[3] + mu_rho[4]*s_v[4]
            den_v = mu_rho[0] + mu_rho[1] + mu_rho[2] + mu_rho[3] + mu_rho[4]
            v_ref = (num_v / den_v) if den_v > 1e-9 else 0.0

            # 3. FS2 (Velocidad Angular)
            mu_a = [
                trapezoidal(alpha, -math.pi, -math.pi, -2, -1),
                triangular(alpha, -2, -1, 0),
                triangular(alpha, -1, 0, 1),
                triangular(alpha, 0, 1, 2),
                trapezoidal(alpha, 1, 2, math.pi, math.pi)
            ]
            mu_b = [
                trapezoidal(beta, -math.pi, -math.pi, -2, -1),
                triangular(beta, -2, -1, 0),
                triangular(beta, -1, 0, 1),
                triangular(beta, 0, 1, 2),
                trapezoidal(beta, 1, 2, math.pi, math.pi)
            ]
            
            num_o = 0.0
            den_o = 0.0
            
            # Recorrido de matriz FAM optimizado (salto si mu_a es 0)
            if mu_a[0] > 0: # Fila 0 (NG) -> Todo RG
                for j in range(5):
                    if mu_b[j] > 0:
                        mu_rule = mu_a[0] if mu_a[0] < mu_b[j] else mu_b[j]
                        num_o += mu_rule * RG
                        den_o += mu_rule
                        
            if mu_a[1] > 0: # Fila 1 (NP) -> Todo RP
                for j in range(5):
                    if mu_b[j] > 0:
                        mu_rule = mu_a[1] if mu_a[1] < mu_b[j] else mu_b[j]
                        num_o += mu_rule * RP
                        den_o += mu_rule
                        
            if mu_a[2] > 0: # Fila 2 (ZE) -> RP, RP, ZE, LP, LP
                for j in range(5):
                    if mu_b[j] > 0:
                        mu_rule = mu_a[2] if mu_a[2] < mu_b[j] else mu_b[j]
                        num_o += mu_rule * r2_s[j]
                        den_o += mu_rule
                        
            if mu_a[3] > 0: # Fila 3 (PP) -> Todo LP
                for j in range(5):
                    if mu_b[j] > 0:
                        mu_rule = mu_a[3] if mu_a[3] < mu_b[j] else mu_b[j]
                        num_o += mu_rule * LP
                        den_o += mu_rule
                        
            if mu_a[4] > 0: # Fila 4 (PG) -> Todo LG
                for j in range(5):
                    if mu_b[j] > 0:
                        mu_rule = mu_a[4] if mu_a[4] < mu_b[j] else mu_b[j]
                        num_o += mu_rule * LG
                        den_o += mu_rule
            
            omega_ref = (num_o / den_o) if den_o > 1e-9 else 0.0
            
            # --- FIN DE MEDICIÓN ---
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
            
            # Integración Euler (No medida)
            x += v_ref * math.cos(phi) * dt
            y += v_ref * math.sin(phi) * dt
            phi += omega_ref * dt
            
    # Guardar tiempos individuales por trayectoria (5000 puntos)
    filename_txt = f"control_times_{name.lower()}.txt"
    np.savetxt(filename_txt, times_ms, fmt='%.6f')
    
    # Calcular estadísticas
    arr_times = np.array(times_ms)
    all_results.append({
        "Trayectoria": name,
        "Media (ms)": arr_times.mean(),
        "Mediana (ms)": np.median(arr_times),
        "Std (ms)": arr_times.std(),
        "Minimo (ms)": arr_times.min(),
        "Maximo (ms)": arr_times.max()
    })

# Exportar el CSV usando el módulo csv estándar
csv_file = "latency_summary.csv"
keys = all_results[0].keys()
with open(csv_file, 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_results)

# Mostrar la tabla final en Markdown
print("| Trayectoria | Media (ms) | Mediana (ms) | Std (ms) | Minimo (ms) | Maximo (ms) |")
print("|---|---|---|---|---|---|")
for r in all_results:
    print(f"| {r['Trayectoria']} | {r['Media (ms)']:.4f} | {r['Mediana (ms)']:.4f} | {r['Std (ms)']:.4f} | {r['Minimo (ms)']:.4f} | {r['Maximo (ms)']:.4f} |")
