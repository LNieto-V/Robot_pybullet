import numpy as np
import matplotlib.pyplot as plt
from controlador_difuso import controlador_difuso

# === PARÁMETROS FÍSICOS ===
m = 0.75
J = 0.005
L = 0.25
r = 0.1
Ts = 0.001
N = 34000
t = np.arange(N) * Ts

# === Cargar parámetros del controlador desde archivo ===
with open("mejores_parametros.txt", "r") as f:
    valores = [float(line.strip()) for line in f.readlines()]
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = valores

# === Instancias del controlador difuso ===
ctr_Vel = controlador_difuso()
ctr = controlador_difuso()

# === NEW DESIRED TRAJECTORY: 3-PETAL FLOWER ===
a = 4
n_petals = 3
w = 0.1
theta = w * t

x_d = a * np.cos(n_petals * theta) * np.cos(theta)
y_d = a * np.cos(n_petals * theta) * np.sin(theta)

# Derivatives
dx_d = -a * n_petals * np.sin(n_petals * theta) * np.cos(theta) - \
       a * np.cos(n_petals * theta) * np.sin(theta)
dy_d = -a * n_petals * np.sin(n_petals * theta) * np.sin(theta) + \
        a * np.cos(n_petals * theta) * np.cos(theta)
v_d = np.sqrt(dx_d**2 + dy_d**2)
v_d = v_d * (2.0 / np.max(v_d))
phi_d = np.unwrap(np.arctan2(dy_d, dx_d))
omega_d = np.diff(phi_d, prepend=phi_d[0]) / Ts
omega_d = np.convolve(omega_d, np.ones(25)/25, mode='same')
#omega_d = np.clip(omega_d, -8.0, 8.0)  # Limitar a ±8 rad/s



# === INICIALIZACIÓN DE ESTADOS ===
x = np.zeros(N)
y = np.zeros(N)
phi = np.zeros(N)
v = np.zeros(N)
omega = np.zeros(N)
v_ref_array = np.zeros(N)
domega_array = np.zeros(N)
tau_r_array = np.zeros(N)
tau_l_array = np.zeros(N)
x[0] = 3.95
y[0] = 0.0
phi[0] = np.pi/4

# === CONTROL SUAVIZADO ===
max_dv = 25.0
alpha_smooth = 0.1
max_domega = 3.0
# === SIMULACIÓN ===
for k in range(N - 1):
    omega[0] = omega_d[0]  # Igualar condición inicial
    dx = x_d[k] - x[k]
    dy = y_d[k] - y[k]
    rho = np.sqrt(dx**2 + dy**2)
    alpha = np.arctan2(dy, dx) - phi[k]
    beta = phi_d[k] - phi[k] - alpha

    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    beta = np.arctan2(np.sin(beta), np.cos(beta))

    kp = rho * x1

    # Funciones de pertenencia para rho
    fuzz_rho = [
        ctr_Vel.trapezoidal(kp, 0.00, 0.00, 0.0, 0.1),
        ctr_Vel.triangular(kp, 0.0, 0.10, 0.25),
        ctr_Vel.triangular(kp, 0.1, 0.25, 0.5),
        ctr_Vel.triangular(kp, 0.25, 0.5, 1.0),
        ctr_Vel.trapezoidal(kp, 0.50, 1.0, 1.00, 1.00)
    ]
    rho_singleton = [0, x2, x3, x4, x5]
    v_ref_raw = ctr_Vel.defusificar(fuzz_rho, rho_singleton)
    v_ref = (1 - alpha_smooth) * v[k] + alpha_smooth * v_ref_raw
    
    v_ref_array[k] = 1.65 *v_ref

    # Funciones de pertenencia para alpha
    alpha_labels = [
        ctr.trapezoidal(alpha, -np.pi/2, -np.pi/2, -np.pi/3, -np.pi/4),
        ctr.triangular(alpha, -np.pi/3, -np.pi/4, 0),
        ctr.triangular(alpha, -np.pi/8, 0, np.pi/8),
        ctr.triangular(alpha, 0, np.pi/4, np.pi/3),
        ctr.trapezoidal(alpha, np.pi/4, np.pi/3, np.pi/2, np.pi/2)
    ]
    Alpha_singleton = [-8, x6, 0, x7, 8]
    w_alpha = ctr_Vel.defusificar(alpha_labels, Alpha_singleton)

    # Funciones de pertenencia para beta
    beta_labels = [
        ctr.trapezoidal(beta, -np.pi, -np.pi, -3*np.pi/4, -np.pi/2),
        ctr.triangular(beta, -3*np.pi/4, -np.pi/2, 0),
        ctr.triangular(beta, -np.pi/6, 0, np.pi/6),
        ctr.triangular(beta, 0, np.pi/2, 3*np.pi/4),
        ctr.trapezoidal(beta, np.pi/2, 3*np.pi/4, np.pi, np.pi)
    ]
    beta_singletons = [-4.0, x8, 0.0, x9, 4.0]
    w_beta = ctr_Vel.defusificar(beta_labels, beta_singletons)

    omega_ref = x10 * w_alpha + x11 * w_beta
    omega_ref = np.clip(omega_ref, -8.0, 8.0)

    #dv = np.clip((v_ref - v[k]) / Ts, -max_dv, max_dv)
    dv =(v_ref - v[k]) / Ts
    domega = (omega_ref - omega[k]) / Ts
    domega = np.clip((omega_ref - omega[k]) / Ts, -max_domega, max_domega)
    domega_array[k] = domega

    tau_r = (m*r/2)*dv + (J*r/L)*domega
    tau_l = (m*r/2)*dv - (J*r/L)*domega
    tau_r_array[k] = tau_r
    tau_l_array[k] = tau_l
    x[k+1] = x[k] + Ts * v[k] * np.cos(phi[k])
    y[k+1] = y[k] + Ts * v[k] * np.sin(phi[k])
    phi[k+1] = phi[k] + Ts * omega[k]
    v[k+1] = v[k] + Ts * (1/(m*r)) * (tau_r + tau_l)
    omega[k+1] = omega[k] + Ts * (L/(2*J*r)) * (tau_r - tau_l)

# === GRÁFICAS ===

# Trayectoria
plt.figure()
plt.plot(x, y, 'b', label='Robot displacement')
plt.plot(x_d, y_d, 'r--', label='Desired trajectory')
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.grid(True)
plt.savefig("trayectoria_robot_petalo.eps", format="eps")

# Velocidad lineal
plt.figure()
#plt.plot(t, v_d, label='v [m/s]')
plt.plot(t[:-1], v_ref_array[:-1], '--', label='v_ref [m/s]')
plt.plot(t, v_d, label='v_ref [m/s]')

plt.xlabel('Time [s]')
plt.ylabel('Linear velocity [m/s]')
plt.grid(True)
plt.legend()
plt.savefig("velocidad_lineal_petalo.eps", format="eps")

# Velocidad angular
plt.figure()
plt.plot(t, omega, label='omega [rad/s]')
plt.plot(t, omega_d, '--', label='omega_d [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [rad/s]')
plt.grid(True)
plt.legend()
plt.savefig("velocidad_angular_petalo.eps", format="eps")

# === TORQUES APPLIED TO THE WHEELS ===
plt.figure()
plt.plot(t, tau_r_array, label='Right torque τ_r [Nm]')
plt.plot(t, tau_l_array, '--', label='Left torque τ_l [Nm]')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
#plt.title('Torques applied to the wheels')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("robot_torques_petalo.eps", format="eps")
plt.show()


