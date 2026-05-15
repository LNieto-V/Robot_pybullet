import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt

class FuzzyController:
    def triangular(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x < c:
            return (c - x) / (c - b)
        else:
            return 0.0
    
    def trapezoidal(self, x, a, b, c, d):
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c)
        else:
            return 0.0
        
    def defusificar(self, entradas_difusas, valores_singleton):
        if len(entradas_difusas) != len(valores_singleton):
            raise ValueError("Las listas deben tener el mismo tamaño")

        numerador = sum(mu * v for mu, v in zip(entradas_difusas, valores_singleton))
        denominador = sum(entradas_difusas)

        if denominador == 0:
            return 0.0  
        return numerador / denominador


class DifferentialDriveSimulator:
    def __init__(self):
        # Configuración de simulación
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.Ts = 0.005         # tiempo de muestreo [s]
        p.setTimeStep(self.Ts)
        p.setRealTimeSimulation(0)

        # Parámetros del robot
        self.m = 3.0            
        self.J = 0.01           
        self.L = 0.3            
        self.r = 0.05           

        # Trayectoria circular
        self.R = 2.0            
        self.w_circ = 0.6       
        self.cx = 0.0           
        self.cy = 0.0           

        # Visualización
        self.trajectory_length = 1000
        self.real_traj = deque(maxlen=self.trajectory_length)
        self.desired_traj = deque(maxlen=self.trajectory_length)

        # Historial de datos para graficar
        self.time_log = []
        self.v_left_log = []
        self.v_right_log = []
        self.torque_left_log = []
        self.torque_right_log = []
        self.linear_velocity_log = []
        self.angular_velocity_log = []
        self.x_log = []
        self.y_log = []
        self.phi_log = [] 

        # Guardar trayectoria deseada
        self.x_d = []
        self.y_d = []
        self.phi_d_log = [] 

        # Instancia del controlador difuso
        self.fuzzy_controller = FuzzyController()
        self.setup_fuzzy_sets()

        # Para el cálculo de aceleraciones
        self.v_prev = 0.0
        self.omega_prev = 0.0

        self.setup_environment()
        self.load_robot()

    def setup_environment(self):
        """Configura el entorno de simulación"""
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)

    def create_robot_urdf(self):
        """Genera el archivo URDF del robot con tus parámetros"""
        urdf_content = """<?xml version="1.0"?>
<robot name="three_wheel_robot">
    <link name="chassis">
        <visual><geometry><box size="0.4 0.3 0.1"/></geometry><material name="blue"/></visual>
        <collision><geometry><box size="0.4 0.3 0.1"/></geometry></collision>
        <inertial><mass value="3.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    </link>
    <link name="left_wheel">
        <visual><origin rpy="1.5708 0 0" xyz="0 0 0"/><geometry><cylinder radius="0.05" length="0.04"/></geometry></visual>
        <collision><origin rpy="1.5708 0 0" xyz="0 0 0"/><geometry><cylinder radius="0.05" length="0.04"/></geometry></collision>
        <inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    </link>
    <link name="right_wheel">
        <visual><origin rpy="1.5708 0 0" xyz="0 0 0"/><geometry><cylinder radius="0.05" length="0.04"/></geometry></visual>
        <collision><origin rpy="1.5708 0 0" xyz="0 0 0"/><geometry><cylinder radius="0.05" length="0.04"/></geometry></collision>
        <inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    </link>
    <link name="caster_wheel">
        <visual><geometry><sphere radius="0.03"/></geometry></visual>
        <collision><geometry><sphere radius="0.03"/></geometry></collision>
        <inertial><mass value="0.2"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    </link>
    <joint name="left_wheel_joint" type="continuous">
        <parent link="chassis"/><child link="left_wheel"/>
        <axis xyz="0 1 0"/><origin xyz="-0.1 0.15 -0.05"/>
    </joint>
    <joint name="right_wheel_joint" type="continuous">
        <parent link="chassis"/><child link="right_wheel"/>
        <axis xyz="0 1 0"/><origin xyz="-0.1 -0.15 -0.05"/>
    </joint>
    <joint name="caster_joint" type="continuous">
        <parent link="chassis"/><child link="caster_wheel"/>
        <axis xyz="0 0 1"/><origin xyz="0.2 0 -0.05"/>
    </joint>
</robot>"""
        filename = "three_wheel_robot.urdf"
        with open(filename, "w") as f:
            f.write(urdf_content)
        return filename

    def load_robot(self):
        """Carga el robot en la simulación"""
        urdf_file = self.create_robot_urdf()

        # Posición inicial del robot
        start_x = self.cx + self.R
        start_y = self.cy 
        start_yaw = np.pi / 2 

        initial_position = [start_x-0.2, start_y-0.2, 0.05] 
        initial_orientation = p.getQuaternionFromEuler([0, 0, start_yaw])

        self.robot_id = p.loadURDF(urdf_file, initial_position, initial_orientation, useFixedBase=False)

        # Identificar articulaciones motorizadas
        self.left_wheel_joint = None
        self.right_wheel_joint = None
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode()
            if 'left' in joint_name:
                self.left_wheel_joint = i
            elif 'right' in joint_name:
                self.right_wheel_joint = i

        if None in [self.left_wheel_joint, self.right_wheel_joint]:
            raise Exception("No se encontraron las articulaciones motorizadas")

        # Configurar fricción
        p.changeDynamics(self.robot_id, self.left_wheel_joint, lateralFriction=1.5)
        p.changeDynamics(self.robot_id, self.right_wheel_joint, lateralFriction=1.5)

        # Limpiar archivo temporal
        if os.path.exists(urdf_file):
            os.remove(urdf_file)

        self.reset_camera()

    def reset_camera(self):
        """Posiciona la cámara para ver mejor el robot"""
        p.resetDebugVisualizerCamera(
            cameraDistance=6, 
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[self.cx, self.cy, 0] 
        )

    def calculate_desired_trajectory(self, steps):
        """Calcula una trayectoria circular deseada"""
        t = np.arange(0, steps * self.Ts, self.Ts)
        x_d = self.cx + self.R * np.cos(self.w_circ * t)
        y_d = self.cy + self.R * np.sin(self.w_circ * t)
        dx_d = -self.R * self.w_circ * np.sin(self.w_circ * t)
        dy_d = self.R * self.w_circ * np.cos(self.w_circ * t)
        
        # phi_d y omega_d con suavizado
        phi_d = np.unwrap(np.arctan2(dy_d, dx_d))
        omega_d_raw = np.diff(phi_d, prepend=phi_d[0]) / self.Ts
        omega_d = np.convolve(omega_d_raw, np.ones(25)/25, mode='same') 

        self.phi_d_log = phi_d 

        return x_d, y_d, phi_d, omega_d

    def setup_fuzzy_sets(self):
        """Define los conjuntos difusos de la versión original, ligeramente ajustados en velocidad."""
        
        self.rho_sets = {
            "MUY_CERCANO": lambda x: self.fuzzy_controller.trapezoidal(x, -0.01, 0, 0.03, 0.08), 
            "CERCANO": lambda x: self.fuzzy_controller.triangular(x, 0.03, 0.1, 0.2),      
            "MEDIO": lambda x: self.fuzzy_controller.triangular(x, 0.15, 0.3, 0.45),        
            "LEJOS": lambda x: self.fuzzy_controller.triangular(x, 0.4, 0.6, 0.8),        
            "MUY_LEJOS": lambda x: self.fuzzy_controller.trapezoidal(x, 0.7, 1.0, 1.5, 1.5) 
        }
        self.v_singleton_values = [0.5, 1.5, 2.5, 3.5, 4.5] 

        self.alpha_sets = {
            "NG": lambda x: self.fuzzy_controller.trapezoidal(x, -np.pi, -np.pi, -np.pi/2, -np.pi/4),
            "NP": lambda x: self.fuzzy_controller.triangular(x, -np.pi/2, -np.pi/4, 0),
            "ZE": lambda x: self.fuzzy_controller.triangular(x, -np.pi/8, 0, np.pi/8),
            "PP": lambda x: self.fuzzy_controller.triangular(x, 0, np.pi/4, np.pi/2),
            "PG": lambda x: self.fuzzy_controller.trapezoidal(x, np.pi/4, np.pi/2, np.pi, np.pi)
        }

        self.beta_sets = {
            "NG": lambda x: self.fuzzy_controller.trapezoidal(x, -np.pi, -np.pi, -np.pi/2, -np.pi/4),
            "NP": lambda x: self.fuzzy_controller.triangular(x, -np.pi/2, -np.pi/4, 0),
            "ZE": lambda x: self.fuzzy_controller.triangular(x, -np.pi/8, 0, np.pi/8),
            "PP": lambda x: self.fuzzy_controller.triangular(x, 0, np.pi/4, np.pi/2),
            "PG": lambda x: self.fuzzy_controller.trapezoidal(x, np.pi/4, np.pi/2, np.pi, np.pi)
        }

        # Matriz de reglas para omega_ref
        self.omega_rules_matrix = [
            ["RG", "RG", "RG", "RG", "RG"], 
            ["RP", "RP", "RP", "RP", "RP"], 
            ["RP", "RP", "Z",  "LP", "LP"], 
            ["LP", "LP", "LP", "LP", "LP"], 
            ["LG", "LG", "LG", "LG", "LG"]  
        ]

        self.omega_singleton_values = {
            "RG": -30.0, 
            "RP": -10.0,  
            "Z":   0.0,  
            "LP":  10.0,  
            "LG": 30.0   
        }

    def control_step(self, k, x_d, y_d, phi_d_target):
        """Ejecuta un paso de control difuso."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y, z = pos
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        self.x_log.append(x)
        self.y_log.append(y)
        self.phi_log.append(yaw)

        # Look-ahead restaurado a la versión original funcional
        look_ahead_steps = 20 
        target_k = min(k + look_ahead_steps, len(x_d) - 1)

        dx = x_d[target_k] - x
        dy = y_d[target_k] - y

        rho = math.sqrt(dx**2 + dy**2)
        
        theta_desired = math.atan2(dy, dx)
        alpha = theta_desired - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha)) 
        
        beta = phi_d_target[target_k] - yaw - alpha 
        beta = math.atan2(math.sin(beta), math.cos(beta)) 

        mf_rho = {name: func(rho) for name, func in self.rho_sets.items()}
        rho_memberships = list(mf_rho.values())
        v_ref = self.fuzzy_controller.defusificar(rho_memberships, self.v_singleton_values)

        alpha_fuzzified = [func(alpha) for func in self.alpha_sets.values()]
        beta_fuzzified = [func(beta) for func in self.beta_sets.values()]

        activaciones = []
        valores_omega = []

        alpha_keys = list(self.alpha_sets.keys()) 
        beta_keys = list(self.beta_sets.keys())

        for i in range(len(alpha_keys)): 
            for j in range(len(beta_keys)): 
                mu_rule = min(alpha_fuzzified[i], beta_fuzzified[j])
                output_label = self.omega_rules_matrix[i][j]
                output_value = self.omega_singleton_values[output_label]
                activaciones.append(mu_rule)
                valores_omega.append(output_value)
        
        omega_ref = self.fuzzy_controller.defusificar(activaciones, valores_omega)

        self.linear_velocity_log.append(v_ref)
        self.angular_velocity_log.append(omega_ref)

        v_left = (v_ref - omega_ref * self.L / 2) / self.r
        v_right = (v_ref + omega_ref * self.L / 2) / self.r

        self.v_left_log.append(v_left)
        self.v_right_log.append(v_right)
        self.time_log.append(k * self.Ts)

        dv = (v_ref - self.v_prev) / self.Ts
        domega = (omega_ref - self.omega_prev) / self.Ts
        self.v_prev = v_ref
        self.omega_prev = omega_ref

        tau_l = (self.m * self.r / 2) * dv - (self.J * self.r / self.L) * domega
        tau_r = (self.m * self.r / 2) * dv + (self.J * self.r / self.L) * domega

        self.torque_left_log.append(tau_l)
        self.torque_right_log.append(tau_r)

        p.setJointMotorControl2(
            self.robot_id, self.left_wheel_joint,
            p.VELOCITY_CONTROL,
            targetVelocity=v_left,
            force=350
        )
        p.setJointMotorControl2(
            self.robot_id, self.right_wheel_joint,
            p.VELOCITY_CONTROL,
            targetVelocity=v_right,
            force=350 
        )

        return x, y, yaw

    def update_visualization(self, x, y, x_d, y_d, k):
        """Actualiza visualización de trayectorias (Optimizada para PyBullet)"""
        if k % 50 == 0:
            self.real_traj.append([x, y, 0.01])
            if len(self.real_traj) > 1:
                try:
                    p.addUserDebugLine(
                        self.real_traj[-2],
                        self.real_traj[-1],
                        lineColorRGB=[1, 0, 0], 
                        lineWidth=4,
                        lifeTime=0
                    )
                except: pass

        if k < len(x_d) and k < len(y_d) and k % 50 == 0:
            self.desired_traj.append([x_d[k], y_d[k], 0.01])
            if len(self.desired_traj) > 1:
                try:
                    p.addUserDebugLine(
                        self.desired_traj[-2],
                        self.desired_traj[-1],
                        lineColorRGB=[0, 0, 1], 
                        lineWidth=3,
                        lifeTime=0
                    )
                except: pass

    def run_simulation(self, steps=30000):
        """Ejecuta la simulación completa"""
        x_d, y_d, phi_d, omega_d_plot = self.calculate_desired_trajectory(steps)
        self.x_d = x_d
        self.y_d = y_d

        try:
            for k in range(steps):
                x, y, yaw = self.control_step(k, x_d, y_d, phi_d)
                self.update_visualization(x, y, x_d, y_d, k)
                p.stepSimulation()
        except KeyboardInterrupt:
            print("Simulación detenida manualmente.")
        finally:
            p.disconnect()
            self.plot_results(omega_d_plot) 

    def plot_results(self, omega_d_plot):
        plt.figure(figsize=(20, 10)) 
        plt.subplot(2, 2, 1) 
        plt.plot(self.x_d, self.y_d, 'r--', label='Trayectoria Deseada')
        plt.plot(self.x_log, self.y_log, 'g-', label='Trayectoria Real')
        plt.xlabel('Posición X [m]')
        plt.ylabel('Posición Y [m]')
        plt.title('Trayectoria del Robot')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.subplot(2, 2, 2) 
        min_len_v = min(len(self.time_log), len(self.linear_velocity_log), len(self.x_d) - 1)
        v_d_calc = np.sqrt(np.diff(self.x_d[:len(self.time_log)])**2 + np.diff(self.y_d[:len(self.time_log)])**2) / self.Ts
        min_len_v_plot = min(len(self.time_log), len(self.linear_velocity_log), len(v_d_calc))
        plt.plot(self.time_log[:min_len_v_plot], v_d_calc[:min_len_v_plot], 'r--', label='Velocidad Lineal Deseada')
        plt.plot(self.time_log[:min_len_v_plot], self.linear_velocity_log[:min_len_v_plot], 'g-', label='Velocidad Lineal Real')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Velocidad Lineal [m/s]')
        plt.title('Velocidad Lineal del Robot')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 3) 
        min_len_omega = min(len(self.time_log), len(self.angular_velocity_log), len(omega_d_plot))
        plt.plot(self.time_log[:min_len_omega], omega_d_plot[:min_len_omega], 'r--', label='Velocidad Angular Deseada')
        plt.plot(self.time_log[:min_len_omega], self.angular_velocity_log[:min_len_omega], 'g-', label='Velocidad Angular Real')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Velocidad Angular [rad/s]')
        plt.title('Velocidad Angular del Robot')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 4)
        min_len_torque = min(len(self.time_log), len(self.torque_left_log), len(self.torque_right_log))
        plt.plot(self.time_log[:min_len_torque], self.torque_left_log[:min_len_torque], 'b-', label='Torque Rueda Izquierda')
        plt.plot(self.time_log[:min_len_torque], self.torque_right_log[:min_len_torque], 'm-', label='Torque Rueda Derecha')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Torque [Nm]')
        plt.title('Torques Aplicados a las Ruedas')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    simulator = DifferentialDriveSimulator()
    simulator.run_simulation(steps=25000)
