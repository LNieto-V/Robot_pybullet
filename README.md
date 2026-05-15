# Grey Wolf Optimizer-Tuned Fuzzy Kanayama Controller para DDMR en PyBullet

Este repositorio contiene la implementación y simulación en **PyBullet** de un Controlador Difuso tipo Sugeno para un Robot Móvil de Accionamiento Diferencial (DDMR). El controlador se basa en el modelo de error cinemático de **Kanayama** y sus parámetros de inferencia difusa fueron sintonizados utilizando **Grey Wolf Optimizer (GWO)**, respaldado en la literatura IEEE.

## Características Principales

- **Control Difuso Sugeno (Kanayama):** Dos sistemas de inferencia difusa (FS1 para velocidad lineal, FS2 para velocidad angular) altamente optimizados.
- **Simulación Física (PyBullet):** Renderizado de trayectorias en tiempo real, gestión de colisiones, cálculo de torques y fricción lateral ajustada.
- **Múltiples Trayectorias de Prueba:**
  - **Circular:** `Circular_Robot_Refactored.py`
  - **Infinito (Lemniscata):** `Infinito_Robot_Refactored.py`
  - **Pétalo (Rosa Polar de 3 Pétalos):** `Petalo_Robot_Refactored.py`
- **Benchmarking de Latencia:** Un script dedicado (`latencia.py`) para perfilar el costo computacional del controlador difuso puro, garantizando tiempos de respuesta estrictamente inferiores a 5 ms por ciclo (promediando ~0.003 ms) sin el uso de dependencias pesadas.

## Requisitos y Configuración

El proyecto utiliza [`uv`](https://github.com/astral-sh/uv) como gestor rápido de paquetes y dependencias de Python.

1. Asegúrate de tener instalado Python 3.11+.
2. Instala las dependencias mediante `uv`:
   ```bash
   uv sync
   ```
   *(Dependencias principales: `pybullet`, `numpy`, `matplotlib`)*

## Ejecución

Para iniciar cualquiera de las simulaciones en PyBullet, utiliza `uv run`:

```bash
# Simular trayectoria circular
uv run python Circular_Robot_Refactored.py

# Simular trayectoria en forma de infinito
uv run python Infinito_Robot_Refactored.py

# Simular trayectoria en forma de pétalo
uv run python Petalo_Robot_Refactored.py
```

Al terminar la simulación (o al cerrar la ventana de PyBullet), se generará automáticamente una gráfica (Matplotlib) comparando las trayectorias real y deseada, y los perfiles de velocidad/torque.

### Análisis de Latencia

Para validar que la carga computacional de la lógica de control difuso cumple con requisitos de tiempo real, ejecuta:

```bash
uv run python latencia.py
```

Esto generará un reporte de estadísticas en consola (y guardará los datos en formato `.txt` y `.csv` excluidos del control de versiones).

## Arquitectura del Código

El sistema está diseñado de forma modular pero autocontenida para facilitar su portabilidad. Dentro de cada script de simulación encontrarás:
- `DifferentialDriveSimulator`: Maneja el servidor físico, la inserción del URDF, motores y debug visual.
- `FuzzyController`: Implementa las funciones de membresía manuales, la matriz FAM optimizada y la defuzzificación por promedio ponderado.
- El ciclo de simulación y cálculo de error Kanayama acoplado a la cinemática inversa (transformando $(v, \omega)$ en velocidades de rueda derecha e izquierda).

## Notas Adicionales
- Para preservar recursos y evitar fugas de memoria en PyBullet, el dibujado de las líneas de trayectoria ("diezmo") está optimizado a 1 actualización de línea cada 50 iteraciones.
- Los reportes CSV y gráficas generadas (como EPS o PDF de resultados) son ignorados por el sistema de control de versiones para mantener la pureza del repositorio.
