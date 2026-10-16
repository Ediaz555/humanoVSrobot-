# Robot que imita tu pose (MediaPipe)

## 1. Descripción

Programa en Python que usa la cámara web y **MediaPipe Pose** para detectar la posición de tus brazos en tiempo real. Según la pose detectada, muestra en una segunda ventana la imagen de un robot que hace el mismo movimiento.

Reconoce **11 poses**: 9 básicas y 2 extras (músculo y diagonal).

| Pose | Archivo de imagen esperado |
|---|---|
| Ambos brazos abajo | `ambos_abajo` |
| Ambos brazos arriba | `ambos_arriba` |
| Ambos brazos extendidos (horizontal) | `ambos_extendida` |
| Derecha arriba, izquierda abajo | `derecha_arriba_izq_abajo` |
| Derecha arriba, izquierda extendida | `derecha_arriba_izq_extendida` |
| Derecha extendida, izquierda abajo | `derecha_extendida_izq_abajo` |
| Izquierda arriba, derecha abajo | `izquierda_arriba_der_abajo` |
| Izquierda arriba, derecha extendida | `izquierda_arriba_der_extendida` |
| Izquierda extendida, derecha abajo | `izquierda_extendida_der_abajo` |
| **Ambos brazos haciendo músculo 💪** | `ambos_musculo` |
| **Ambos brazos en diagonal hacia arriba** | `ambos_diagonal_arriba` |

Las imágenes pueden ser `.png`, `.jpg` o `.jpeg` (la extensión no importa) y van dentro de una carpeta llamada `robot/`, junto a los archivos `.py`.

---

## 2. Requisitos y versión recomendada

| Requisito | Versión |
|---|---|
| **Python** | **3.12** (también funciona 3.10 y 3.11; evitar 3.13 o superior) |
| numpy | `>=1.23, <2` |
| opencv-python | `>=4.6, <4.11` |
| **mediapipe** | **`0.10.14`** (fija) |
| Hardware | Cámara web |
| Sistema | Windows (PowerShell) |

> **Importante:** MediaPipe debe estar fijado en la versión `0.10.14`. Las versiones más nuevas ya no incluyen la API `mp.solutions` que usa este proyecto y producen el error `module 'mediapipe' has no attribute 'solutions'`.

---

## 3. Archivos esperados en la entrega

```
robot/                       <- carpeta del proyecto
├── app.py                   <- versión alternativa (detección con Holistic)
├── robotEdiaz.py            <- programa principal
├── requirements.txt         <- dependencias
├── README.md
└── robot/                   <- imágenes del robot (11 archivos)
    ├── ambos_abajo.jpg
    ├── ambos_arriba.jpg
    ├── ambos_extendida.jpg
    ├── derecha_arriba_izq_abajo.jpg
    ├── derecha_arriba_izq_extendida.jpg
    ├── derecha_extendida_izq_abajo.jpg
    ├── izquierda_arriba_der_abajo.jpg
    ├── izquierda_arriba_der_extendida.jpg
    ├── izquierda_extendida_der_abajo.jpg
    ├── ambos_musculo.jpg
    └── ambos_diagonal_arriba.jpg
```

Archivos obligatorios: `app.py`, `robotEdiaz.py` y `requirements.txt`.

---

## 4. Comandos para configurar y ejecutar

Abrir PowerShell **dentro de la carpeta del proyecto** y ejecutar:

**1) Crear el entorno virtual con Python 3.12**

```powershell
py -3.12 -m venv .venv
```

**2) Activar el entorno virtual**

```powershell
.venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea la activación por permisos, ejecutar primero:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**3) Instalar las dependencias**

```powershell
pip install -r requirements.txt
```

**4) Ejecutar el programa**

```powershell
python robotEdiaz.py
```

Se abrirán dos ventanas: tu cámara (con el esqueleto detectado) y el robot imitando tu pose.

### Controles de teclado (`robotEdiaz.py`)

| Tecla | Acción |
|---|---|
| `ESC` | Salir |
| `v` | Activar/desactivar mensajes en consola |

### Versión alternativa

```powershell
python app.py
```

Salir con `q` o `ESC`.

---

## Solución de problemas

| Problema | Solución |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | `pip uninstall mediapipe -y` y luego `pip install mediapipe==0.10.14` |
| `[WARN] falta imagen: ...` | Revisar que la carpeta `robot/` esté junto al `.py` y que los nombres coincidan con la tabla |
| No abre la cámara | Cerrar otras aplicaciones que la estén usando (Zoom, Teams, etc.) |
| Una pose no se detecta bien | Alejarse un poco de la cámara para que se vean ambos brazos completos; los umbrales se ajustan en las constantes al inicio de `robotEdiaz.py` |
