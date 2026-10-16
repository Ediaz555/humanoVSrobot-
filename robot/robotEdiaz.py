from __future__ import annotations
import argparse
import math
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp

#-------------------
# UMbrales / constantes (modificables)
# -------------------
VIS_THRESH = 0.45            # visibilidad mínima para confiar en un landmark
HOR_RATIO_THRESHOLD = 0.50   # ratio horizontal mínimo para considerar brazo "horizontal/extendido"
ELBOW_EXT_ANGLE = 150.0      # ángulo codo >= -> considerado extendido (grados)
ANGLE_DOWN_THRESH = -60.0    # umbral para considerar mano "abajo" usando ángulo hombro->muñeca
CONFIRM_FRAMES = 3           # frames necesarios para confirmar estado (histeresis)
SMOOTH_ALPHA = 0.4           # alpha para suavizado exponencial (0..1)
FLEX_ELBOW_MIN = 25.0        # ángulo de codo mínimo (grados)
FLEX_ELBOW_MAX = 115.0       # ángulo de codo máximo (grados)
FLEX_UPPER_TILT_MAX = 0.60   # |codo_y - hombro_y| / largo_brazo_sup  (0 = perfectamente horizontal)
FLEX_ELBOW_OUT_MIN = 0.40    # separación horizontal del codo hacia afuera / largo_brazo_sup
FLEX_FOREARM_UP_MIN = 0.50   # (codo_y - muñeca_y) / largo_antebrazo  (1 = antebrazo vertical)
FLEX_WRIST_OUT_MIN = 0.25    # muñeca hacia afuera del hombro / largo_brazo_sup (descarta manos en la cabeza)
DIAG_ELBOW_MIN = 140.0       # brazo casi estirado
DIAG_ELEV_MIN = 25.0         # elevación mínima sobre la horizontal (grados)
DIAG_ELEV_MAX = 70.0         # elevación máxima (más que esto = brazo vertical -> both_up)

# Carpeta "robot" junto a ESTE archivo, sin importar desde dónde se ejecute
ROBOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot")
IMG_EXTS = (".png", ".jpg", ".jpeg")   # extensiones aceptadas (da igual cuál uses)

# Mapeo de estados -> nombres de archivos (exactos)
STATE_TO_FILE = {
    "both_down": "ambos_abajo.png",
    "both_up": "ambos_arriba.png",
    "both_extended": "ambos_extendida.png",
    "right_up_left_down": "derecha_arriba_izq_abajo.png",
    "right_up_left_extended": "derecha_arriba_izq_extendida.png",
    "right_extended_left_down": "derecha_extendida_izq_abajo.png",
    "left_up_right_down": "izquierda_arriba_der_abajo.png",
    "left_up_right_extended": "izquierda_arriba_der_extendida.png",
    "left_extended_right_down": "izquierda_extendida_der_abajo.png",
    "both_flex": "ambos_musculo.png",
    "both_diag_up": "ambos_diagonal_arriba.png",
}

# -------------------
# UTILIDADES GEOMÉTRICAS
# -------------------
def angle_between(a: tuple, b: tuple, c: tuple) -> float:
    """
    Ángulo en B entre los vectores BA y BC (grados).
    a, b, c son tuplas (x, y) en píxeles.
    """
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    na = math.hypot(ax, ay)
    nc = math.hypot(cx, cy)
    if na == 0 or nc == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cosv))


def shoulder_to_wrist_angle(shoulder: tuple, wrist: tuple) -> float:
    """
    Ángulo del vector hombro->muñeca usando atan2.
    Invertimos Y para que +90 sea hacia arriba en pantalla.
    Devuelve grados en (-180,180].
    """
    dx = wrist[0] - shoulder[0]
    dy = -(wrist[1] - shoulder[1])  # invertido
    return math.degrees(math.atan2(dy, dx))


def dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def arm_geometry(shoulder: tuple, elbow: tuple, wrist: tuple, mid_x: float):
    """
    Geometría de un brazo relativa al CUERPO (no a la pantalla), así funciona
    igual con cámara normal, espejada o video.

    mid_x = x del punto medio entre ambos hombros. "Afuera" = alejarse de mid_x.
      elbow_out : desplazamiento horizontal del codo hacia afuera (px)
      wrist_out : desplazamiento horizontal de la muñeca hacia afuera (px, - = cruza hacia el centro)
      elbow_dy  : codo_y - hombro_y (px, + = codo más abajo que el hombro)
      forearm_up: codo_y - muñeca_y (px, + = muñeca por encima del codo)
      elev      : elevación hombro->muñeca sobre la horizontal exterior (grados)
                  0 = horizontal hacia afuera, 90 = vertical, >90 = cruzando el cuerpo
    """
    upper_len = dist(shoulder, elbow)
    fore_len = dist(elbow, wrist)
    if upper_len < 1e-6 or fore_len < 1e-6:
        return None
    side = 1.0 if shoulder[0] >= mid_x else -1.0
    wrist_out = (wrist[0] - shoulder[0]) * side
    return {
        "upper_len": upper_len,
        "fore_len": fore_len,
        "elbow_out": (elbow[0] - shoulder[0]) * side,
        "wrist_out": wrist_out,
        "elbow_dy": elbow[1] - shoulder[1],
        "forearm_up": elbow[1] - wrist[1],
        "elev": math.degrees(math.atan2(shoulder[1] - wrist[1], wrist_out)),
    }


def is_flexed(elbow_angle: float, g: dict) -> bool:
    """Brazo haciendo músculo: codo flexionado, húmero ~horizontal hacia afuera, antebrazo hacia arriba."""
    return (FLEX_ELBOW_MIN <= elbow_angle <= FLEX_ELBOW_MAX
            and abs(g["elbow_dy"]) / g["upper_len"] <= FLEX_UPPER_TILT_MAX
            and g["elbow_out"] / g["upper_len"] >= FLEX_ELBOW_OUT_MIN
            and g["forearm_up"] / g["fore_len"] >= FLEX_FOREARM_UP_MIN
            and g["wrist_out"] / g["upper_len"] >= FLEX_WRIST_OUT_MIN)


def is_diag_up(elbow_angle: float, g: dict) -> bool:
    """Brazo estirado apuntando en diagonal hacia arriba y hacia afuera."""
    return elbow_angle >= DIAG_ELBOW_MIN and DIAG_ELEV_MIN <= g["elev"] <= DIAG_ELEV_MAX


# Suavizado exponencial simple
def smooth(prev: float, value: float, alpha: float = SMOOTH_ALPHA) -> float:
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev


# -------------------
# Carga de imágenes del robot (placeholder si falta)
# -------------------
def find_image(fname: str):
    """Busca la imagen probando png, jpg y jpeg. Devuelve la ruta o None."""
    base = os.path.splitext(fname)[0]
    for ext in IMG_EXTS:
        path = os.path.join(ROBOT_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


def load_robot_images(target_size: tuple) -> dict:
    imgs = {}
    w, h = target_size
    for state, fname in STATE_TO_FILE.items():
        path = find_image(fname)
        if path is None:
            print(f"[WARN] falta imagen: {os.path.join(ROBOT_DIR, fname)} (png/jpg/jpeg)")
            imgs[state] = make_placeholder(w, h, text=f"falta: {fname}")
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] no se pudo leer imagen: {path}")
            imgs[state] = make_placeholder(w, h, text=f"error: {fname}")
        else:
            imgs[state] = cv2.resize(img, (w, h))
    return imgs


def make_placeholder(w: int, h: int, text: str = "missing") -> np.ndarray:
    """Crea una imagen gris con texto central para usar como placeholder."""
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    cv2.putText(img, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# -------------------
# Detección de estado (reglas simples)
# -------------------
class SimpleStateDetector:
    def __init__(self, confirm_frames: int = CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.max_count = confirm_frames * 2   # tope: evita que un estado viejo "pese" demasiado
        self.counters = defaultdict(int)
        self.current = "both_down"

    def update(self, candidates: list) -> str:
        # subir contador de los candidatos, bajar el de los demás (con tope)
        for s in STATE_TO_FILE.keys():
            if s in candidates:
                self.counters[s] = min(self.max_count, self.counters[s] + 1)
            else:
                self.counters[s] = max(0, self.counters[s] - 1)
        # candidatos ya confirmados; si el actual sigue siendo válido, se queda
        ready = [s for s in candidates if self.counters[s] >= self.confirm_frames]
        if ready and self.current not in ready:
            best = max(ready, key=lambda s: self.counters[s])
            print(f"[INFO] transición confirmada: {self.current} -> {best}")
            self.current = best
        return self.current


def frame_candidates(L_elbow_ang, R_elbow_ang, L_sh_wrist_ang, R_sh_wrist_ang,
                     L_hor_ratio, R_hor_ratio, L_vis_ok, R_vis_ok, hip_y,
                     L_geo=None, R_geo=None):
    """
    Heurística simple que devuelve lista de estados candidatos (pueden ser múltiples).
    Reglas (mínimo viable):
      - Extended: elbow_angle >= ELBOW_EXT_ANGLE AND hor_ratio >= HOR_RATIO_THRESHOLD
      - Up: wrist_y < shoulder_y  (usamos sh_wrist_ang > 0 para simplificar además)
      - Down: wrist_y > hip_y
    """
    cands = []

    # definiciones booleanas sencillas
    L_extended = (L_elbow_ang >= ELBOW_EXT_ANGLE) and (L_hor_ratio >= HOR_RATIO_THRESHOLD)
    R_extended = (R_elbow_ang >= ELBOW_EXT_ANGLE) and (R_hor_ratio >= HOR_RATIO_THRESHOLD)

    L_up = L_sh_wrist_ang > 45  # si vector hombro->muñeca apunta hacia arriba
    R_up = R_sh_wrist_ang > 45

    L_down = L_sh_wrist_ang < ANGLE_DOWN_THRESH or (L_sh_wrist_ang < 0 and L_sh_wrist_ang < (hip_y * -1))
    R_down = R_sh_wrist_ang < ANGLE_DOWN_THRESH or (R_sh_wrist_ang < 0 and R_sh_wrist_ang < (hip_y * -1))

    # simplificar usando visibilidad
    if L_vis_ok and R_vis_ok:
        if L_geo and R_geo:
            if is_flexed(L_elbow_ang, L_geo) and is_flexed(R_elbow_ang, R_geo):
                return ["both_flex"]
            if is_diag_up(L_elbow_ang, L_geo) and is_diag_up(R_elbow_ang, R_geo):
                return ["both_diag_up"]

        if L_up and R_up:
            cands.append("both_up")
        if L_down and R_down:
            cands.append("both_down")
        if L_extended and R_extended:
            cands.append("both_extended")

        if R_up and L_down:
            cands.append("right_up_left_down")
        if R_up and L_extended:
            cands.append("right_up_left_extended")
        if R_extended and L_down:
            cands.append("right_extended_left_down")

        if L_up and R_down:
            cands.append("left_up_right_down")
        if L_up and R_extended:
            cands.append("left_up_right_extended")
        if L_extended and R_down:
            cands.append("left_extended_right_down")

    return cands


# -------------------
# MAIN
# -------------------
def main():
    parser = argparse.ArgumentParser(description="robot.py minimal - Tú vs Robot (didáctico)")
    parser.add_argument("--video", help="Ruta a archivo de video (opcional). Si no, usa cámara 0.")
    args = parser.parse_args()

    # abrir captura
    if args.video:
        if not os.path.exists(args.video):
            print(f"[ERROR] archivo de video no encontrado: {args.video}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la fuente de video (cámara/archivo).")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # cargar imágenes del robot (derecha)
    robot_imgs = load_robot_images((width, height))

    # init mediapipe pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    detector = SimpleStateDetector(confirm_frames=CONFIRM_FRAMES)

    # variables de suavizado
    prev_vals = {
        "L_elbow": None,
        "R_elbow": None,
        "L_shwr_ang": None,
        "R_shwr_ang": None,
        "L_hor": None,
        "R_hor": None,
    }

    show_console = False

    print("Iniciando. Presiona ESC para salir, 'v' para alternar logs en consola.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] fin de stream o error leyendo frame.")
                break

            orig = frame.copy()  # procesamos el original con MediaPipe (no flip)
            h, w = orig.shape[:2]

            # ---- MediaPipe ----
            img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            # defaults
            L_elbow_ang = 0.0
            R_elbow_ang = 0.0
            L_shwr_ang = 0.0
            R_shwr_ang = 0.0
            L_hor_ratio = 0.0
            R_hor_ratio = 0.0
            L_vis_ok = False
            R_vis_ok = False
            hip_y = h  # usamos bottom como referencia si no hay cadera
            L_geo = None
            R_geo = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # indices útiles
                LEFT_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                RIGHT_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                LEFT_EL = mp_pose.PoseLandmark.LEFT_ELBOW.value
                RIGHT_EL = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                LEFT_WR = mp_pose.PoseLandmark.LEFT_WRIST.value
                RIGHT_WR = mp_pose.PoseLandmark.RIGHT_WRIST.value
                LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
                RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

                def to_px(i):
                    l = lm[i]
                    return (l.x * w, l.y * h), getattr(l, "visibility", 1.0)

                L_sh_px, L_sh_vis = to_px(LEFT_SH)
                R_sh_px, R_sh_vis = to_px(RIGHT_SH)
                L_el_px, L_el_vis = to_px(LEFT_EL)
                R_el_px, R_el_vis = to_px(RIGHT_EL)
                L_wr_px, L_wr_vis = to_px(LEFT_WR)
                R_wr_px, R_wr_vis = to_px(RIGHT_WR)
                L_hip_px, L_hip_vis = to_px(LEFT_HIP)
                R_hip_px, R_hip_vis = to_px(RIGHT_HIP)

                # hip y promedio (si está)
                hip_y = ((L_hip_px[1] if L_hip_px else h) + (R_hip_px[1] if R_hip_px else h)) / 2.0

                # visibilidad global por brazo
                L_vis_ok = (L_sh_vis >= VIS_THRESH and L_el_vis >= VIS_THRESH and L_wr_vis >= VIS_THRESH)
                R_vis_ok = (R_sh_vis >= VIS_THRESH and R_el_vis >= VIS_THRESH and R_wr_vis >= VIS_THRESH)

                # geometría relativa al cuerpo para los movimientos nuevos
                mid_x = (L_sh_px[0] + R_sh_px[0]) / 2.0
                L_geo = arm_geometry(L_sh_px, L_el_px, L_wr_px, mid_x)
                R_geo = arm_geometry(R_sh_px, R_el_px, R_wr_px, mid_x)

                # calcular ángulo de codo (shoulder-elbow-wrist)
                if all(v is not None for v in (L_sh_px, L_el_px, L_wr_px)):
                    raw = angle_between(L_sh_px, L_el_px, L_wr_px)
                    L_elbow_ang = smooth(prev_vals["L_elbow"], raw)
                    prev_vals["L_elbow"] = L_elbow_ang

                if all(v is not None for v in (R_sh_px, R_el_px, R_wr_px)):
                    raw = angle_between(R_sh_px, R_el_px, R_wr_px)
                    R_elbow_ang = smooth(prev_vals["R_elbow"], raw)
                    prev_vals["R_elbow"] = R_elbow_ang

                # ángulo hombro->muñeca (dirección)
                if L_sh_px and L_wr_px:
                    raw = shoulder_to_wrist_angle(L_sh_px, L_wr_px)
                    L_shwr_ang = smooth(prev_vals["L_shwr_ang"], raw)
                    prev_vals["L_shwr_ang"] = L_shwr_ang
                if R_sh_px and R_wr_px:
                    raw = shoulder_to_wrist_angle(R_sh_px, R_wr_px)
                    R_shwr_ang = smooth(prev_vals["R_shwr_ang"], raw)
                    prev_vals["R_shwr_ang"] = R_shwr_ang

                # ratio horizontal normalizada
                if all(v is not None for v in (L_sh_px, L_el_px, L_wr_px)):
                    denom = dist(L_sh_px, L_el_px) + dist(L_el_px, L_wr_px)
                    if denom > 1e-6:
                        raw = abs(L_wr_px[0] - L_sh_px[0]) / denom
                        L_hor_ratio = smooth(prev_vals["L_hor"], raw)
                        prev_vals["L_hor"] = L_hor_ratio
                if all(v is not None for v in (R_sh_px, R_el_px, R_wr_px)):
                    denom = dist(R_sh_px, R_el_px) + dist(R_el_px, R_wr_px)
                    if denom > 1e-6:
                        raw = abs(R_wr_px[0] - R_sh_px[0]) / denom
                        R_hor_ratio = smooth(prev_vals["R_hor"], raw)
                        prev_vals["R_hor"] = R_hor_ratio

                # dibujar landmarks sobre una copia para mostrar (mirroring al final)
                disp = orig.copy()
                mp.solutions.drawing_utils.draw_landmarks(disp, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                disp = orig.copy()

            # calcular candidatos de estado según heurística simple
            candidates = frame_candidates(
                L_elbow_ang=L_elbow_ang,
                R_elbow_ang=R_elbow_ang,
                L_sh_wrist_ang=L_shwr_ang,
                R_sh_wrist_ang=R_shwr_ang,
                L_hor_ratio=L_hor_ratio,
                R_hor_ratio=R_hor_ratio,
                L_vis_ok=L_vis_ok,
                R_vis_ok=R_vis_ok,
                hip_y=hip_y,
                L_geo=L_geo,
                R_geo=R_geo
            )

            state = detector.update(candidates)

            # seleccionar imagen del robot para el estado
            robot_img = robot_imgs.get(state, make_placeholder(width, height, text=state))

            # preparar ventana izquierda: mirroring para usuario (flip horizontal)
            disp_shown = cv2.flip(disp, 1)

            # overlay simple con valores (izquierda)
            cv2.putText(disp_shown, f"State: {state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp_shown, f"L_elb:{L_elbow_ang:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
            cv2.putText(disp_shown, f"R_elb:{R_elbow_ang:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
            cv2.putText(disp_shown, f"L_hor:{L_hor_ratio:.2f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(disp_shown, f"R_hor:{R_hor_ratio:.2f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            if L_geo and R_geo:  # útil para calibrar los umbrales de diagonal
                cv2.putText(disp_shown, f"elev L/R:{L_geo['elev']:.0f}/{R_geo['elev']:.0f}", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 1)

            # ventana derecha ya está en orientación normal (sin flip)
            # mostramos las dos ventanas separadas
            cv2.imshow("You - Camera (left)", disp_shown)
            cv2.imshow("Robot - State (right)", robot_img)

            # manejo teclado
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("v"):
                show_console = not show_console
            # logs opcionales
            if show_console:
                print(f"DEBUG state={state} L_elb={L_elbow_ang:.1f} R_elb={R_elbow_ang:.1f} L_hor={L_hor_ratio:.2f} R_hor={R_hor_ratio:.2f}")

    except KeyboardInterrupt:
        print("[INFO] interrumpido por usuario")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
