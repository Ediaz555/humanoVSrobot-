"""
Detección de pose con MediaPipe Holistic: calcula ángulos de los brazos,
clasifica el estado combinado y muestra la imagen del robot correspondiente.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import os
from collections import deque

# ==================== CONSTANTES ====================

# Estados básicos
EXTENDED_ELBOW_ANGLE = 150
EXTENDED_HOR_RATIO = 0.55
DOWN_ANGLE_THRESHOLD = -60

# Músculo: húmero ~horizontal hacia afuera, antebrazo hacia arriba
FLEX_ELBOW_MIN = 25.0
FLEX_ELBOW_MAX = 115.0
FLEX_UPPER_TILT_MAX = 0.60   # |codo_y - hombro_y| / largo brazo sup
FLEX_ELBOW_OUT_MIN = 0.40    # codo hacia afuera / largo brazo sup
FLEX_FOREARM_UP_MIN = 0.50   # (codo_y - muñeca_y) / largo antebrazo
FLEX_WRIST_OUT_MIN = 0.25    # muñeca hacia afuera / largo brazo sup (descarta manos en la cabeza)
# Diagonal hacia arriba (forma de "V")
DIAG_ELBOW_MIN = 140.0       # brazo casi estirado
DIAG_ELEV_MIN = 25.0         # elevación mínima sobre la horizontal (grados)
DIAG_ELEV_MAX = 70.0         # más que esto = brazo vertical -> both_up

# Anti-ruido
CONFIRM_FRAMES = 3   # frames para confirmar cambio de estado
ALPHA = 0.3          # suavizado exponencial (0..1)

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Carpeta "robot" junto a este archivo, sin importar desde dónde se ejecute
ROBOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot")
IMG_EXTS = (".png", ".jpg", ".jpeg")

ROBOT_WINDOW_WIDTH = 640
ROBOT_WINDOW_HEIGHT = 480

# ==================== INICIALIZACIÓN MEDIAPIPE ====================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==================== CLASES AUXILIARES ====================

class SmoothedValue:
    """Suavizado exponencial: alpha alto = reacciona rápido, alpha bajo = más estable."""
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


class StateConfirmation:
    """Histéresis: el estado solo cambia tras CONFIRM_FRAMES detecciones seguidas iguales."""
    def __init__(self, confirm_frames=CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.current_state = "unknown"
        self.candidate_state = "unknown"
        self.counter = 0
    
    def update(self, detected_state):
        if detected_state == self.candidate_state:
            self.counter = min(self.counter + 1, self.confirm_frames)
        else:
            self.candidate_state = detected_state
            self.counter = 0
        
        if self.counter >= self.confirm_frames:
            self.current_state = self.candidate_state
        
        return self.current_state


# ==================== FUNCIONES DE CÁLCULO ====================

def angle_between(a, b, c):
    """Ángulo en b formado por a-b-c (producto punto + arccos)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product < 1e-8:
        return 0.0
    
    cosang = np.dot(ba, bc) / norm_product
    cosang = np.clip(cosang, -1.0, 1.0)  # evita errores numéricos en arccos
    
    return np.degrees(np.arccos(cosang))


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def shoulder_to_wrist_angle(shoulder, wrist):
    """Dirección hombro->muñeca: 0° = derecha, 90° = arriba, -90° = abajo."""
    vx = wrist[0] - shoulder[0]
    vy = shoulder[1] - wrist[1]  # Y invertida (píxeles)
    
    return math.degrees(math.atan2(vy, vx))


def calculate_horizontal_ratio(shoulder, elbow, wrist):
    """|muñeca_x - hombro_x| / largo total del brazo. Distingue brazo horizontal de brazo pegado al cuerpo."""
    arm_len = distance(shoulder, elbow) + distance(elbow, wrist)
    if arm_len < 1.0:
        return 0.0
    
    horizontal_distance = abs(wrist[0] - shoulder[0])
    return horizontal_distance / arm_len


def arm_geometry(shoulder, elbow, wrist, mid_x):
    """
    Geometría del brazo relativa al CUERPO (no a la pantalla).
    "Afuera" = alejarse de mid_x (x del punto medio entre hombros).
      elbow_out : codo hacia afuera (px)
      wrist_out : muñeca hacia afuera (px, negativo = cruza hacia el centro)
      elbow_dy  : codo_y - hombro_y (px, + = codo más abajo)
      forearm_up: codo_y - muñeca_y (px, + = muñeca sobre el codo)
      elev      : elevación hombro->muñeca (0 = horizontal, 90 = vertical, >90 = cruza el cuerpo)
    """
    upper_len = distance(shoulder, elbow)
    fore_len = distance(elbow, wrist)
    if upper_len < 1e-6 or fore_len < 1e-6:
        return None
    side = 1.0 if shoulder[0] >= mid_x else -1.0
    wrist_out = (wrist[0] - shoulder[0]) * side
    return {
        'upper_len': upper_len,
        'fore_len': fore_len,
        'elbow_out': (elbow[0] - shoulder[0]) * side,
        'wrist_out': wrist_out,
        'elbow_dy': elbow[1] - shoulder[1],
        'forearm_up': elbow[1] - wrist[1],
        'elev': math.degrees(math.atan2(shoulder[1] - wrist[1], wrist_out)),
    }


def update_pose_flags(analysis):
    """
    Calcula is_flexed / is_diag_up desde elbow_angle + geometría.
    Se llama en analyze_arm y de nuevo tras suavizar el ángulo de codo.
    """
    g = analysis.get('geo')
    if g is None:
        analysis['is_flexed'] = False
        analysis['is_diag_up'] = False
        return analysis

    elbow = analysis['elbow_angle']
    analysis['is_flexed'] = (
        FLEX_ELBOW_MIN <= elbow <= FLEX_ELBOW_MAX
        and abs(g['elbow_dy']) / g['upper_len'] <= FLEX_UPPER_TILT_MAX
        and g['elbow_out'] / g['upper_len'] >= FLEX_ELBOW_OUT_MIN
        and g['forearm_up'] / g['fore_len'] >= FLEX_FOREARM_UP_MIN
        and g['wrist_out'] / g['upper_len'] >= FLEX_WRIST_OUT_MIN
    )
    analysis['is_diag_up'] = (
        elbow >= DIAG_ELBOW_MIN and DIAG_ELEV_MIN <= g['elev'] <= DIAG_ELEV_MAX
    )
    return analysis


# ==================== EXTRACCIÓN DE LANDMARKS ====================

def get_landmarks_coords(landmarks, width, height, indices):
    """Landmarks normalizados -> coordenadas en píxeles."""
    coords = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        px = int(lm.x * width)
        py = int(lm.y * height)
        coords.append((px, py))
    
    return coords


def extract_arm_data(landmarks, width, height, side='right'):
    """Devuelve hombro, codo, muñeca y cadera del lado indicado ('right' o 'left')."""
    if side == 'right':
        shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        elbow_idx = mp_holistic.PoseLandmark.RIGHT_ELBOW.value
        wrist_idx = mp_holistic.PoseLandmark.RIGHT_WRIST.value
        hip_idx = mp_holistic.PoseLandmark.RIGHT_HIP.value
    else:
        shoulder_idx = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        elbow_idx = mp_holistic.PoseLandmark.LEFT_ELBOW.value
        wrist_idx = mp_holistic.PoseLandmark.LEFT_WRIST.value
        hip_idx = mp_holistic.PoseLandmark.LEFT_HIP.value
    
    coords = get_landmarks_coords(landmarks, width, height, 
                                   [shoulder_idx, elbow_idx, wrist_idx, hip_idx])
    
    return {
        'shoulder': coords[0],
        'elbow': coords[1],
        'wrist': coords[2],
        'hip': coords[3]
    }


# ==================== DETECCIÓN DE ESTADOS ====================

def analyze_arm(arm_data, mid_x=None):
    """
    Analiza un brazo. Devuelve dict con elbow_angle, hor_ratio, direction_angle,
    is_extended, is_up, is_down, geo, is_flexed, is_diag_up.
    """
    shoulder = arm_data['shoulder']
    elbow = arm_data['elbow']
    wrist = arm_data['wrist']
    hip = arm_data['hip']
    
    elbow_angle = angle_between(shoulder, elbow, wrist)
    hor_ratio = calculate_horizontal_ratio(shoulder, elbow, wrist)
    direction_angle = shoulder_to_wrist_angle(shoulder, wrist)
    
    is_extended = elbow_angle > EXTENDED_ELBOW_ANGLE and hor_ratio > EXTENDED_HOR_RATIO
    is_up = wrist[1] < shoulder[1]
    is_down = wrist[1] > hip[1] or direction_angle < DOWN_ANGLE_THRESHOLD
    
    analysis = {
        'elbow_angle': elbow_angle,
        'hor_ratio': hor_ratio,
        'direction_angle': direction_angle,
        'is_extended': is_extended,
        'is_up': is_up,
        'is_down': is_down,
        'geo': arm_geometry(shoulder, elbow, wrist, mid_x) if mid_x is not None else None,
    }
    return update_pose_flags(analysis)


def detect_combined_state(right_analysis, left_analysis):
    """Estado combinado de ambos brazos. El orden de los if define la prioridad."""
    r_ext = right_analysis['is_extended']
    r_up = right_analysis['is_up']
    r_down = right_analysis['is_down']
    
    l_ext = left_analysis['is_extended']
    l_up = left_analysis['is_up']
    l_down = left_analysis['is_down']
    
    # Músculo y diagonal primero: si no, también cuentan como "both_up"
    # (la muñeca queda sobre el hombro)
    if right_analysis['is_flexed'] and left_analysis['is_flexed']:
        return "both_flex"
    if right_analysis['is_diag_up'] and left_analysis['is_diag_up']:
        return "both_diag_up"

    if r_down and l_down:
        return "both_down"
    if r_up and l_up:
        return "both_up"
    if r_ext and l_ext:
        return "both_extended"
    
    if r_up and l_down:
        return "right_up_left_down"
    if r_up and l_ext:
        return "right_up_left_extended"
    if r_ext and l_down:
        return "right_extended_left_down"
    
    if l_up and r_down:
        return "left_up_right_down"
    if l_up and r_ext:
        return "left_up_right_extended"
    if l_ext and r_down:
        return "left_extended_right_down"
    
    return "neutral"


# ==================== VISUALIZACIÓN ====================

def draw_text_with_background(img, text, pos, font_scale=0.6, thickness=2):
    """Texto con fondo negro para que se lea sobre cualquier imagen."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y + 5), 
                  (0, 0, 0), -1)
    
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def draw_debug_overlay(img, right_analysis, left_analysis, state):
    height, width = img.shape[:2]
    
    draw_text_with_background(img, "Pose Detection Debug", (10, 30), 0.7, 2)
    draw_text_with_background(img, f"State: {state}", (10, 70), 0.8, 2)
    
    y_offset = 110
    draw_text_with_background(img, "RIGHT ARM:", (10, y_offset), 0.6, 1)
    draw_text_with_background(img, f"  Elbow: {right_analysis['elbow_angle']:.1f}deg", 
                            (10, y_offset + 25), 0.5, 1)
    draw_text_with_background(img, f"  H-Ratio: {right_analysis['hor_ratio']:.2f}", 
                            (10, y_offset + 50), 0.5, 1)
    draw_text_with_background(img, f"  Dir: {right_analysis['direction_angle']:.1f}deg", 
                            (10, y_offset + 75), 0.5, 1)
    
    y_offset = 210
    draw_text_with_background(img, "LEFT ARM:", (10, y_offset), 0.6, 1)
    draw_text_with_background(img, f"  Elbow: {left_analysis['elbow_angle']:.1f}deg", 
                            (10, y_offset + 25), 0.5, 1)
    draw_text_with_background(img, f"  H-Ratio: {left_analysis['hor_ratio']:.2f}", 
                            (10, y_offset + 50), 0.5, 1)
    draw_text_with_background(img, f"  Dir: {left_analysis['direction_angle']:.1f}deg", 
                            (10, y_offset + 75), 0.5, 1)
    
    # Útil para calibrar los umbrales de diagonal
    if right_analysis.get('geo') and left_analysis.get('geo'):
        draw_text_with_background(
            img, f"Elev R/L: {right_analysis['geo']['elev']:.0f}/{left_analysis['geo']['elev']:.0f}deg",
            (10, 310), 0.5, 1)

    draw_text_with_background(img, "Press 'q' or ESC to quit", 
                            (10, height - 20), 0.5, 1)


def load_robot_images(robot_dir=ROBOT_DIR):
    """Carga las imágenes del robot (estado -> imagen). Los estados sin archivo se omiten."""
    robot_images = {}
    
    if not os.path.exists(robot_dir):
        print(f"Warning: Directorio {robot_dir}/ no encontrado")
        return robot_images
    
    # Estado (inglés) -> archivo (español). "neutral" reutiliza ambos_abajo
    state_mapping = {
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
        "neutral": "ambos_abajo.png"  
    }
    
    for state, filename in state_mapping.items():
        base = os.path.splitext(filename)[0]
        img_path = None
        for ext in IMG_EXTS:  
            candidate = os.path.join(robot_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"[WARN] falta imagen: {os.path.join(robot_dir, base)} (png/jpg/jpeg)")
            continue
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (ROBOT_WINDOW_WIDTH, ROBOT_WINDOW_HEIGHT))
            robot_images[state] = img
            print(f"Loaded: {state} -> {img_path}")
    
    return robot_images


def get_robot_display(robot_images, state):
    """Imagen del robot para el estado, o una imagen de reemplazo si falta."""
    if state in robot_images:
        return robot_images[state].copy()
    
    default_img = np.zeros((ROBOT_WINDOW_HEIGHT, ROBOT_WINDOW_WIDTH, 3), dtype=np.uint8)
    cv2.putText(default_img, f"State: {state}", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(default_img, "(Image not found)", (150, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    return default_img


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    parser = argparse.ArgumentParser(description='Detección de Pose con MediaPipe')
    parser.add_argument('--video', type=str, default=None, 
                       help='Ruta al archivo de video (opcional, por defecto usa webcam)')
    args = parser.parse_args()
    
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Cargando video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Usando webcam")
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la fuente de video")
        return
    
    robot_images = load_robot_images(ROBOT_DIR)
    
    state_confirmer = StateConfirmation(CONFIRM_FRAMES)
    
    # Un suavizador por valor y por brazo
    right_elbow_smooth = SmoothedValue(ALPHA)
    right_ratio_smooth = SmoothedValue(ALPHA)
    left_elbow_smooth = SmoothedValue(ALPHA)
    left_ratio_smooth = SmoothedValue(ALPHA)
    
    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as holistic:
        
        print("\n=== Aplicación iniciada ===")
        print("Controles: 'q' o ESC para salir\n")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Fin del video o error de lectura")
                break
            
            height, width, _ = frame.shape
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe espera RGB
            results = holistic.process(image_rgb)
            
            annotated_image = frame.copy()
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                right_arm = extract_arm_data(results.pose_landmarks, width, height, 'right')
                left_arm = extract_arm_data(results.pose_landmarks, width, height, 'left')
                
                mid_x = (right_arm['shoulder'][0] + left_arm['shoulder'][0]) / 2.0
                right_analysis = analyze_arm(right_arm, mid_x)
                left_analysis = analyze_arm(left_arm, mid_x)
                
                # Suavizar ángulo de codo y ratio horizontal
                right_analysis['elbow_angle'] = right_elbow_smooth.update(right_analysis['elbow_angle'])
                right_analysis['hor_ratio'] = right_ratio_smooth.update(right_analysis['hor_ratio'])
                left_analysis['elbow_angle'] = left_elbow_smooth.update(left_analysis['elbow_angle'])
                left_analysis['hor_ratio'] = left_ratio_smooth.update(left_analysis['hor_ratio'])
                
                # Re-evaluar las banderas con los valores ya suavizados
                right_analysis['is_extended'] = (right_analysis['elbow_angle'] > EXTENDED_ELBOW_ANGLE and 
                                                 right_analysis['hor_ratio'] > EXTENDED_HOR_RATIO)
                left_analysis['is_extended'] = (left_analysis['elbow_angle'] > EXTENDED_ELBOW_ANGLE and 
                                                left_analysis['hor_ratio'] > EXTENDED_HOR_RATIO)
                update_pose_flags(right_analysis)
                update_pose_flags(left_analysis)
                
                detected_state = detect_combined_state(right_analysis, left_analysis)
                confirmed_state = state_confirmer.update(detected_state)
                
                draw_debug_overlay(annotated_image, right_analysis, left_analysis, confirmed_state)
                robot_display = get_robot_display(robot_images, confirmed_state)
            else:
                draw_text_with_background(annotated_image, "No pose detected", (10, 30), 0.8, 2)
                robot_display = get_robot_display(robot_images, "neutral")
            
            cv2.imshow('Camera Feed - Pose Detection', annotated_image)
            cv2.imshow('Robot State', robot_display)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:  # 'q' o ESC
                print("Saliendo...")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación cerrada")


if __name__ == "__main__":
    main()
