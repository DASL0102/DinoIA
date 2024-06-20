import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np

# Inicializar mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Inicializar pygame
pygame.init()
screen = pygame.display.set_mode((800, 400))
clock = pygame.time.Clock()

# Variables del juego
dino_rect = pygame.Rect(100, 300, 50, 50)
gravity = 0
is_jumping = False
obstacles = []
distance = 0  # Variable de distancia
distance_increment_counter = 0  # Contador para controlar la velocidad de incremento de la distancia
font = pygame.font.Font(None, 36)  # Fuente para mostrar la distancia

obstacle_timer = pygame.USEREVENT + 1
pygame.time.set_timer(obstacle_timer, 3000)  # Aumentar el tiempo entre obstáculos

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def draw_dino():
    pygame.draw.rect(screen, BLACK, dino_rect)

def create_obstacle():
    obstacle_rect = pygame.Rect(800, 300, 30, 50)  # Obstáculo más estrecho
    return obstacle_rect

def move_obstacles(obstacles):
    for obstacle in obstacles:
        obstacle.x -= 5
    return [obstacle for obstacle in obstacles if obstacle.x > -50]

def draw_obstacles(obstacles):
    for obstacle in obstacles:
        pygame.draw.rect(screen, BLACK, obstacle)

def detect_collisions(dino, obstacles):
    for obstacle in obstacles:
        if dino.colliderect(obstacle):
            return True
    return False

def detect_hand_movement(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detectar el punto de la muñeca (landmark 0)
            wrist_y = hand_landmarks.landmark[0].y
            
            # Ajustar el umbral para detectar el movimiento hacia arriba
            if wrist_y < 0.5:  # Movimiento hacia arriba más sensible
                return True
    return False

def draw_detection_results(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
            )
    return image

# Bucle principal del juego
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == obstacle_timer:
                obstacles.append(create_obstacle())

        # Procesar la imagen de la cámara
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Voltear la imagen para una mejor experiencia de usuario
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if detect_hand_movement(frame) and not is_jumping:
            is_jumping = True
            gravity = -25  # Fuerza del salto incrementada

        # Lógica del juego
        if is_jumping:
            gravity += 1.5  # Gravedad ajustada
            dino_rect.y += gravity
            if dino_rect.y >= 300:  # Suelo del juego
                dino_rect.y = 300
                is_jumping = False
                gravity = 0  # Resetear la gravedad

        obstacles = move_obstacles(obstacles)
        
        # Incrementar la distancia cada 10 frames
        distance_increment_counter += 1
        if distance_increment_counter >= 10:
            distance += 1
            distance_increment_counter = 0

        if detect_collisions(dino_rect, obstacles):
            cap.release()
            pygame.quit()
            sys.exit()

        # Dibujar el frame del juego
        screen.fill(WHITE)
        draw_dino()
        draw_obstacles(obstacles)
        
        # Mostrar la distancia en pantalla
        distance_text = font.render(f"Distance: {distance}", True, BLACK)
        screen.blit(distance_text, (10, 10))

        pygame.display.flip()
        clock.tick(30)

        # Mostrar la cámara con los resultados de la detección de manos
        frame = draw_detection_results(frame, results)

        # Mostrar la precisión en la pantalla
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                cv2.putText(frame, f'Precision: {results.multi_handedness[0].classification[0].score:.2f}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
