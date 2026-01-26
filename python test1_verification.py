import cv2
import mediapipe as mp
import time
import sys
import numpy as np

sys.path.append('./') 

# --- CONFIGURATION ---
ROBOT_IP = "192.168.1.228" 
SIMULATION_MODE = True  # <--- SET THIS TO TRUE FOR HOME MEETING!

# --- IMPORT DRIVER ---
if SIMULATION_MODE:
    print(">>> RUNNING IN SIMULATION MODE (HOME)")
    from myLibs.ufactory.mock_xarm import MockXArm as xArm7 # Rename it to trick the code
else:
    print(">>> RUNNING IN LAB MODE (REAL ROBOT)")
    from myLibs.ufactory.xarm_class_joint_space import xArm7

# --- INITIALIZATION ---
print(">>> TEST 1: SYSTEM STARTUP...")

try:
    # This code looks exactly the same, but now uses the Mock robot!
    robot = xArm7(ROBOT_IP) 
    robot.start_up() 
    robot_connected = True
    print(">>> SUCCESS: System Connected!")
except Exception as e:
    print(f">>> ERROR: {e}")
    robot_connected = False

# --- VISION SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
pTime = 0

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # Process
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Get (Fake) Robot Data
    robot_text = "Disconnected"
    color = (0, 0, 255)
    
    if robot_connected:
        pos = robot.get_states()
        # The Mock robot adds random 'noise' so these numbers will change!
        robot_text = f"XYZ: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
        color = (0, 255, 0)

    # Draw
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)
            
    # Dashboard
    cv2.putText(image, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(image, robot_text, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    if SIMULATION_MODE:
        cv2.putText(image, "SIMULATION MODE", (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

    cv2.imshow('Test 1: Verification', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if robot_connected:
    robot.destroy()