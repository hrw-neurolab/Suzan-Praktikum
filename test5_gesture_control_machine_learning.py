"""
Praktikum Thesis: Robot Control System
Final Build: Custom ML Gesture Control via TFLite
Description: 
    - Replaces MediaPipe with Custom MobileNetV2 Model.
    - Decoupled Control: 9 Gestures (Translations & Gripper).
Author: Suzan Elmasry
24.03.2026
"""

import cv2                          # Handles webcam streaming
import numpy as np                  # Handles matrix math
import tensorflow as tf             # Used for the TFLite Interpreter to run the model
import os                           # locates the path to the .tflite model file
import sys                          # locate the custom xArm7 hardware
import time                         # Enforces the strict (dt)


# 1. SETUP ROBOT CONNECTION
sys.path.append('./') 

try:
    from myLibs.ufactory.xarm_class_joint_space import xArm7
except ImportError:
    print("xArm Library not found. Running in Simulation Mode.")
    xArm7 = None

ROBOT_IP = "10.2.134.151"
robot = None

if xArm7:
    try:
        robot = xArm7(ROBOT_IP)
        robot.start_up() 
        print(">>> SUCCESS: Robot & Gripper Connected!")
    except:
        print(">>> ERROR: Robot connection failed")


# 2. SETUP ML MODEL (TFLITE)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "gesture_model_final.tflite")

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print(">>> STATUS: ML System Ready. Inference Pipeline Synchronized.")
except Exception as e:
    print(">>> ERROR: Could not load TFLite model.")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['GRIPPER_CLOSED', 'GRIPPER_OPEN', 'MOVE_BACKWARD', 'MOVE_DOWN', 
               'MOVE_FORWARD', 'MOVE_RIGHT', 'MOVE_LEFT', 'MOVE_UP', 'SAFTEY']


# 3. INITIALIZE VARIABLES & TIMER
cap = cv2.VideoCapture(0)

# Default Robot Position
curr_x, curr_y, curr_z = 300, 0, 200
curr_roll, curr_pitch, curr_yaw = -180, 0, -90 

if robot:
    try:
        pos = robot.get_position()
        if pos and len(pos) >= 6: 
            curr_x, curr_y, curr_z = pos[0], pos[1], pos[2]
            curr_roll, curr_pitch, curr_yaw = pos[3], pos[4], pos[5]
    except: pass

# Motion Parameters
dt = 0.02           #  THE CRITICAL TIMER STEP (20ms)
v = 100             # Translation velocity
step_size = v * dt  # Cartesian translation step per frame 100x0.02=2mm per frame

print(">>> SYSTEM LIVE: Starting Camera Loop...")


# 4. MAIN CONTROL LOOP
while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    
    #  START TIMER
    start_timer = time.time()
    
   
    
    # ML PREPROCESSING & INFERENCE ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Extracts a perfect square from the center of the webcam feed before resizing
    # ensuring the live input  matches the training data 
    h, w, _ = rgb_frame.shape
    min_dim = min(h, w)
    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)
    cropped_frame = rgb_frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    # resizing and normalization
    img_resized = cv2.resize(rgb_frame, (224, 224))
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    input_data = np.expand_dims(img_normalized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(predictions)
    confidence = predictions[idx]
    
    # ROBOT LOGIC MAPPING ---
    command = "STOP"
    gripper_state = "--" 
    should_move = False 
    
    label = class_names[idx]
    
   
    required_confidence = 0.45 if label == 'GRIPPER_CLOSED' else 0.70
    
    if confidence > required_confidence:
        command = label
        
        # Translation Mapping
        if label == 'MOVE_FORWARD': 
            curr_x += step_size
            should_move = True
        elif label == 'MOVE_BACKWARD': 
            curr_x -= step_size
            should_move = True
        elif label == 'MOVE_LEFT': 
            curr_y += step_size
            should_move = True
        elif label == 'MOVE_RIGHT': 
            curr_y -= step_size
            should_move = True
        elif label == 'MOVE_UP': 
            curr_z += step_size
            should_move = True
        elif label == 'MOVE_DOWN': 
            curr_z -= step_size
            should_move = True
            
        # Gripper Mapping
        elif label == 'GRIPPER_OPEN':
            gripper_state = "OPEN"
            if robot: robot.set_gripper_pos(850)
        elif label == 'GRIPPER_CLOSED':
            gripper_state = "CLOSE"
            if robot: robot.set_gripper_pos(0)
            
        # Safety / Stop
        elif label == 'SAFTEY':
            command = "SAFTEY HOLD"
            should_move = False

    # SAFETY BOUNDING BOX ---
    if curr_x > 600: curr_x = 600
    if curr_x < 200: curr_x = 200
    if curr_y > 300: curr_y = 300
    if curr_y < -300: curr_y = -300
    if curr_z > 500: curr_z = 500
    if curr_z < 100: curr_z = 100

    #  EXECUTE ROBOT MOTION ---
    if robot and should_move:
        try:
            # wait=False so the loop doesnt block
            robot.set_position(x=curr_x, y=curr_y, z=curr_z, 
                               roll=curr_roll, pitch=curr_pitch, yaw=curr_yaw, wait=False)
        except: pass

    # UI DASHBOARD 
    # Green if confident, Red if unsure
    # display_frame = cv2.flip(frame, 1) # flipping the frame
    text_color = (0, 255, 0) if confidence > 0.50 else (0, 0, 255)
    cv2.putText(frame, f"CMD: {command} ({int(confidence*100)}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame, f"Gripper: {gripper_state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
    cv2.imshow("xArm ML Control Demo", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # END TIMER & ENFORCE DELAY
    end_time = time.time()
    elapsed = end_time - start_timer
    if elapsed < dt:
        time.sleep(dt - elapsed)

# Cleanup
cap.release()
cv2.destroyAllWindows()
