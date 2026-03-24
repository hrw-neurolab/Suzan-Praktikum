"""
Praktikum Thesis: Robot Control System
Test 4: Two-Hand Gesture Control 
Description: 
    - Decoupled Control: Left hand controls spatial movement (X, Y, Z axes).
    - Gripper Control: Right hand controls Open/Close AND Rotation (Roll +/- 90 deg).
 
Author: Suzan Elmasry
Date: 26 Feb 2026
"""

import cv2
import mediapipe as mp
import sys
import time

# Setup Libraries 
sys.path.append('./') 

try:
    from myLibs.ufactory.xarm_class_joint_space import xArm7
except ImportError:
    print("xArm Library not found")
    xArm7 = None

# Robot Connection 
ROBOT_IP = "10.2.134.151"
robot = None

if xArm7:
    try:
        robot = xArm7(ROBOT_IP)
        # start_up() handles motion enable, mode setting, and initial gripper opening!
        robot.start_up() 
        print(">>> SUCCESS: Robot & Gripper Connected!")
    except:
        print(">>> ERROR: Robot connection failed")

# Vision Setup 
# Initialize MediaPipe Hand Tracking for two hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Initialize Robot Position & Orientation 
curr_x, curr_y, curr_z = 300, 0, 200

# Default Euler Angles
curr_roll, curr_pitch, curr_yaw = -180, 0, -90 
initial_roll = -180 

if robot:
    try:
        # real-time position 
        pos = robot.get_position()
        if pos and len(pos) >= 6: 
            curr_x, curr_y, curr_z = pos[0], pos[1], pos[2]
            curr_roll, curr_pitch, curr_yaw = pos[3], pos[4], pos[5]
            initial_roll = curr_roll 
    except: pass


dt = 0.02           # Time step
v = 100             # Translation velocity
step_size = v * dt  # Cartesian translation step
v_deg = 30          # TRotation velocity
yaw_step = v_deg * dt     # Rotation speed (degrees per frame)


while cap.isOpened():
    success, image = cap.read()
    if not success: continue
    
    start_timer = time.time()
    
  
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # UI Variables
    command = "STOP"
    gripper_state = "--" 
    should_move = False 
    
    if results.multi_hand_landmarks and results.multi_handedness:
        # Loop through all detected hands
        for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)
            hand_label = results.multi_handedness[hand_idx].classification[0].label
            
            # Finger State Analysis 
            fingers = []
            if hand_lms.landmark[4].x < hand_lms.landmark[3].x: fingers.append(1) 
            else: fingers.append(0) 
                
            for id in [8, 12, 16, 20]:
                if hand_lms.landmark[id].y < hand_lms.landmark[id-2].y: fingers.append(1)
                else: fingers.append(0)
            
           
            # LEFT HAND (X, Y, Z)
            
            if hand_label == "Left":
                if fingers == [0, 1, 0, 0, 0]:
                    curr_x += step_size 
                    command = "FORWARD"
                    should_move = True
                elif fingers == [0, 1, 1, 0, 0]:
                    curr_x -= step_size 
                    command = "BACKWARD"
                    should_move = True
                elif fingers == [0, 0, 0, 0, 1]:
                    curr_y -= step_size 
                    command = "RIGHT"
                    should_move = True
                elif fingers == [1, 0, 0, 0, 0]:
                    curr_y += step_size 
                    command = "LEFT"
                    should_move = True
                elif fingers == [0, 1, 1, 1, 0]:
                    curr_z += step_size 
                    command = "UP"
                    should_move = True
                elif fingers == [0, 1, 1, 1, 1]:
                    curr_z -= step_size 
                    command = "DOWN"
                    should_move = True

          
            # RIGHT HAND: Gripper & Rotation
          
            elif hand_label == "Right":
                # Open Hand -> Open Gripper
                if fingers == [1, 1, 1, 1, 1]:
                    gripper_state = "OPEN"
                    if robot: robot.set_gripper_pos(850) # wrapper function from xarm_class_joint_space.py
                    
                # Fist -> Close Gripper
                elif fingers == [0, 0, 0, 0, 0]:
                    gripper_state = "CLOSE"
                    if robot: robot.set_gripper_pos(0)  # wrapper function from xarm_class_joint_space.py
                    
                # Index Finger -> Rotate Gripper Right (+Roll)
                elif fingers == [0, 1, 0, 0, 0]:
                    curr_yaw  += yaw_step
                    gripper_state = "ROT RIGHT"
                    should_move = True
                    
                # Peace Sign -> Rotate Gripper Left (-Roll)
                elif fingers == [0, 1, 1, 0, 0]:
                    curr_yaw  -= yaw_step
                    gripper_state = "ROT LEFT"
                    should_move = True

    # Safe Box (X , Y & Z) and Rotation Limits 
    if curr_x > 600: curr_x = 600                                                 # Max.
    if curr_x < 200: curr_x = 200                                                 # Min.
    if curr_y > 300: curr_y = 300                                                 # Max.
    if curr_y < -300: curr_y = -300                                               # Min.
    if curr_z > 500: curr_z = 500                                                 # Max.
    if curr_z < 100: curr_z = 100                                                 # Min.
    if curr_roll > initial_roll + 90: curr_roll = initial_roll + 90               # Max.
    if curr_roll < initial_roll - 90: curr_roll = initial_roll - 90               # Min.
    
    # Execute Motion 
    if robot and should_move:
        try:
            # Passes all 6 parameters to the wrapper
          robot.set_position(x=curr_x, y=curr_y, z=curr_z, roll=curr_roll, pitch=curr_pitch, yaw=curr_yaw, wait=False)
        except: pass

    # UI Dashboard 
    cv2.putText(image, f"Arm: {command}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(image, f"Gripper: {gripper_state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imshow("Robot Control Demo", image)
    
    # Exit Routine 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Loop Timer
    end_time = time.time()
    if end_time-start_timer<dt:
         time.sleep(dt-(end_time-start_timer))

# Cleanup
cap.release()
cv2.destroyAllWindows()