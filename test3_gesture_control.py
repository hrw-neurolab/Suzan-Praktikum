"""
Praktikum Thesis: Robot Control System
Test 3: Gesture Control (Final Demo Version)
Description: 
    - Robot Control via Hand Gestures.
    - Integrated Gripper Control (Open/Close).
    - UI for demonstration.
    
Suzan Elmasry
18. Feb 2026
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
        robot.start_up()
        
        # Enable Gripper, speed min 1000, speed max 5000
        robot.arm.set_gripper_enable(True)
        robot.arm.set_gripper_speed(1000)
        
        print(">>> SUCCESS: Robot & Gripper Connected!")
    except:
        print(">>> ERROR: Robot connection failed")

# Vision Setup 
# Initialize the Hand Tracking Model
mp_hands = mp.solutions.hands
# We set max_num_hands=1 to ensure the robot listens to only ONE operator
hands = mp_hands.Hands(max_num_hands=1)
# visualize the hand skeleton (red lines) 
mp_draw = mp.solutions.drawing_utils
# Open the Camera
cap = cv2.VideoCapture(0)

# Initialize Robot Position 
# Start from a safe known point
curr_x = 300
curr_y = 0
curr_z = 200

if robot:
    try:
        # Get real robot position to avoid jumping
        pos = robot.get_position()
        if pos:
            curr_x, curr_y, curr_z = pos[0], pos[1], pos[2]
    except: pass
dt = 0.02
v= 100
step_size = v*dt

# Main Control Loop (Repeats every frame)
while cap.isOpened():
    success, image = cap.read()
    if not success: continue
    
    start_timer = time.time()
    # Mirror View
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    
    # We assume "STOP" at the start of every frame.
    # If no hand is found, or gesture is unknown, this stays "STOP".
    command = "STOP"
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw Skeleton on hand
            mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Finger Logic 
            # representing fingers (0=Closed, 1=Open)
            fingers = []
            
            # Thumb Logic (Checks X-axis)
            # Unlike other fingers, the thumb moves sideways.
            # We compare the X-coordinate of the Tip (4) vs. the IP Joint (3)
            if hand_lms.landmark[4].x < hand_lms.landmark[3].x:
                fingers.append(1) # Open
            else:
                fingers.append(0) # Closed
                
            # Other Fingers Logic (Checks Y-axis) (Vertical Movement)
            # For Index (8) Middle (12) Ring (16) Pinky (20).
            # These fingers move Up/Down so we check the Y-axis.
            # Y=0 is at the TOP of the screen.
            # if Tip_Y < Joint_Y, the finger is pointing UP.
            for id in [8, 12, 16, 20]:
                if hand_lms.landmark[id].y < hand_lms.landmark[id-2].y: # Compare Tip (id) with PIP Joint
                    fingers.append(1) # Open
                else:
                    fingers.append(0) # Closed
            
            # Gesture Mapping 
            
            # 1. Index -> Move Forward (+X)
            if fingers == [0, 1, 0, 0, 0]:
                curr_x += step_size 
                command = "FORWARD"
                
            # 2. Peace Sign -> Move Backward (-X)
            elif fingers == [0, 1, 1, 0, 0]:
                curr_x -= step_size 
                command = "BACKWARD"
                
            # 3. Pinky -> Move Right (-Y)
            elif fingers == [0, 0, 0, 0, 1]:
                curr_y -= step_size 
                command = "RIGHT"
                
            # 4. Thumb -> Move Left (+Y)
            elif fingers == [1, 0, 0, 0, 0]:
                curr_y += step_size 
                command = "LEFT"
                
            # 5. Open Hand -> Open Gripper
            elif fingers == [1, 1, 1, 1, 1]:
                command = "OPEN GRIPPER"
                if robot: robot.arm.set_gripper_position(850, wait=False)
                
            # 6. Fist -> Close Gripper
            elif fingers == [0, 0, 0, 0, 0]:
                command = "CLOSE GRIPPER"
                if robot: robot.arm.set_gripper_position(0, wait=False)

    # Safety Boundaries (The Safe Box) 
    if curr_x > 600: curr_x = 600
    if curr_x < 200: curr_x = 200
    
    if curr_y > 300: curr_y = 300
    if curr_y < -300: curr_y = -300
    
    # Execute Motion
    # Only send command IF it is NOT "STOP" and NOT a gripper command
    # It prevents the arm from moving if I am using the gripper or if I removed my hand
    # 1. robot: Is it connected?
    # 2. not GRIPPER: Dont move the arm (X/Y/Z) if we are just using the Gripper
    # 3. not STOP: Dont move if the hand is gone
    if robot and "GRIPPER" not in command and command != "STOP":
        try:
            robot.set_position(x=curr_x, y=curr_y, z=curr_z, wait=False)
        except: pass

    # UI Display 
    # Shows the command in RED text on the screen
    # It prints the active command on the screen so we know exactly what the robot is about to do
    cv2.putText(image, command, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("Robot Control Demo", image)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit
        break

    end_time = time.time() # loop timer 
    # if end_time-start_timer<dt:
    #     time.sleep(dt-(end_time-start_timer))


# Close all the windows and camera properly to exit the program 
cap.release()
cv2.destroyAllWindows()