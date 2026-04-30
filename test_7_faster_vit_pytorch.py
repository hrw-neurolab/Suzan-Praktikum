import os
import sys
import time
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
from fastervit import create_model


# ROBOT CONNECTION

# Direct path to where the file is actually sitting
target_path = '/home/jonas/programming/suzan_praktikum_phase2/myLibs/ufactory'

if target_path not in sys.path:
    sys.path.append(target_path)

try:
    import xarm_class_joint_space
    from xarm_class_joint_space import xArm7
    print(" SUCCESS: xArm7 Class Loaded!")
except Exception as e:
    print(f" FAILED to load xArm7: {e}")
    xArm7 = None


# Robot Paramters

ROBOT_IP = "10.2.134.151"
DT = 0.01             
STEP_SIZE = 75 *DT
CONF_THRESHOLD = 0.75  

GRIPPER_OPEN_POS = 850   
GRIPPER_CLOSE_POS = 0    
GRIPPER_WAIT = False     

X_LIMITS = (200, 600)
Y_LIMITS = (-300, 300)
Z_LIMITS = (100, 500)
YAW_LIMITS = (-180, 180)


# MODEL INITIALIZATION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GESTURES = [
    'GRIPPER_CLOSED', 'GRIPPER_LEFT', 'GRIPPER_OPEN', 'GRIPPER_RIGHT',
    'MOVE_BACKWARD', 'MOVE_DOWN', 'MOVE_FORWARD', 'MOVE_RIGHT', 
    'MOVE_LEFT', 'MOVE_UP'
]

model = create_model('faster_vit_0_224')
model.head = nn.Linear(model.head.in_features, len(GESTURES))
model.load_state_dict(torch.load("xarm_gestures_fastervit.pth", map_location=device))
model.to(device).eval()

test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# HARDWARE INITIALIZATION

robot = None 

if xArm7 is not None:
    try:
        print(f"Attempting to connect to robot at {ROBOT_IP}...")
        #
        robot = xArm7(ROBOT_IP)
        # If the line above fails, it jumps straight to 'except'
        robot.start_up()
        print(">>> SUCCESS: ROBOT CONNECTED AND READY.")
    except Exception as e:
        print(f">>> ROBOT NOT FOUND: {e}")
        robot = None 
else:
    print(">>> WARNING: xArm7 Class was not imported. Simulation mode active.")

cap = cv2.VideoCapture(0)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

code, position =robot.arm.get_position()
curr_x, curr_y, curr_z = position[0], position[1], position[2]
curr_roll, curr_pitch, curr_yaw = -180, 0, -90
history = collections.deque(maxlen=10) 
last_gripper_state = None 


# MAIN LOOP

print(f"SYSTEM LIVE ON {device}. SEARCHING...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    start_timer = time.time()

 
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)        # converting BGR to LAB
    l, a, b_chan = cv2.split(lab)
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b_chan)), cv2.COLOR_LAB2BGR)

    # MODEL INFERENCE 
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    input_tensor = test_transforms(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
        idx = np.argmax(preds)
        conf = preds[idx]

    # --- FILTERING LOGIC (
    is_confused = (abs(preds[0] - preds[5]) < 0.15) and (conf < 0.85)  # If the model is "toss-up" between two specific gestures
                                                                       # this flag marks the current frame as unreliable
  
    if conf >= CONF_THRESHOLD and not (is_confused and (idx in [0, 5])):   #To save a gesture, the confidence must meet the 
                                                                           # CONF_THRESHOLD and the model must not be in the 
                                                                           # "confused" state for those specific indices
        history.append(idx)
    else:
        if len(history) > 0: history.popleft()  #If the prediction is weak or ambiguous, the oldest memory is removed

    command = "SEARCHING..."
    status_color = (0, 0, 255) # Default RED
    
    if len(history) >= 5:  # The system waits until it has at least 5 frames of data stored in the history memory
        counts = collections.Counter(history)   # This creates a frequency map
        stable_idx, count = counts.most_common(1)[0] # This identifies the Winner of the vote
        if count >= (len(history) * 0.7): # This is the Confidence Threshold It requires a 70% majority.
            command = GESTURES[stable_idx]
            status_color = (0, 255, 0) # ACTIVE GREEN

    # ROBOT CONTROL 
    should_move = False
    if command != "SEARCHING...":
        if command == 'MOVE_FORWARD':       curr_x += STEP_SIZE; should_move = True
        elif command == 'MOVE_BACKWARD':    curr_x -= STEP_SIZE; should_move = True
        elif command == 'MOVE_LEFT':        curr_y += STEP_SIZE; should_move = True
        elif command == 'MOVE_RIGHT':       curr_y -= STEP_SIZE; should_move = True
        elif command == 'MOVE_UP':          curr_z += STEP_SIZE; should_move = True
        elif command == 'MOVE_DOWN':        curr_z -= STEP_SIZE; should_move = True
        elif command == 'GRIPPER_LEFT':     curr_yaw += 5; should_move = True
        elif command == 'GRIPPER_RIGHT':    curr_yaw -= 5; should_move = True
        
        elif command == 'GRIPPER_OPEN' and last_gripper_state != 'OPEN':
            if robot: robot.set_gripper_pos(GRIPPER_OPEN_POS)
            last_gripper_state = 'OPEN'
        elif command == 'GRIPPER_CLOSED' and last_gripper_state != 'CLOSED':
            if robot: robot.set_gripper_pos(GRIPPER_CLOSE_POS)
            last_gripper_state = 'CLOSED'

    if should_move:
        curr_x = np.clip(curr_x, *X_LIMITS)
        curr_y = np.clip(curr_y, *Y_LIMITS)
        curr_z = np.clip(curr_z, *Z_LIMITS)
        curr_yaw = np.clip(curr_yaw, *YAW_LIMITS)


        robot.arm.set_servo_cartesian(
            [curr_x, curr_y, curr_z, curr_roll, curr_pitch, curr_yaw], is_radian=True
        )

    # UDI
    frame_display = cv2.flip(enhanced, 1)
    cv2.putText(frame_display, f"STATUS: {command}", (180, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2)

    cv2.imshow("xArm Live Control", frame_display)
    
    elapsed = time.time() - start_timer
    if elapsed < DT: time.sleep(DT - elapsed)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
if robot: robot.disconnect()