import cv2
import os
import numpy as np

# CONFIGURATION 
dataset_path = 'gesture_data_final'
gestures = ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_UP', 'MOVE_DOWN', 
            'MOVE_LEFT', 'MOVE_RIGHT', 'GRIPPER_OPEN', 'GRIPPER_CLOSED', 'SAFTEY']

print("Starting Brightness & Darkness Augmentation...")

# AUGMENTATION FUNCTION 
def adjust_brightness(image, value):
    # Convert to HSV to adjust brightness (V channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Add/Subtract value and clip between 0-255
    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = abs(value)
        v[v < value] = 0
        v[v >= value] -= value

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# PROCESSING LOOP 
for gesture in gestures:
    class_path = os.path.join(dataset_path, gesture)
    if not os.path.exists(class_path): continue
    
    images = [f for f in os.listdir(class_path) if f.endswith('.jpg') and 'aug' not in f]
    print(f"Processing {len(images)} images for {gesture}...")

    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)
        
        if image is None: continue

        # A. Create Brighter Version
        bright_img = adjust_brightness(image, 50) # Increase brightness by 50
        cv2.imwrite(os.path.join(class_path, f"aug_bright_{img_name}"), bright_img)

        # B. Create Darker Version
        dark_img = adjust_brightness(image, -50) # Decrease brightness by 50
        cv2.imwrite(os.path.join(class_path, f"aug_dark_{img_name}"), dark_img)


