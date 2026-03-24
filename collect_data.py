import cv2
import os

# SETUP CONFIGURATION ---
gestures = ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_UP', 'MOVE_DOWN', 
            'MOVE_LEFT', 'MOVE_RIGHT', 'GRIPPER_OPEN', 'GRIPPER_CLOSED', 'SAFTEY']

current_folder = os.getcwd()
dataset_path = os.path.join(current_folder, 'gesture_data_final')

# Create directory if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

# DATA COLLECTION LOOP ---
for gesture in gestures:
    class_path = os.path.join(dataset_path, gesture)
    os.makedirs(class_path, exist_ok=True)
    
    # Check how many images already exist in the folder
    existing_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
    count = len(existing_files)
    
    target_count = count + 60 # We want 60 more on top of what we have
    
    print(f"\n--- Current Class: {gesture} ---")
    print(f"Found {count} existing images. Collecting 60 more (Target: {target_count})")
    print("Press 'S' to Save, 'Q' to Skip this gesture")

    while count < target_count:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        # ROI box
        cv2.rectangle(frame, (100, 100), (350, 350), (255, 0, 0), 2)
        roi = frame[100:350, 100:350]

        # Display info
        status_text = f"Recording: {gesture} | Total: {count}/{target_count}"
        cv2.putText(frame, status_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Smart Data Collector (Add +60)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save using the current count as the index to avoid overwriting
            img_name = f"{gesture}_{count}.jpg"
            img_path = os.path.join(class_path, img_name)
            cv2.imwrite(img_path, roi)
            count += 1
            print(f"Saved: {img_name}")
            
        elif key == ord('q'):
            print(f"Skipping {gesture}...")
            break


cap.release()
cv2.destroyAllWindows()