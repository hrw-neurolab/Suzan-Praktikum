"""
Praktikum Thesis: Robot Control System
Test 2: Cartesian Kinematics Verification 
Description: Manual control of xArm7 using Keyboard (X, Y, Z axes).
Suzan Elmasry
11.Feb.2026
"""

import cv2
import sys
import time

# --- 1. SETUP ---
sys.path.append('./') 


try:
    from myLibs.ufactory.xarm_class_joint_space import xArm7
except ImportError:
    print(">>> ERROR: Could not import xArm class. Check 'myLibs' folder.")
    xArm7 = None

ROBOT_IP = "10.2.134.151"
STEP_SIZE = 10  # 10 mm per step

def main():
    # --- 2. CONNECTION ---
    print(f">>> Connecting to Real Robot ({ROBOT_IP})...")
    
    if xArm7 is None:
        print(">>> FATAL ERROR: Library not found. Cannot proceed.")
        return

    try:
        arm = xArm7(ROBOT_IP)
        arm.start_up()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        print(">>> SUCCESS: Robot Connected & Ready.")
    except Exception as e:
        
        print(f">>> FATAL ERROR: Could not connect to robot. Check cables.\n>>> DETAILS: {e}")
        print(">>> NOTE: This test REQUIRES the real robot. It will not work at home.")
        return

    # --- 3. UI SETUP ---
    print("\n" + "="*30)
    print("   [ CONTROL PANEL ]")
    print("   W/S : X-Axis (Forward/Back)")
    print("   A/D : Y-Axis (Left/Right)")
    print("   Q/E : Z-Axis (Up/Down)")
    print("   ESC : Quit")
    print("="*30 + "\n")

    cv2.namedWindow("Robot_Controller")

    # --- 4. MAIN LOOP ---
    while True:
        # A. Get Real Position
        try:
            pos = arm.get_position()
        except:
            pos = None
        
        # Safety check
        if pos is None or len(pos) < 3: 
            current_x, current_y, current_z = 0, 0, 0
            status_msg = "Reading Error"
        else:
            current_x, current_y, current_z = pos[0], pos[1], pos[2]
            status_msg = "OK"

        # B. Wait for Command
        print(f"\r[STATUS: {status_msg}] Pos: X={int(current_x)}, Y={int(current_y)}, Z={int(current_z)}  ", end="")
        key = cv2.waitKey(0)

        # C. Calculate Target
        target_x, target_y, target_z = current_x, current_y, current_z

        if key == ord('w'): target_x += STEP_SIZE
        elif key == ord('s'): target_x -= STEP_SIZE
        elif key == ord('a'): target_y += STEP_SIZE
        elif key == ord('d'): target_y -= STEP_SIZE
        elif key == ord('q'): target_z += STEP_SIZE
        elif key == ord('e'): target_z -= STEP_SIZE
        elif key == 27: # ESC
            print("\n>>> Exiting...")
            break
        else:
            continue

        # D. Execute
        print(f"\n>>> MOVING to: {target_x:.1f}, {target_y:.1f}, {target_z:.1f}")
        try:
            arm.set_position(x=target_x, y=target_y, z=target_z, wait=True)
        except Exception as e:
            print(f"Move Error: {e}")

    # --- 5. SHUTDOWN ---
    try:
        arm.disconnect()
    except:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()