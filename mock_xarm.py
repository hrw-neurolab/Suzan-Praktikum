import time
import numpy as np

class MockXArm:
    """
    A Simulation Class that mimics the real xArm 7
    """
    def __init__(self, ip):
        print(f"[MOCK] Connecting to Fake Robot at {ip}...")
        self.connected = True
        self.position = [300, 0, 200, 180, 0, 0] # Fake starting position

    def start_up(self):
        print("[MOCK] Running Safety Checks...")
        time.sleep(0.5)
        print("[MOCK] Motion Enabled.")
        print("[MOCK] Mode 0 (Position) Selected.")
        print("[MOCK] Moving to Home Position...")
        time.sleep(1)
        print("[MOCK] Robot Ready!")

    def get_states(self):
        # Return fake coordinates that wiggle slightly to look alive
        noise = np.random.uniform(-1, 1, 3) # Random noise +/- 1mm
        fake_pos = [300 + noise[0], 0 + noise[1], 200 + noise[2], 180, 0, 0]
        return np.array(fake_pos)

    def set_position(self, x=None, y=None, z=None, speed=None, wait=False):
        # Fake the movement command
        print(f"[MOCK] Moving to: X={x}, Y={y}, Z={z}")

    def destroy(self):
        print("[MOCK] Disconnected safely.")