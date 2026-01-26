from xarm.wrapper import XArmAPI
import numpy as np
import time

# --- SAFETY FIX: Try to import IK_Solver, but don't crash if missing ---
try:
    from myLibs.kinematic.ik_solver import IK_Solver
    HAS_IK = True
except ImportError:
    print(">>> WARNING: IK_Solver not found. Running in basic mode.")
    HAS_IK = False

init_pose = np.array([-0.02069, -0.9796, -0.0085, 0.6497, 0.0185, 1.630, -0.0007])

class xArm7:
    def __init__(self, ip, gripper_g2=False):
        self.gripper_g2 = gripper_g2
        self.arm = XArmAPI(ip)
        self.ip = ip
        self.arm.motion_enable(enable=True)
        self.arm.set_collision_tool_model(tool_type=1)
        self.arm.set_gripper_enable(enable=True)
        self.status_code, self.position = self.arm.get_position()
        self.gripper_pos = self.get_gripper_pos()
        self.previous_gripper_pos = self.get_gripper_pos()
        self.gripper_timer = 0
        self.gripper_moving = True
        
        # Initialize IK Solver only if we found the library
        if HAS_IK:
            self.IK_Solver = IK_Solver()
        else:
            self.IK_Solver = None

    def start_up(self):
        self.is_error()
        self.position_mode()
        # Move to home position safely
        self.arm.set_servo_angle(angle=init_pose, speed=0.35, mvacc=10, wait=True, is_radian=True)
        self.servo_mode()
        self.set_gripper_pos(840)

    def get_states(self) -> np.ndarray:
        self.status, position = self.arm.get_position()
        self.position = position
        return np.array(position)

    def set_gripper_pos(self, pos: float) -> None:
        if self.gripper_g2:
            pos_g2 = pos / 10
            self.arm.set_gripper_g2_position(pos=pos_g2, speed=200)
        else:
            self.arm.set_gripper_position(pos=pos, speed=1000)

    def get_gripper_pos(self) -> float:
        gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos[1]
        return gripper_pos[1]

    def position_mode(self) -> None:
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def servo_mode(self) -> None:
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)

    def is_error(self) -> bool:
        code_robo, [error_code_robo, warn_code] = self.arm.get_err_warn_code()
        code_gripper, error_code_gripper = self.arm.get_gripper_err_code()
        if error_code_robo != 0 or error_code_gripper != 0:
            return True
        return False

    def destroy(self) -> None:
        self.arm.set_state(state=4)
        self.arm.disconnect()