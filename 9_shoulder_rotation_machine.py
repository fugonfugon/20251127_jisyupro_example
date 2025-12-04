#!/usr/bin/env python3
"""
9_shoulder_rotation_machine.py

Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import time
import logging
import traceback

import numpy as np
import math

from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#position config
TRACKER_CFG = {
    "init": {
        "shoulder_pan_deg": 0.0, 
        "x": 0.1329,            
        "y": 0.0831,              
    },
}


# Joint calibration coefficients - manually edit
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # Joint1: zero position offset, scale factor
    ['shoulder_lift', 2.0, 0.97],     # Joint2: zero position offset, scale factor
    ['elbow_flex', 0.0, 1.05],        # Joint3: zero position offset, scale factor
    ['wrist_flex', 0.0, 0.94],        # Joint4: zero position offset, scale factor
    ['wrist_roll', 0.0, 0.5],        # Joint5: zero position offset, scale factor
    ['gripper', 0.0, 1.0],           # Joint6: zero position offset, scale factor
]

def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients
    
    Args:
        joint_name: Joint name
        raw_position: Raw position value
    
    Returns:
        calibrated_position: Calibrated position value
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # Zero position offset
            scale = joint_cal[2]   # Scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # If calibration coefficients not found, return raw value


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position
    
    Args:
        robot: Robot instance
        duration: Time required to move to zero position (seconds)
        kp: Proportional gain
    """
    print("Using P control to slowly move robot to zero position...")
    
    # Get current robot state
    current_obs = robot.get_observation()
    
    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value
    
    # Zero position targets
    zero_positions = {
        'shoulder_pan': 0.0, #外100, 内-95
        'shoulder_lift': 0.0, #上-100, 下100
        'elbow_flex': 0.0,  #上-100, 下100
        'wrist_flex': 0.0, #上-100, 下100
        'wrist_roll': 0.0,#-80右回転, 150左回転　単位度
        'gripper': 0.0 #閉じ0, 開き100
    }
    
    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will move to zero position in {duration} seconds using P control, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value
        
        # P control calculation
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                
                # P control: output = Kp * error
                control_output = kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)
        
        # Display progress
        if step % (control_freq // 2) == 0:  # Display progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to zero position progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("Robot has moved to zero position")

def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    2リンク平面アームの逆運動学（肩=joint2, 肘=joint3に対応）
    ------------------------------------------------------------
    目的：
      エンドエフェクタ目標座標 (x, y) から、肩・肘の目標角（URDF定義系）を計算する

    入力：
      x, y (float) : エンドエフェクタ先端の目標位置 [m]
      l1, l2 (float): 上腕・前腕のリンク長 [m]

    出力：
      joint2_deg, joint3_deg (float, float): 肩・肘の目標角 [deg]
                                             （URDFの制限内にクリップ済み）
    注意：
      - 実機の取付角のズレを theta1_offset/theta2_offset で補正
      - 到達不能点は最近傍の作業空間境界に丸める（スケール）
      - 角度は最終的に度（deg）で返却（以降の内部表現に合わせる）
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0
    
    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # Maximum reachable distance
    
    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
    
    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
    
    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)
    
    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    
    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset
    
    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    
    # Convert from radians to degrees
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90-joint2_deg
    joint3_deg = joint3_deg-90
    
    return joint2_deg, joint3_deg



def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position
    
    Args:
        robot: Robot instance
        start_positions: Start joint positions dictionary
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    print("Returning to start position...")
    
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds
    
    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # Don't apply calibration coefficients
        
        # P control calculation
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)
                
                # P control: output = Kp * error
                control_output = kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)
        
        # Check if start position is reached
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break
        
        time.sleep(control_period)
    
    print("Return to start position completed")

"""
def p_control_loop(robot, keyboard, target_positions, start_positions, kp=0.5, control_freq=50):
    
    #P control loop
    
    #Args:
    #    robot: Robot instance
    #    keyboard: Keyboard instance
    #    target_positions: Target joint positions dictionary
    #    start_positions: Start joint positions dictionary
    #    kp: Proportional gain
    #    control_freq: Control frequency (Hz)
    
    control_period = 1.0 / control_freq
    
    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    while True:
        try:
            # Get keyboard input
            keyboard_action = keyboard.get_action()
            
            if keyboard_action:
                # Process keyboard input, update target positions
                for key, value in keyboard_action.items():
                    if key == 'x':
                        # Exit program, first return to start position
                        print("Exit command detected, returning to start position...")
                        return_to_start_position(robot, start_positions, 0.2, control_freq)
                        return
                    
                    # Joint control mapping
                    joint_controls = {
                        'q': ('shoulder_pan', -1),    # Joint1 decrease
                        'a': ('shoulder_pan', 1),     # Joint1 increase
                        'w': ('shoulder_lift', -1),   # Joint2 decrease
                        's': ('shoulder_lift', 1),    # Joint2 increase
                        'e': ('elbow_flex', -1),      # Joint3 decrease
                        'd': ('elbow_flex', 1),       # Joint3 increase
                        'r': ('wrist_flex', -1),      # Joint4 decrease
                        'f': ('wrist_flex', 1),       # Joint4 increase
                        't': ('wrist_roll', -1),      # Joint5 decrease
                        'g': ('wrist_roll', 1),       # Joint5 increase
                        'y': ('gripper', -1),         # Joint6 decrease
                        'h': ('gripper', 1),          # Joint6 increase
                    }
                    
                    if key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"Updated target position {joint_name}: {current_target} -> {new_target}")
            
            # Get current robot state
            current_obs = robot.get_observation()
            
            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # Apply calibration coefficients
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value
            
            # P control calculation
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    
                    # P control: output = Kp * error
                    control_output = kp * error
                    
                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # Send action to robot
            if robot_action:
                robot.send_action(robot_action)
            
            time.sleep(control_period)
            
        except KeyboardInterrupt:
            print("User interrupted program")
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break
"""


"""
def sweep_pan(current_deg, span_deg, speed_deg_s, dt):
    #探索時に肩を左右に振るための「単純スイープ」。
    #- current_deg: 現在の肩角度（deg）
    #- span_deg   : 総振幅（例：100 -> -50°〜+50°）
    #- speed_deg_s: 回す速度（deg/s）
    #- dt         : 経過時間（s）

    # 振り子のように往復させるため、角度の進行方向を保持
    if not hasattr(sweep_pan, "_dir"):
        sweep_pan._dir = 1  # 1:正方向, -1:反対方向
    if not hasattr(sweep_pan, "_center"):
        sweep_pan._center = 0.0

    half = span_deg / 2.0
    next_deg = current_deg + sweep_pan._dir * speed_deg_s * dt
    if next_deg > sweep_pan._center + half:
        next_deg = sweep_pan._center + half
        sweep_pan._dir = -1
    elif next_deg < sweep_pan._center - half:
        next_deg = sweep_pan._center - half
        sweep_pan._dir = 1
    return next_deg
"""

# new function
def move_to_initial_position(robot, duration=3.0, kp=0.5, cfg=TRACKER_CFG):
    #Args:
    #robot: Robot instance
    #duration: Time required to move to initial position (seconds)
    #kp: Proportional gain
    print("Change initial set")
    
    # Get current robot state
    current_obs = robot.get_observation()

    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value


    pan_deg = cfg["init"]["shoulder_pan_deg"]
    x = cfg["init"]["x"] 
    y = cfg["init"]["y"] 

    #ikでinitial positionの計算.
    j2, j3 = inverse_kinematics(x, y)


    # initial positon targets
    initial_positions = {
        'shoulder_pan': pan_deg, #外100, 内-95
        'shoulder_lift': j2, #上-100, 下100
        'elbow_flex': j3,  #上-100, 下100
        'wrist_flex': -j2-j3, #上-100, 下100
        'wrist_roll': 0.0,#-80右回転, 150左回転　単位度
        'gripper': 0.0 #閉じ0, 開き100
    }

    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will move to initial position in {duration} seconds using P control, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value
        
        # P control calculation
        robot_action = {}
        for joint_name, target_pos in initial_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                
                # P control: output = Kp * error
                control_output = kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)
        
        # Display progress
        if step % (control_freq // 2) == 0:  # Display progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to initial position progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("Robot has moved to initial position")


# new func
def move_single_joint(robot, joint_name, target_deg, duration=3.0, kp=0.5):
    print(f"Moving {joint_name} → {target_deg} deg (keyboard-style control)")

    control_freq = 50
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    # 初期状態取得 → 他軸はこの値で設定
    current_obs = robot.get_observation()
    target_positions = {}
    
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            calibrated = apply_joint_calibration(motor_name, value)
            target_positions[motor_name] =0.0 #calibrated

    # 動かす軸だけ目標値更新
    target_positions[joint_name] = target_deg

    for step in range(total_steps):
        current_obs = robot.get_observation()
        current_positions = {}

        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                calibrated = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated

        robot_action = {}

        # 全軸をP制御
        # forの中はmotor_nameか
        for motor_name, target_pos in target_positions.items():
            current_pos = current_positions[motor_name]
            error = target_pos - current_pos
            control_output = kp * error
            new_position = current_pos + control_output
            robot_action[f"{motor_name}.pos"] = new_position

        if robot_action:
            robot.send_action(robot_action)
        time.sleep(step_time)

    print(f"{joint_name} reached {target_deg}° (approx.)")


def move_single_joint_2(robot, joint_name, target_deg, duration=3.0, kp=0.5):
    control_freq = 50
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    current_obs = robot.get_observation()
    current_key = f"{joint_name}.pos"
    if current_key not in current_obs:
        print(f"Error: Joint '{joint_name}' not found.")
        return

    start_deg = apply_joint_calibration(joint_name, current_obs[current_key])

    for step in range(total_steps):
        t = step / total_steps
        interp_deg = start_deg + (target_deg - start_deg) * t
        robot.send_action({current_key: interp_deg})
        time.sleep(step_time)

    robot.send_action({current_key: target_deg})
    print(f"[DONE] {joint_name} reached {target_deg}° directly.")


# enter押されるの待ち関数
def wait_enter(msg="Press ENTER to continue..."):
    input(msg)



def main():
    print("LeRobot Simplified Keyboard Control Example (P Control)")
    print("="*50)
    
    try:
        # Import necessary modules
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
        
        # Get port
        port = input("Please enter SO100 robot USB port (e.g.: /dev/ttyACM0): ").strip()
        
        # If directly press enter, use default port
        if not port:
            port = "/dev/ttyACM0"
            print(f"Using default port: {port}")
        else:
            print(f"Connecting to port: {port}")
        
        # Configure robot
        robot_config = SO100FollowerConfig(port=port)
        robot = SO100Follower(robot_config)
        
        # Configure keyboard
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)
        
        # Connect devices
        robot.connect()
        keyboard.connect()
        
        print("Devices connected successfully!")
        
        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")
        
        # Read starting joint angles
        print("Reading starting joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)  # Don't apply calibration coefficients
        
        print("Starting joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")
        
        # Move to zero position
        move_to_zero_position(robot, duration=3.0)
        
        # Initialize target positions to current positions (integers)
        """
        target_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
          }
        """
        

    # ここまで初期設定   
     
    #ここで一連の行動指南 
        #wait_enter("ZERO position completed. Press ENTER to move to INITIAL position...")
        #move_to_initial_position(robot, duration=3.0, kp=0.5)


        # motion 1
        """
        wait_enter("Initial position completed. Press ENTER to move shoulder_pan = +... ...")
        move_single_joint(robot, "shoulder_pan", 95, duration=3.0, kp=0.5)
        wait_enter("Shoulder_pan +95 done. Press ENTER to move shoulder_pan = -... ...")
        # 早くしたいならduration（s）を調整.
        move_single_joint(robot, "shoulder_pan", -30, duration=3.0, kp=1.0)
        """
        """
        # motion 2
        wait_enter("Initial position completed. Press ENTER to move shoulder_pan = +... ...")
        move_single_joint(robot, "shoulder_pan", 95, duration=3.0, kp=0.5)
        wait_enter("Shoulder_pan +95 done. Press ENTER to move shoulder_pan = -... ...")
        # 早くしたいならduration（s）を調整.
        move_single_joint(robot, "shoulder_pan", -30, duration=2.0, kp=1.0)
        """
        """
        # motion 3
        wait_enter("Initial position completed. Press ENTER to move shoulder_pan = +... ...")
        move_single_joint(robot, "shoulder_pan", 95, duration=3.0, kp=0.5)
        wait_enter("Shoulder_pan +95 done. Press ENTER to move shoulder_pan = -... ...")
        # 早くしたいならduration（s）を調整.
        move_single_joint(robot, "shoulder_pan", -30, duration=2.0, kp=1.25)
        """
        for i in range (10):
            wait_enter("Takeback is completed. please press ENTER to next action")
            robot.send_action({"shoulder_pan.pos": 70})
            wait_enter("Forehand is completed. please press ENTER to next action")
            robot.send_action({"shoulder_pan.pos": -30})





        
        wait_enter("Shoulder_pan -75 done. Press ENTER to DISCONNECT robot.")
        robot.disconnect()
        keyboard.disconnect()
        print("Robot power released. Program finished.")
        return


        """
        角度参考
        zero_positions = {
            'shoulder_pan': 0.0, #外100, 内-95
            'shoulder_lift': 0.0, #上-100, 下100
            'elbow_flex': 0.0,  #上-100, 下100
            'wrist_flex': 0.0, #上-100, 下100
            'wrist_roll': 0.0,#-80右回転, 150左回転 単位度
            'gripper': 0.0 #閉じ0, 開き100
        }
        """


        """
        print("Keyboard control instructions:")
        print("- Q/A: Joint1 (shoulder_pan) decrease/increase")
        print("- W/S: Joint2 (shoulder_lift) decrease/increase")
        print("- E/D: Joint3 (elbow_flex) decrease/increase")
        print("- R/F: Joint4 (wrist_flex) decrease/increase")
        print("- T/G: Joint5 (wrist_roll) decrease/increase")
        print("- Y/H: Joint6 (gripper) decrease/increase")
        print("- X: Exit program (first return to start position)")
        print("- ESC: Exit program")
        print("="*50)
        print("Note: Robot will continuously move to target position")
        """

        """
        # Start P control loop
        p_control_loop(robot, keyboard, target_positions, start_positions, kp=0.5, control_freq=50)
        """






        # Disconnect
        robot.disconnect()
        keyboard.disconnect()
        print("Program ended")
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Is the robot correctly connected")
        print("2. Is the USB port correct")
        print("3. Do you have sufficient permissions to access USB device")
        print("4. Is the robot correctly configured")

if __name__ == "__main__":
    main() 