#!/usr/bin/env python3
"""
このスクリプトの全体像（最初に読みましょう！）
================================================
目的：
  - キーボードで SO100/SO101 ロボットの関節目標角を操作し、P制御で追従させる
  - 別スレッドで YOLO による物体検出映像を表示する（※映像は制御に使わない）

設計方針：
  - 「ロボット制御」と「カメラ映像＋YOLO表示」を完全分離（独立スレッド）
  - 安全のため、関節角は P (比例) 制御でゆっくり追従
  - エンドエフェクタの x,y 操作 → 2リンク逆運動学で shoulder_lift / elbow_flex を自動算出
  - pitch（手首の屈曲）だけは補正量（加算項）として個別に調整

入出力（大枠）：
  - 入力：キーボード操作、カメラ画像
  - 出力：ロボットへの関節角コマンド、YOLO注釈付きの映像ウィンドウ

安全上の注意：
  - 実機では可動範囲・速度を守ること（URDFにある角度制限にはクリップ）
  - start姿勢に戻す処理あり（Xキー）
  - USBポート権限や校正ファイルの有無に注意
"""

import time
import logging
import traceback
import math
import cv2
import numpy as np
import threading
from ultralytics import YOLOE

# ------------------------------------------------------------
# ログ設定：開発/運用時に何が起きたかを追跡しやすくする
# ------------------------------------------------------------logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 関節のキャリブレーション係数
# 目的：センサの原点ズレやスケール誤差を補正して、制御を安定化
# 形式： [joint_name, zero_position_offset(deg), scale_factor]
# 使い方：センサ値を "適用後値 = (生値 - オフセット) * スケール" で補正する
# ------------------------------------------------------------
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
    関数の目的：
      指定関節の生データ角度にキャリブレーションを適用して補正値を返す

    入力：
      joint_name (str)     : 関節名（JOINT_CALIBRATIONにある名前と一致させる）
      raw_position (float) : センサ等から読み取った角度の生値（度）

    出力：
      calibrated_position (float) : 補正後の角度（度）
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # Zero position offset
            scale = joint_cal[2]   # Scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # If no calibration coefficient found, return raw value

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

def move_to_zero_position(robot, duration=3.0, kp=0.5):   
    """
    目的：
      すべての関節を「ゼロ姿勢（0度）」へ、P制御でゆっくり移動させる

    入力：
      robot   : SO100Follower のインスタンス（send_action/get_observation を持つ）
      duration (float): 目標到達にかける時間（秒）
      kp (float)      : Pゲイン（比例）

    出力/副作用：
      - ロボットへ連続的にコマンドを送信して、徐々にゼロ姿勢へ移動
      - 進捗をprintで表示
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
    
    # Zero position target
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    # Calculate control steps
    control_freq = 50  # 60Hz control frequency
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
        
        # Show progress
        if step % (control_freq // 2) == 0:  # Show progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to zero position progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("Robot has moved to zero position")

def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    目的：
      終了時などに「開始時に読み取った関節角」にロボットを戻す

    入力：
      robot : SO100Follower
      start_positions (dict[str->float]): 起動直後の各関節の角度（deg）
      kp (float)      : Pゲイン
      control_freq (int): 制御周波数 [Hz]

    出力/副作用：
      - 5秒を上限にP制御で戻す（総誤差が小さくなったら途中で終了）
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

# ------------------------------------------------------------
# カメラ映像＋YOLO表示（制御とは完全に独立）
#   - 引数 model : YOLOEモデル
#   - 引数 cap   : OpenCVのVideoCapture（選択したカメラ）
#   - 引数 target_objects : ユーザー入力で選んだ対象名のリスト
# 出力：ウィンドウ表示のみ（ロボット制御には不使用）
# ------------------------------------------------------------
def video_stream_loop(model, cap, target_objects=None):
    """
    目的：
      カメラからフレームを取得してYOLOで推論→注釈描画して表示
      （終了キー： 'q' または ESC）

    注意：
      - 失敗時は例外をcatchして表示を閉じる
      - この関数は別スレッドで動作し、制御ループと独立
    """
    print("Starting YOLO video stream...")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available")
                continue

            results = model(frame)
            if not results or not hasattr(results[0], 'boxes') or not results[0].boxes:
                # No objects detected - show original frame
                annotated_frame = frame
            else:
                # Show detection results
                annotated_frame = results[0].plot()
            
            # Show detection results in a window
            cv2.imshow("YOLO Live Detection", annotated_frame)
            
            # Allow quitting vision mode with 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
                
        except Exception as e:
            print(f"Video stream error: {e}")
            break
    
    print("Video stream ended")
    cv2.destroyAllWindows()

def p_control_loop(
    robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50
):
    """
    目的：
      キーボード入力を読み、目標角(target_positions)を更新し続ける。
      その目標に対してP制御で現在角を追従させるメインループ。

    入力：
      robot : SO100Follower
      keyboard : KeyboardTeleop
      target_positions (dict[str->float]) : 現在の目標角（deg）
      start_positions (dict[str->float])  : 起動時の角度（Xキーで戻す）
      current_x, current_y (float) : EEの現在目標位置 [m]
      kp (float)  : Pゲイン
      control_freq (int) : 制御周波数 [Hz]

    重要な内部変数：
      pitch : 手首屈曲の補正量（姿勢を保つための追加項）
    """
    control_period = 1.0 / control_freq

    # Initialize pitch control variables
    pitch = 0.0  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size

    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    while True:
        try:
            # Get keyboard input
            keyboard_action = keyboard.get_action()

            if keyboard_action:
                # Process keyboard input, update target positions
                for key, value in keyboard_action.items():
                    if key == "x":
                        # Exit program, first return to start position
                        print("Exit command detected, returning to start position...")
                        return_to_start_position(robot, start_positions, 0.2, control_freq)
                        return

                    # Joint control mapping
                    joint_controls = {
                        "q": ("shoulder_pan", -1),  # Joint 1 decrease
                        "a": ("shoulder_pan", 1),  # Joint 1 increase
                        "t": ("wrist_roll", -1),  # Joint 5 decrease
                        "g": ("wrist_roll", 1),  # Joint 5 increase
                        "y": ("gripper", -1),  # Joint 6 decrease
                        "h": ("gripper", 1),  # Joint 6 increase
                    }

                    # x,y coordinate control
                    xy_controls = {
                        "w": ("x", -0.004),  # x decrease
                        "s": ("x", 0.004),  # x increase
                        "e": ("y", -0.004),  # y decrease
                        "d": ("y", 0.004),  # y increase
                    }

                    # Pitch control
                    if key == "r":
                        pitch += pitch_step
                        print(f"Increase pitch adjustment: {pitch:.3f}")
                    elif key == "f":
                        pitch -= pitch_step
                        print(f"Decrease pitch adjustment: {pitch:.3f}")

                    if key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"Update target position {joint_name}: {current_target} -> {new_target}")

                    elif key in xy_controls:
                        coord, delta = xy_controls[key]
                        if coord == "x":
                            current_x += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions["shoulder_lift"] = joint2_target
                            target_positions["elbow_flex"] = joint3_target
                            print(
                                f"Update x coordinate: {current_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )
                        elif coord == "y":
                            current_y += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions["shoulder_lift"] = joint2_target
                            target_positions["elbow_flex"] = joint3_target
                            print(
                                f"Update y coordinate: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )

            # Apply pitch adjustment to wrist_flex
            # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
            if "shoulder_lift" in target_positions and "elbow_flex" in target_positions:
                target_positions["wrist_flex"] = (
                    -target_positions["shoulder_lift"] - target_positions["elbow_flex"] + pitch
                )
                # Show current pitch value (display every 100 steps to avoid screen flooding)
                if hasattr(p_control_loop, "step_counter"):
                    p_control_loop.step_counter += 1
                else:
                    p_control_loop.step_counter = 0

                if p_control_loop.step_counter % 100 == 0:
                    print(
                        f"Current pitch adjustment: {pitch:.3f}, wrist_flex target: {target_positions['wrist_flex']:.3f}"
                    )

            # Get current robot state
            current_obs = robot.get_observation()

            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
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

def main():
    """Main function"""
    print("LeRobot Keyboard Control + Independent YOLO Display")
    print("="*60)
    
    try:
        # Import necessary modules
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
        
        # Get port
        port = input("Please enter SO100 robot USB port (e.g.: /dev/ttyACM0): ").strip()
        
        # If Enter is pressed directly, use default port
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
        
        # Initialize target positions as current positions (integers)
        target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # Initialize x,y coordinate control
        x0, y0 = 0.1629, 0.1131
        current_x, current_y = x0, y0
        print(f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}")
        
        # Initialize YOLO and camera
        #model = YOLOE("yoloe-11l-seg.pt") 
        #yoloのサイズを変える 
        model = YOLOE("yoloe-11s-seg.pt") 
        
        
        # or select yoloe-11s/m-seg.pt for different sizes
        
        # Get detection targets from user input
        print("\n" + "="*60)
        print("YOLO Detection Target Setup")
        print("="*60)
        target_input = input("Enter objects to detect (separate multiple objects with commas, e.g., bottle,cup,mouse): ").strip()
        
        # If Enter is pressed directly, use default targets
        if not target_input:
            target_objects = ["bottle"]
            print(f"Using default targets: {target_objects}")
        else:
            # Parse multiple objects separated by commas
            target_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
            print(f"Detection targets: {target_objects}")
        
        # Set text prompt to detect the specified objects
        model.set_classes(target_objects, model.get_text_pe(target_objects))
        
        # List available cameras and prompt user
        def list_cameras(max_index=100):
            available = []
            for idx in range(max_index):
                cap_test = cv2.VideoCapture(idx)
                if cap_test.isOpened():
                    available.append(idx)
                    cap_test.release()
            return available
        cameras = list_cameras()
        if not cameras:
            print("No cameras found!")
            return
        print(f"Available cameras: {cameras}")
        selected = int(input(f"Select camera index from {cameras}: "))
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            print("Camera not found!")
            return
        
        print("Control instructions:")
        print("Keyboard control (independent of video stream):")
        print("- Q/A: Joint 1 (shoulder_pan) decrease/increase")
        print("- W/S: Control end effector x coordinate (joint2+3)")
        print("- E/D: Control end effector y coordinate (joint2+3)")
        print("- R/F: Pitch adjustment increase/decrease (affects wrist_flex)")
        print("- T/G: Joint 5 (wrist_roll) decrease/increase")
        print("- Y/H: Joint 6 (gripper) close/open")
        print("- X: Exit program (return to start position first)")
        print("- ESC: Exit program")
        print("")
        print("Video stream:")
        print("- Independent YOLO detection display (no robot control)")
        print("- Q (in YOLO window): Exit video stream")
        print("="*60)
        print("Note: Video stream and keyboard control are completely independent")
        
        # Start video stream in a separate thread
        video_thread = threading.Thread(target=video_stream_loop, args=(model, cap, target_objects), daemon=True)
        video_thread.start()
        
        # Start keyboard control loop (main thread)
        p_control_loop(robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50)
        
        # Disconnect
        robot.disconnect()
        keyboard.disconnect()
        cap.release()
        cv2.destroyAllWindows()
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