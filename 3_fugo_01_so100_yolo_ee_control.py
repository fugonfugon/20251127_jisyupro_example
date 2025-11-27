#!/usr/bin/env python3
"""
追跡のみ
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





# =========================
# 変更点実装
# =========================
from dataclasses import dataclass
import threading
import time

# TRACKER_CFG: 自動追跡に使う「調整パラメータ」を1か所に集めた辞書。
# - init：起動直後の「肩の回転角（deg）」と EE の (x,y) 初期値（m）
# - fov/gain/閾値/レート等：制御の強さや許容誤差、YOLOを何Hzで回すかなど
TRACKER_CFG = {
    "init": {
        "shoulder_pan_deg": 0.0,  # ←★あとで実測値に置き換えてください
        "x": 0.1329,              # ←★あとで実測値(m)に置き換えてください
        "y": 0.0831,              # ←★あとで実測値(m)に置き換えてください
    },

    # カメラの水平視野角（deg）。おおよそでOK。左右補正のスケール調整に使用。
    "fov_h_deg": 70.0,

    # 画像中心化の許容誤差（画面幅・高さ比）。これより小さくなれば「中央に来た」とみなす。
    "eps_u": 0.03,
    "eps_v": 0.03,
    "hold_N": 8,  # 収束状態を「連続N回」満たしたらAPPROACHへ遷移

    # 画像誤差 → 関節・座標への変換ゲイン（まずは小さく）
    "K_pan": 6.0,       # 肩回転  [deg / (正規化誤差)]
    "K_y": 0.06,        # y補正   [m   / (正規化誤差)]
    "K_forward": 0.04,  # 前進量   [m   / (面積不足)]

    # 前進の停止条件（寄りすぎ防止）。bbox面積の「画面比」目標、各種リミット。
    "rho_target": 0.80,            # bbox_area / (W*H)
    "x_limits": [0.12, 0.25],      # m
    "y_limits": [0.05, 0.17],      # m
    "pan_limits_deg": [-60, 60],   # deg（安全のため肩回転の範囲を固定）

    # 検出ロスト判定 & 探索動作（SEARCH状態）設定
    "conf_min": 0.4,
    "lost_timeout_s": 0.5,     # この秒数YOLO更新が無ければLOST扱い
    "search_speed_deg_s": 0.1,#20  # 探索時の肩回転の角速度（deg/s）
    "search_span_deg": 100,    # 探索の振り幅（±50°）

    # ループ周波数・YOLO軽量化（CPU向け）
    "control_hz": 60,          # 制御ループHz
    "yolo_hz_max": 10,         # YOLOは最大このHzでしか回さない（重さ対策）
    "smooth_alpha": 0.6,       # bbox中心の指数移動平均(EMA)の係数（なめらかに）
}

def clip(val, lo, hi):
    """数値を [lo, hi] にクリップする小道具（安全対策）"""
    return max(lo, min(hi, val))

def sweep_pan(current_deg, span_deg, speed_deg_s, dt):
    """
    探索時に肩を左右に振るための「単純スイープ」。
    - current_deg: 現在の肩角度（deg）
    - span_deg   : 総振幅（例：100 -> -50°〜+50°）
    - speed_deg_s: 回す速度（deg/s）
    - dt         : 経過時間（s）
    """
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

@dataclass
class Detection:
    """YOLOの検出結果を制御スレッドに渡すための軽量構造体"""
    found: bool
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    conf: float = 0.0
    W: int = 0
    H: int = 0
    t_update: float = 0.0  # 検出更新時刻（ロスト判定に使用）

class SharedState:
    """
    画像スレッド（YOLO）→制御スレッドへ「最新の検出1件」を渡すための共有メモリ。
    - スレッドセーフ：Lockで単純に守る
    - 常に「最新1件」を保持（古いものは上書き）
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._det = Detection(False)

    def update(self, det: Detection):
        with self._lock:
            self._det = det

    def read_latest(self) -> Detection:
        with self._lock:
            return self._det

# =========================
# ### 追加：YOLO 検出スレッド（CPU想定）
# =========================
def yolo_detection_loop(model, cap, target_objects, shared_state: SharedState, cfg=TRACKER_CFG, show_window=True):
    """
    目的：
      - カメラからフレーム取得 → YOLOで bottle を検出 → 「最新1件」を shared_state に格納
      - CPU向けに「間引き（yolo_hz_max）」＆「EMA平滑」＆「小さめ入力サイズ」などで軽量化
      - 必要に応じて imshow でオーバーレイ表示（性能が厳しければ show_window=False に）

    注意：
      - GPUは使わない（device='cpu' を強制）
      - set_classes で bottle のみ検出するよう制限（既存の指定を尊重）
    """
    print("[YOLO] start detection (CPU only). Heavy → throttled to", cfg["yolo_hz_max"], "Hz")

    # OpenCVのバッファを小さくして遅延を減らす（対応していない環境もあるため失敗してもOK）
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # YOLOEが device パラメタを直接受けるかは実装に依存するため、可能なら明示的にCPUへ。
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass

    # 推論間引きのためのタイマ
    infer_period = 1.0 / max(1, int(cfg["yolo_hz_max"]))
    last_infer = 0.0

    # EMA初期化フラグ
    ema_ready = False
    ema_cx = ema_cy = ema_w = ema_h = 0.0
    alpha = cfg["smooth_alpha"]

    # 画面表示（開くかはフラグ次第）
    win_name = "YOLO Live Detection (CPU)"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[YOLO] camera frame not available")
            time.sleep(0.01)
            continue

        H, W = frame.shape[:2]
        now = time.time()
        det = Detection(False, W=W, H=H, t_update=now)

        # 間引き：一定間隔より短ければ推論をスキップ（最新フレームだけ見続ける）
        if True: #now - last_infer >= infer_period:
            last_infer = now

            # 低解像度での推論（CPU向け最適化）
            # ※ Ultralytics実装差を吸収するため、極力デフォルトAPIで呼ぶ
            results = None
            try:
                results = model(
                    frame,               # 生フレーム
                    device="cpu",        # CPUを強制
                    #imgsz=640,
                    imgsz=320,    # k-okada 2025/11/26
                    #imgsz=1280,           # 解像度抑えめ（必要なら 512/416 に落とす）
                    verbose=False
                )
            except TypeError:
                # 一部実装では **kwargs を受けない場合があるためフォールバック
                results = model(frame)

            # 検出から bottle を1件選ぶ（最大信頼度）
            box = None
            if results and hasattr(results[0], "boxes") and results[0].boxes:
                # Ultralytics系の共通的な取出し方法（多少の差異を吸収）
                max_conf = -1.0
                for b in results[0].boxes:
                    try:
                        cls_id = int(b.cls[0])
                        conf = float(b.conf[0])
                        # set_classes で bottle のみのはずだが、念のため conf で選別
                        if conf > max_conf:
                            xywh = b.xywh[0].cpu().numpy()
                            cx_, cy_, w_, h_ = map(float, xywh)
                            box = (cx_, cy_, w_, h_, conf)
                            max_conf = conf
                    except Exception:
                        continue

            if box is not None:
                cx_, cy_, w_, h_, conf_ = box

                # EMA：前回値があれば滑らかに更新
                if not ema_ready:
                    ema_cx, ema_cy, ema_w, ema_h = cx_, cy_, w_, h_
                    ema_ready = True
                else:
                    ema_cx = alpha * cx_ + (1 - alpha) * ema_cx
                    ema_cy = alpha * cy_ + (1 - alpha) * ema_cy
                    ema_w  = alpha * w_  + (1 - alpha) * ema_w
                    ema_h  = alpha * h_  + (1 - alpha) * ema_h

                det = Detection(
                    found=True, cx=ema_cx, cy=ema_cy, w=ema_w, h=ema_h,
                    conf=conf_, W=W, H=H, t_update=now
                )
            else:
                # 見つからないときは found=False のまま
                pass

            # 共有状態に最新を保存（制御スレッドが読む）
            shared_state.update(det)

            # 画面に可視化（任意）
            
            if show_window:
                vis = frame.copy()
            
                cv2.drawMarker(vis, (W//2, H//2), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                if det.found:
                    
                    x1 = int(det.cx - det.w/2)
                    y1 = int(det.cy - det.h/2)
                    x2 = int(det.cx + det.w/2)
                    y2 = int(det.cy + det.h/2)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis, f"conf={det.conf:.2f}", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                cv2.imshow(win_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            

        else:
            # 推論待ちの間は軽く休む（CPUを無駄に回さない）
            time.sleep(0.002)

    if show_window:
        cv2.destroyWindow(win_name)
    print("[YOLO] detection loop ended")


# =========================
# ### 追加：自動追跡ループ（状態機械：SEARCH→TRACK→APPROACH）
# =========================
def auto_track_loop(
    robot, target_positions, start_positions, current_x, current_y, shared_state: SharedState, cfg=TRACKER_CFG
):
    """
    目的：
      - YOLOが出す「bottleの画像上の位置」を使い、肩回転＋(x,y) の2リンクIKで
        「常に中心へ寄せる（TRACK）」→「十分中央なら前進して近づく（APPROACH）」。
      - 見失ったら SEARCH 状態で肩をスイープし再捕捉。

    既存設計への影響：
      - 既存のP制御やIK関数はそのまま使用（最小変更）。
      - キーボード手動は停止し、自動のみ（必要なら併用版も後で追加可能）。

    安全：
      - 角度・座標は cfg の範囲でクリップ。
      - 非常停止は別途 X キー（既存仕様）で main が抜ける想定。
    """
    control_hz = cfg["control_hz"]
    dt = 1.0 / control_hz
    print(f"[CTRL] auto_track_loop start: {control_hz} Hz")

    # 初期姿勢の設定（あとで埋めやすいよう cfg.init を採用）
    pan_deg = cfg["init"]["shoulder_pan_deg"]
    x = cfg["init"]["x"] if current_x is None else current_x
    y = cfg["init"]["y"] if current_y is None else current_y

    # 目標角（deg）の辞書は既存と互換に更新
    target_positions["shoulder_pan"] = pan_deg
    j2, j3 = inverse_kinematics(x, y)
    target_positions["shoulder_lift"] = j2
    target_positions["elbow_flex"]   = j3
    target_positions["wrist_flex"]   = -j2 - j3  # ピッチ加算は必要に応じて別途

    # 状態機械
    state = "SEARCH"
    hold_count = 0

    # ループ本体
    try:
        while True:
            t0 = time.time()
            det = shared_state.read_latest()

            # ロスト判定（更新無し/信頼度不足）
            lost = (not det.found) or ((time.time() - det.t_update) > cfg["lost_timeout_s"]) or (det.conf < cfg["conf_min"])

            if state == "SEARCH":
                if not lost:
                    state = "TRACK"
                else:
                    # 肩をスイープして探索（deg/s * dt 分だけ動かす）
                    pan_deg = sweep_pan(
                        pan_deg,
                        cfg["search_span_deg"],
                        cfg["search_speed_deg_s"],
                        dt
                    )
                    pan_deg = clip(pan_deg, cfg["pan_limits_deg"][0], cfg["pan_limits_deg"][1])
                    target_positions["shoulder_pan"] = pan_deg

            else:
                if lost:
                    # 見失ったら探索へ戻る
                    state = "SEARCH"
                    hold_count = 0
                else:
                    # 画像中心に寄せるための誤差（正規化）
                    eu = (det.cx - det.W/2) / det.W   # 横ずれ（右＋）
                    ev = (det.cy - det.H/2) / det.H   # 縦ずれ（下＋）

                    # --- 横方向：肩回転で補正 ---
                    # 画像の横ずれを肩角度に変換して足し込む（FOVでスケール調整しても良い）
                    pan_deg += cfg["K_pan"] * eu # たぶん逆だったので修正
                    pan_deg = clip(pan_deg, cfg["pan_limits_deg"][0], cfg["pan_limits_deg"][1])
                    target_positions["shoulder_pan"] = pan_deg

                    # --- 縦方向：y座標で補正（2リンクIKで lift/elbow へ） ---
                    # 下にズレたら y を上げる（符号に注意：ここでは -ev で上方向へ）
                    y += cfg["K_y"] * (-ev)
                    y = clip(y, cfg["y_limits"][0], cfg["y_limits"][1])

                    # IKで関節角に変換
                    j2, j3 = inverse_kinematics(x, y)
                    target_positions["shoulder_lift"] = j2
                    target_positions["elbow_flex"]   = j3
                    target_positions["wrist_flex"]   = -j2 - j3  # 必要に応じてピッチ加算

                    # 収束判定（中心化できているか）
                    if abs(eu) < cfg["eps_u"] and abs(ev) < cfg["eps_v"]:
                        hold_count += 1
                    else:
                        hold_count = 0

                    # TRACK → 十分中央なら APPPROACH へ
                    if state == "TRACK":
                        if hold_count >= cfg["hold_N"]:
                            state = "APPROACH"

                    # APPPROACH：中央を維持しつつ「前へ」近づく
                    elif state == "APPROACH":
                        rho = (det.w * det.h) / float(det.W * det.H + 1e-9)  # 画面比（0〜1）
                        # 足りない分だけxを増やす（近づく）
                        dx = cfg["K_forward"] * (cfg["rho_target"] - rho)
                        if dx > 0:
                            x = clip(x + dx, cfg["x_limits"][0], cfg["x_limits"][1])

                            # x を動かしたので改めてIK反映
                            j2, j3 = inverse_kinematics(x, y)
                            target_positions["shoulder_lift"] = j2
                            target_positions["elbow_flex"]   = j3
                            target_positions["wrist_flex"]   = -j2 - j3

                        # 目標面積に届いた/上限まで来たら、APPROACH終了し TRACK に戻る
                        if (rho >= cfg["rho_target"]) or (x >= cfg["x_limits"][1] - 1e-6):
                            state = "TRACK"

            # --- P制御でロボットに反映（既存ロジックと同じ思想：現在→目標に寄せる） ---
            # 現在角の取得（既存関数を流用）
            current_obs = robot.get_observation()
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    # センサ値のキャリブレーションを適用（既存の補正関数を利用）
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value

            # P制御：目標へ少しずつ寄せる（急激な動きにならないよう kp は低め）
            kp = 0.5
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    control_output = kp * error
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position

            if robot_action:
                robot.send_action(robot_action)

            # ログ（1秒おき程度）
            if int(time.time()) % 1 == 0:
                print(f"[CTRL] state={state} pan={pan_deg:.1f} x={x:.3f} y={y:.3f}")

            # 制御周期の維持
            elapsed = time.time() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("[CTRL] interrupted")
    except Exception as e:
        print(f"[CTRL] error: {e}")
        traceback.print_exc()
    finally:
        print("[CTRL] auto_track_loop ended")









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
        
        # Initialize YOLO and camera yoloxモデル
        model = YOLOE("yoloe-11s-seg.pt")  # CPUが重ければ "yoloe-11s-seg.pt" 等の小型に変更推奨







        # ---- ここから変更：共有状態の作成＆検出スレッド起動 ----
        shared_state = SharedState()

        # set_classes は既存ロジックを尊重（bottle のみ）
        print("\n" + "="*60)
        print("YOLO Detection Target Setup")
        print("="*60)
        target_input = input("Enter objects to detect (e.g., bottle) [default: bottle]: ").strip()
        if not target_input:
            target_objects = ["bottle"]
            print(f"Using default targets: {target_objects}")
        else:
            target_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
            print(f"Detection targets: {target_objects}")
        model.set_classes(target_objects, model.get_text_pe(target_objects))

        # カメラ選択は既存と同じ
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
        
        # カメラ解像度設定変更（例：1280x720）
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #print(f"Camera resolution set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")


        # ---- 旧：video_stream_loop → 新：yolo_detection_loop（CPU軽量化＆共有状態に格納）----
        # 画面表示が重い場合は show_window=False にしてください。
        yolo_thread = threading.Thread(
            target=yolo_detection_loop, args=(model, cap, target_objects, shared_state), kwargs={"show_window": True}, daemon=True
            #target=yolo_detection_loop, args=(model, cap, target_objects, shared_state), kwargs={"show_window": False}, daemon=True
        )
        yolo_thread.start()

        # =========================
        # ★ここを差し替え★：手動P制御 → 自動追跡ループ
        # （元）p_control_loop(robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50)
        # （新）
        auto_track_loop(
            robot=robot,
            target_positions=target_positions,
            start_positions=start_positions,
            current_x=current_x,
            current_y=current_y,
            shared_state=shared_state,
            cfg=TRACKER_CFG
        )








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


"""
重いと感じたら：

TRACKER_CFG["yolo_hz_max"] を 8 → 6 → 4 と落とす

imgsz=640 を imgsz=512 や 416 に下げる（yolo_detection_loop 内）

モデルを yoloe-11s-seg.pt など小型へ

show_window=False にして描画を切る

中心化が振動するなら：

K_pan, K_y を下げる

eps_u/eps_v を少し緩める（0.03 → 0.04）

smooth_alpha を上げる（0.6 → 0.7 でより滑らか）

近づきすぎる/寄り足りない：

rho_target を上下させて調整

x_limits の上限を見直す
"""
