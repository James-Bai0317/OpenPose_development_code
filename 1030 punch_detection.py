# 單人的punch_detection.py - 專注於動作辨識，按run後在Pycharm上打開攝影機，可先按一次q即可重置攝影機，測試完後再按q可退出攝影機
import math
import time
import cv2
# import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import csv # 紀錄log檔日誌，以便整理實驗數據

# BODY_25身體關節對應，供相關程式邏輯對應編寫
NOSE = 0
NECK = 1
RIGHT_SHOULDER = 2
RIGHT_ELBOW = 3
RIGHT_WRIST = 4
LEFT_SHOULDER = 5
LEFT_ELBOW = 6
LEFT_WRIST = 7
MID_HIP = 8
RIGHT_HIP = 9
RIGHT_KNEE = 10
RIGHT_ANKLE = 11
LEFT_HIP = 12
LEFT_KNEE = 13
LEFT_ANKLE = 14
RIGHT_EYE = 15
LEFT_EYE = 16
RIGHT_EAR = 17
LEFT_EAR = 18
LEFT_BIG_TOE = 19
LEFT_SMALL_TOE = 20
LEFT_HEEL = 21
RIGHT_BIG_TOE = 22
RIGHT_SMALL_TOE = 23
RIGHT_HEEL = 24

# 引入 OpenPose API，往後根據你的資料夾位置進行調整
from pose_capture.openpose_api import get_keypoints_stream

# 引入angle.py

from angle import calculate_normalized_angle, calculate_shoulder_width
from angle import (NOSE, NECK, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
                       LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, MID_HIP,
                       RIGHT_HIP, LEFT_HIP, RIGHT_KNEE, LEFT_KNEE,
                       RIGHT_ANKLE, LEFT_ANKLE)

print("Successfully imported angle module functions")

# 引入guard_detection.py
from guard_detection import BoxingDefenseDetector, BoxingDefenseVisualizer

#以下定義參數資料型別

# 定義動作類型
class ActionType(Enum):
    IDLE = "idle"
    PUNCH_STRAIGHT = "straight_punch"
    PUNCH_HOOK = "hook_punch"
    PUNCH_UPPERCUT = "uppercut_punch"
    GUARD = "guard"
    DODGE = "dodge"

# 拳法配置的資料型別
@dataclass
class PunchConfig:
    name: str
    angle_threshold: float
    speed_threshold: float
    min_extension: float

# 統計玩家動作數據資料型別
@dataclass
class PlayerActionData:
    player_id: int
    action_type: str
    attack_hand: Optional[str]
    punch_type: Optional[str]
    velocity: float
    confidence: float
    arm_angles: Dict[str, Optional[float]]
    shoulder_width: Optional[float]
    body_center: Optional[Tuple[float, float]]
    is_attacking: bool
    is_guarding: bool
    timestamp: float

# 單幀數據
@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    players: List[PlayerActionData]
    players_distance: Optional[float]


class BoxingActionDetector:
    """拳擊動作檢測器相關參數打包"""

    def __init__(self):
        self.frame_count = 0

        # 拳法配置
        self.punch_configs = {
            "straight": PunchConfig("直拳", 120, 0.02, 70),
            "hook": PunchConfig("勾拳", 80, 0.03, 70),
            "uppercut": PunchConfig("上勾拳", 50, 0.02, 70)
        }

        # 位置歷史記錄 (用於計算速度)
        self.position_history = {}

        # 玩家動作冷卻
        self.player_cooldowns = {}
        self.cooldown_frames = 10

        # 初始化拳法驗證策略
        self.validation_strategies = {
            "straight": self._validate_straight_punch,
            "hook": self._validate_hook_punch,
            "uppercut": self._validate_uppercut_punch
        }

        print("=== Boxing Action Detector Initialized ===")

        # 防禦檢測器
        self.defense_detector = BoxingDefenseDetector()
        print("Defense detector initialized")

    def detect_actions(self, keypoints) -> FrameData:
        """檢測所有玩家的動作"""
        self.frame_count += 1
        timestamp = time.time() # 用來看幾frame為一秒

        players_data = []
        players_distance = None

        if keypoints is not None and len(keypoints) > 0:
            # 確保最多處理兩個人
            num_persons = min(len(keypoints), 2)

            for person_id in range(num_persons):
                player_data = self._detect_player_action(person_id, keypoints, timestamp)
                if player_data:
                    players_data.append(player_data)

                # 更新位置歷史
                self._update_position_history(person_id, keypoints)

            # 計算玩家間距離
            if len(players_data) >= 2:
                players_distance = self._calculate_players_distance(keypoints)

        return FrameData(
            frame_id=self.frame_count,
            timestamp=timestamp,
            players=players_data,
            players_distance=players_distance
        )

    def _detect_player_action(self, person_id: int, keypoints, timestamp: float) -> Optional[PlayerActionData]:
        """檢測單個玩家的動作"""
        try:
            person_keypoints = keypoints[person_id]

            # 檢查動作冷卻
            if person_id in self.player_cooldowns:
                if self.frame_count - self.player_cooldowns[person_id] < self.cooldown_frames:
                    return None

            # 計算手臂角度
            right_arm_angle = calculate_normalized_angle(
                keypoints, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], person_id)
            left_arm_angle = calculate_normalized_angle(
                keypoints, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], person_id)

            # 計算肩寬
            shoulder_width = calculate_shoulder_width(keypoints, person_id)

            # 獲取身體中心
            body_center = self._get_body_center(person_keypoints)

            # 初始化動作數據
            action_data = PlayerActionData(
                player_id=person_id,
                action_type=ActionType.IDLE.value,
                attack_hand=None,
                punch_type=None,
                velocity=0.0,
                confidence=0.0,
                arm_angles={"right": right_arm_angle, "left": left_arm_angle},
                shoulder_width=shoulder_width,
                body_center=body_center,
                is_attacking=False,
                is_guarding=False,
                timestamp=timestamp
            )


            punch_result = self._detect_punch_action(person_id, keypoints, right_arm_angle, left_arm_angle)

            if punch_result:
                action_data.action_type = punch_result["action_type"]
                action_data.attack_hand = punch_result["hand"]
                action_data.punch_type = punch_result["punch_type"]
                action_data.velocity = punch_result["velocity"]
                action_data.confidence = punch_result["confidence"]
                action_data.is_attacking = True
                self.player_cooldowns[person_id] = self.frame_count

            return action_data

        except Exception as e:
            print(f"Error detecting player {person_id} action: {e}")
            return None

    def _detect_punch_action(self, person_id: int, keypoints, right_arm_angle, left_arm_angle) -> Optional[Dict]:
        """檢測拳擊動作"""
        # 檢查右手攻擊
        right_punch = self._analyze_punch_motion(person_id, keypoints, "right", right_arm_angle)
        if right_punch:
            return right_punch

        # 檢查左手攻擊
        left_punch = self._analyze_punch_motion(person_id, keypoints, "left", left_arm_angle)
        if left_punch:
            return left_punch

        return None

    def _analyze_punch_motion(self, person_id: int, keypoints, hand: str, arm_angle: float) -> Optional[Dict]:
        """分析拳擊動作類型，回傳包含置信度、速度與手臂資訊的字典"""
        if arm_angle is None:
            return None

        try:
            person_keypoints = keypoints[person_id]

            # 選擇手臂關鍵點索引
            if hand == "right":
                wrist_idx, elbow_idx, shoulder_idx = RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER
            else:
                wrist_idx, elbow_idx, shoulder_idx = LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER

            wrist = person_keypoints[wrist_idx]
            elbow = person_keypoints[elbow_idx]
            shoulder = person_keypoints[shoulder_idx]

            # 關鍵點置信度檢查
            if wrist[2] < 0.3 or elbow[2] < 0.3 or shoulder[2] < 0.3:
                return None

            # 計算手腕相對肩膀位置與手臂伸展距離
            wrist_x_offset = wrist[0] - shoulder[0]
            wrist_y_offset = wrist[1] - shoulder[1]
            arm_extension = math.sqrt(wrist_x_offset ** 2 + wrist_y_offset ** 2)

            # 計算手臂速度
            velocity = self._calculate_hand_velocity(person_id, hand)

            # 關鍵點平均置信度
            keypoint_conf = (wrist[2] + elbow[2] + shoulder[2]) / 3.0

            # 標準化手臂伸展度
            shoulder_width = calculate_shoulder_width(keypoints, person_id)
            if shoulder_width and shoulder_width > 0:
                normalized_extension = arm_extension / shoulder_width
                ext_score = 1.0 / (1.0 + math.exp(-(normalized_extension - 1.5) * 3))
            else:
                ext_score = min(arm_extension / 120.0, 1.0)

            # 速度分數
            velocity_threshold = 0.3
            vel_score = 1.0 / (1.0 + math.exp(-(velocity - velocity_threshold) * 8))

            # 計算角度評分
            angle_scores = self._calculate_angle_scores(arm_angle)

            # 基本條件檢查（降低門檻）
            min_extension = 40
            if arm_extension < min_extension or velocity < 0.01:
                return None

            # 選擇最佳拳法
            best_punch = None
            best_confidence = 0

            for punch_name, config in self.punch_configs.items():
                angle_score = angle_scores.get(punch_name, 0)

                # 如果角度分數過低，直接跳過
                if angle_score < 0.1:
                    continue

                # 使用策略模式驗證拳法
                validate_func = self.validation_strategies[punch_name]
                punch_valid, position_bonus = validate_func(
                    keypoints, person_id, hand, wrist_x_offset, wrist_y_offset, shoulder_width
                )

                if not punch_valid:
                    continue

                # 計算綜合置信度
                base_confidence = (
                        keypoint_conf * 0.15 +
                        ext_score * 0.20 +
                        vel_score * 0.20 +
                        angle_score * 0.30 +
                        position_bonus * 0.15
                )

                final_confidence = min(base_confidence, 1.0)

                # 降低最低置信度閾值
                if final_confidence > best_confidence and final_confidence > 0.35:
                    best_punch = {
                        "action_type": getattr(ActionType, f"PUNCH_{punch_name.upper()}").value,
                        "hand": hand,
                        "punch_type": punch_name,
                        "velocity": round(velocity, 3),
                        "confidence": round(final_confidence, 3),
                        "subscores": {
                            "keypoint_conf": round(keypoint_conf, 3),
                            "extension": round(ext_score, 3),
                            "velocity": round(vel_score, 3),
                            "angle_score": round(angle_score, 3),
                            "position_bonus": round(position_bonus, 3)
                        }
                    }
                    best_confidence = final_confidence

            return best_punch

        except (IndexError, TypeError) as e:
            print(f"Error in _analyze_punch_motion: {e}")
            return None

    @staticmethod
    def _calculate_angle_scores(arm_angle: float) -> Dict[str, float]:
        """計算各種拳法的角度評分"""
        scores = {}

        # 直拳：角度應該非常大（手臂非常伸直）
        straight_optimal = 140  # 理想角度
        straight_tolerance = 25  # 容忍度 (標準差的近似值，控制曲線寬度)
        # 計算角度差的平方，並除以容忍度的平方，作sigmoid平滑處理
        straight_diff_sq = ((arm_angle - straight_optimal) / straight_tolerance) ** 2
        scores["straight"] = math.exp(-straight_diff_sq)

        # 勾拳：角度中等
        hook_optimal = 90
        hook_tolerance = 30
        hook_diff_sq = ((arm_angle - hook_optimal) / hook_tolerance) ** 2
        scores["hook"] = math.exp(-hook_diff_sq)

        # 上勾拳：角度較小
        uppercut_optimal = 60
        uppercut_tolerance = 25
        uppercut_diff_sq = ((arm_angle - uppercut_optimal) / uppercut_tolerance) ** 2
        scores["uppercut"] = math.exp(-uppercut_diff_sq)

        return scores

    # 調整各種拳擊參數與邏輯的地方，從這裡更改
    @staticmethod
    def _validate_straight_punch(keypoints, person_id: int, hand: str,
                                 wrist_y_offset: float,
                                 shoulder_width: Optional[float]) -> Tuple[bool, float]:
        """驗證直拳的空間位置特徵"""
        person_keypoints = keypoints[person_id]

        # 獲取頸部參考點
        neck_keypoint = person_keypoints[NECK]
        if neck_keypoint[2] < 0.3:
            return False, 0.0

        wrist_idx = RIGHT_WRIST if hand == "right" else LEFT_WRIST
        shoulder_idx = RIGHT_SHOULDER if hand == "right" else LEFT_SHOULDER
        wrist = person_keypoints[wrist_idx]
        shoulder = person_keypoints[shoulder_idx]

        # 條件 1: 手腕必須在肩膀前方 (向前伸展)
        if hand == "right":
            if wrist[0] <= shoulder[0]:  # 右手應該向右前方
                return False, 0.0
        else:
            if wrist[0] >= shoulder[0]:  # 左手應該向左前方
                return False, 0.0

        # 條件 2: 手腕必須接近身體中線 (與頸部X座標對齊)
        if shoulder_width and shoulder_width > 0:
            neck_alignment = abs(wrist[0] - neck_keypoint[0])
            if neck_alignment > shoulder_width * 0.6:
                return False, 0.0

        # 條件 3: 垂直位置 - 手腕與肩膀高度接近
        if abs(wrist_y_offset) > shoulder_width * 0.25 if shoulder_width else 50:
            return False, 0.0


        # 位置加分: 基於與頸部的對齊程度
        if shoulder_width and shoulder_width > 0:
            alignment_score = 1.0 - min(1.0, abs(wrist[0] - neck_keypoint[0]) / (shoulder_width * 0.6))
            position_bonus = 0.4 * alignment_score
        else:
            position_bonus = 0.2

        return True, position_bonus

    @staticmethod
    def _validate_hook_punch(hand: str,
                             wrist_x_offset: float, wrist_y_offset: float,
                             shoulder_width: Optional[float]) -> Tuple[bool, float]:
        """驗證勾拳的空間位置特徵"""


        if shoulder_width is None or shoulder_width <= 0:
            shoulder_width = 100  # 預設值



        # 條件 1: 手腕必須在身體外側 (遠離中線)
        if hand == "right":
            if wrist_x_offset <= shoulder_width * 0.5:  # 右手應明顯在右側
                return False, 0.0
        else:
            if wrist_x_offset >= -shoulder_width * 0.5:  # 左手應明顯在左側
                return False, 0.0

        # 條件 2: 垂直位置 - 與肩膀高度接近
        if abs(wrist_y_offset) > shoulder_width * 0.25:
            return False, 0.0

        # 位置加分: 越在外側,加分越多
        lateral_distance = abs(wrist_x_offset)
        lateral_score = min(1.0, lateral_distance / (shoulder_width * 0.8))
        position_bonus = 0.3 * lateral_score

        return True, position_bonus

    @staticmethod
    def _validate_uppercut_punch(keypoints, person_id: int, hand: str,
                                 wrist_x_offset: float,
                                 shoulder_width: Optional[float]) -> Tuple[bool, float]:
        """驗證上勾拳的空間位置特徵"""
        person_keypoints = keypoints[person_id]

        if shoulder_width is None or shoulder_width <= 0:
            shoulder_width = 100

        wrist_idx = RIGHT_WRIST if hand == "right" else LEFT_WRIST
        elbow_idx = RIGHT_ELBOW if hand == "right" else LEFT_ELBOW
        wrist = person_keypoints[wrist_idx]
        elbow = person_keypoints[elbow_idx]

        # 置信度檢查
        if wrist[2] < 0.3 or elbow[2] < 0.3:
            return False, 0.0

        # 條件 1: 手腕必須高於手肘 (Y座標更小) - 上勾拳的核心特徵
        if wrist[1] >= elbow[1]:
            return False, 0.0

        # 條件 2: 手腕接近身體中線
        if abs(wrist_x_offset) > shoulder_width * 0.4:
            return False, 0.0

        # 條件 3: 手腕不能低於肩膀 (應該在向上運動)
        shoulder_idx = RIGHT_SHOULDER if hand == "right" else LEFT_SHOULDER
        shoulder = person_keypoints[shoulder_idx]
        if wrist[1] > shoulder[1]:  # 手腕Y座標更大表示更低
            return False, 0.0

        # 位置加分: 手腕高於手肘越多,加分越多
        vertical_lift = elbow[1] - wrist[1]  # Y軸向下為正,所以這個值為正表示手腕更高
        lift_score = min(1.0, vertical_lift / (shoulder_width * 0.3))
        position_bonus = 0.4 * lift_score

        return True, position_bonus

    def _update_position_history(self, person_id: int, keypoints):
        """更新位置歷史記錄"""
        if person_id not in self.position_history:
            self.position_history[person_id] = {"right": [], "left": []}

        try:
            person_keypoints = keypoints[person_id]

            for hand, wrist_idx in [("right", RIGHT_WRIST), ("left", LEFT_WRIST)]:
                wrist = person_keypoints[wrist_idx]
                if wrist[2] > 0.3:
                    history = self.position_history[person_id][hand]
                    history.append((wrist[0], wrist[1], time.time()))

                    # 保持歷史記錄長度
                    if len(history) > 8:
                        history.pop(0)

        except (IndexError, TypeError):
            pass

    def _calculate_hand_velocity(self, person_id: int, hand: str) -> float:
        """計算手部速度 (像素/秒)"""
        if person_id not in self.position_history or hand not in self.position_history[person_id]:
            return 0.0

        history = self.position_history[person_id][hand]
        if len(history) < 2:
            return 0.0

        # 只考慮最近三幀的移動
        recent_positions = history[-3:]
        total_distance = 0.0
        total_time = 0.0

        for i in range(1, len(recent_positions)):
            prev = recent_positions[i - 1]
            curr = recent_positions[i]
            distance = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            time_diff = max(curr[2] - prev[2], 1e-3)
            total_distance += distance
            total_time += time_diff

        if total_time == 0.0:
            return 0.0

        velocity = total_distance / total_time
        velocity = min(velocity, 3.0)
        return velocity

    def _calculate_players_distance(self, keypoints) -> Optional[float]:
        """計算兩個玩家之間的距離"""
        try:
            if len(keypoints) < 2:
                return None

            center_0 = self._get_body_center(keypoints[0])
            center_1 = self._get_body_center(keypoints[1])

            if center_0 is None or center_1 is None:
                return None

            distance = math.sqrt((center_0[0] - center_1[0]) ** 2 + (center_0[1] - center_1[1]) ** 2)
            return distance

        except (IndexError, TypeError):
            return None

    @staticmethod
    def _get_body_center(person_keypoints) -> Optional[Tuple[float, float]]:
        """獲取身體中心點"""
        try:
            neck = person_keypoints[NECK]
            hip = person_keypoints[MID_HIP]

            if neck[2] > 0.3 and hip[2] > 0.3:
                return (neck[0] + hip[0]) / 2, (neck[1] + hip[1]) / 2
            elif neck[2] > 0.3:
                return neck[0], neck[1]
            elif hip[2] > 0.3:
                return hip[0], hip[1]

        except (IndexError, TypeError):
            pass

        return None

    def reset_history(self):
        """重置歷史記錄（用於新場景開始）"""
        self.position_history.clear()
        self.player_cooldowns.clear()
        self.frame_count = 0
        print("Boxing detector history reset")

    def get_detection_stats(self) -> Dict[str, Any]:
        """獲取檢測統計信息"""
        return {
            "total_frames": self.frame_count,
            "tracked_players": len(self.position_history),
            "punch_configs": {name: config.name for name, config in self.punch_configs.items()},
            "cooldown_frames": self.cooldown_frames
        }

    def detect_comprehensive_actions(self, keypoints):
        """同時檢測攻擊和防禦動作"""
        # 檢測攻擊動作
        attack_frame_data = self.detect_actions(keypoints)

        # 檢測防禦動作
        defense_frame_data = self.defense_detector.detect_defense_actions(keypoints)

        return attack_frame_data, defense_frame_data


class BoxingVisualizer:
    """拳擊動作可視化器 - 獨立的視覺化模組"""

    def __init__(self, show_skeleton=True, show_debug=True):
        self.show_skeleton = show_skeleton
        self.show_debug = show_debug
        self.last_frame_time = time.time()

    def draw_debug_frame(self, frame, keypoints, frame_data: FrameData):
        """繪製調試信息幀"""
        if not self.show_debug:
            return frame

        result_frame = frame.copy()

        # 繪製骨架
        if self.show_skeleton and keypoints is not None:
            for person_id in range(min(len(keypoints), 2)):
                self._draw_skeleton(result_frame, keypoints[person_id], person_id)

        # 繪製動作信息
        for player_data in frame_data.players:
            self._draw_action_info(result_frame, keypoints, player_data)

        # 繪製系統狀態 (移除 width, height 參數)
        self._draw_system_status(result_frame, frame_data)

        return result_frame

    @staticmethod
    def _draw_skeleton(frame, person_keypoints, person_id):
        """繪製人體骨架"""
        color = (0, 255, 0) if person_id == 0 else (255, 0, 0)

        # 骨架連接定義
        skeleton_connections = [
            (NECK, RIGHT_SHOULDER), (NECK, LEFT_SHOULDER),
            (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
            (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
            (NECK, MID_HIP), (MID_HIP, RIGHT_HIP), (MID_HIP, LEFT_HIP),
            (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
            (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE)
        ]

        # 繪製骨架線條
        for start_idx, end_idx in skeleton_connections:
            start_point = person_keypoints[start_idx]
            end_point = person_keypoints[end_idx]

            if start_point[2] > 0.3 and end_point[2] > 0.3:
                cv2.line(frame,
                         (int(start_point[0]), int(start_point[1])),
                         (int(end_point[0]), int(end_point[1])),
                         color, 2)

        # 繪製關鍵點
        for keypoint in person_keypoints:
            if keypoint[2] > 0.3:
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 4, color, -1)

    @staticmethod
    def _draw_action_info(frame, keypoints, player_data: PlayerActionData):
        """繪製動作信息"""
        if keypoints is None or player_data.player_id >= len(keypoints):
            return

        person_keypoints = keypoints[player_data.player_id]
        color = (0, 255, 0) if player_data.player_id == 0 else (255, 0, 0)

        # 獲取頭部位置
        head_pos = None
        if person_keypoints[NOSE][2] > 0.3:
            head_pos = (int(person_keypoints[NOSE][0]), int(person_keypoints[NOSE][1]) - 50)
        elif person_keypoints[NECK][2] > 0.3:
            head_pos = (int(person_keypoints[NECK][0]), int(person_keypoints[NECK][1]) - 30)

        if head_pos:
            # 顯示動作類型
            action_text = player_data.action_type.replace('_', ' ').title()
            cv2.putText(frame, action_text, head_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 顯示攻擊手和拳法類型
            if player_data.is_attacking and player_data.attack_hand:
                hand_text = f"{player_data.attack_hand.upper()} {player_data.punch_type.upper()}"
                cv2.putText(frame, hand_text,
                            (head_pos[0], head_pos[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 顯示置信度
                conf_text = f"Conf: {player_data.confidence:.2f}"
                cv2.putText(frame, conf_text,
                            (head_pos[0], head_pos[1] + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_system_status(self, frame, frame_data: FrameData):
        """繪製系統狀態信息"""
        # 背景框
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 100), (255, 255, 255), 2)

        # 幀信息
        cv2.putText(frame, f"Frame: {frame_data.frame_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家數量
        cv2.putText(frame, f"Players: {len(frame_data.players)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家間距離
        if frame_data.players_distance is not None:
            cv2.putText(frame, f"Distance: {frame_data.players_distance:.1f}px", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS計算
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.last_frame_time = current_time


class ComprehensiveVisualizer:
    """綜合攻擊和防禦動作可視化器"""

    def __init__(self, show_skeleton=True, show_debug=True):
        self.attack_visualizer = BoxingVisualizer(show_skeleton, show_debug)
        super().__init__(show_skeleton, show_debug)
        self.defense_visualizer = BoxingDefenseVisualizer(show_skeleton, show_debug)
        self.last_frame_time = time.time()

    def draw_comprehensive_frame(self, frame, keypoints, attack_data, defense_data):
        """繪製綜合動作信息"""
        result_frame = frame.copy()

        # 繪製骨架
        if keypoints is not None:
            for person_id in range(min(len(keypoints), 2)):
                self._draw_skeleton(result_frame, keypoints[person_id], person_id)

        # 繪製每個玩家的狀態信息
        self._draw_player_actions(result_frame, keypoints, attack_data, defense_data)

        # 繪製系統狀態 (移除 width, height 參數)
        self._draw_system_status(result_frame, attack_data, defense_data)

        return result_frame

    @staticmethod
    def _draw_skeleton(frame, person_keypoints, person_id):
        """繪製人體骨架"""
        color = (0, 255, 0) if person_id == 0 else (255, 0, 0)

        skeleton_connections = [
            (NECK, RIGHT_SHOULDER), (NECK, LEFT_SHOULDER),
            (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
            (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
            (NECK, MID_HIP), (MID_HIP, RIGHT_HIP), (MID_HIP, LEFT_HIP),
            (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
            (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE)
        ]

        for start_idx, end_idx in skeleton_connections:
            start_point = person_keypoints[start_idx]
            end_point = person_keypoints[end_idx]

            if start_point[2] > 0.3 and end_point[2] > 0.3:
                cv2.line(frame,
                         (int(start_point[0]), int(start_point[1])),
                         (int(end_point[0]), int(end_point[1])),
                         color, 2)

        for keypoint in person_keypoints:
            if keypoint[2] > 0.3:
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 4, color, -1)


    def _draw_player_actions(self, frame, keypoints, attack_data, defense_data):
        """繪製玩家動作信息"""
        if keypoints is None:
            return

        # 為每個玩家繪製信息
        for person_id in range(min(len(keypoints), 2)):
            person_keypoints = keypoints[person_id]
            head_pos = self._get_head_position(person_keypoints)

            if not head_pos:
                continue

            color = (0, 255, 0) if person_id == 0 else (255, 0, 0)
            y_offset = 0

            # 查找該玩家的攻擊數據
            attack_player = None
            for player in attack_data.players:
                if player.player_id == person_id:
                    attack_player = player
                    break

            # 查找該玩家的防禦數據
            defense_player = None
            for player in defense_data.players:
                if player.player_id == person_id:
                    defense_player = player
                    break

            # 顯示攻擊動作
            if attack_player and attack_player.is_attacking:
                attack_text = f"ATTACK: {attack_player.punch_type.upper()} ({attack_player.attack_hand.upper()})"
                cv2.putText(frame, attack_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                y_offset += 25

                conf_text = f"Conf: {attack_player.confidence:.2f}"
                cv2.putText(frame, conf_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                y_offset += 25

            # 顯示防禦動作
            if defense_player and defense_player.is_defending:
                defense_text = f"DEFENSE: {defense_player.defense_type.replace('_', ' ').upper()}"
                cv2.putText(frame, defense_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                y_offset += 25

                if defense_player.guard_hand:
                    hand_text = f"Guard: {defense_player.guard_hand.upper()}"
                    cv2.putText(frame, hand_text, (head_pos[0], head_pos[1] + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    y_offset += 25

            # 顯示閃避動作
            if defense_player and defense_player.is_dodging:
                dodge_text = f"DODGE: {defense_player.dodge_direction.upper()}"
                cv2.putText(frame, dodge_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                y_offset += 25

            # 如果沒有特殊動作，顯示 IDLE
            if ((not attack_player or not attack_player.is_attacking) and
                    (not defense_player or (not defense_player.is_defending and not defense_player.is_dodging))):
                cv2.putText(frame, "IDLE", (head_pos[0], head_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    @staticmethod
    def _get_head_position(person_keypoints):
        """獲取頭部位置"""
        if person_keypoints[NOSE][2] > 0.3:
            return int(person_keypoints[NOSE][0]), int(person_keypoints[NOSE][1]) - 50
        elif person_keypoints[NECK][2] > 0.3:
            return int(person_keypoints[NECK][0]), int(person_keypoints[NECK][1]) - 30
        return None

    def _draw_system_status(self, frame, attack_data, defense_data):
        """繪製系統狀態信息"""
        # 背景框
        cv2.rectangle(frame, (10, 10), (220, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (220, 140), (255, 255, 255), 2)

        # 幀信息
        cv2.putText(frame, f"Frame: {attack_data.frame_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家數量
        cv2.putText(frame, f"Players: {len(attack_data.players)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 攻擊統計
        attacking_count = sum(1 for p in attack_data.players if p.is_attacking)
        cv2.putText(frame, f"Attacking: {attacking_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 防禦統計
        defending_count = sum(1 for p in defense_data.players if p.is_defending)
        cv2.putText(frame, f"Defending: {defending_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家間距離
        if attack_data.players_distance is not None:
            cv2.putText(frame, f"Distance: {attack_data.players_distance:.1f}px", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS計算
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.last_frame_time = current_time


def run_standalone_detection(camera_index=0, ground_truth_action="test"):
    """獨立運行綜合動作檢測（攻擊+防禦）"""
    print("=== Running Comprehensive Boxing Detection (Attack + Defense) ===")
    print(f"=== Running Detection for: {ground_truth_action} ===")

    detector = BoxingActionDetector()
    visualizer = ComprehensiveVisualizer()

    # --- 開始：日誌設定 ---
    # 產生帶有時間戳和動作名稱的日誌檔名
    log_filename = f"log_{ground_truth_action}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    # 打開檔案準備寫入
    log_file = open(log_filename, 'w', newline='', encoding='utf-8')

    # 建立 CSV 寫入器
    log_writer = csv.writer(log_file)

    # 寫入試算表之實驗數據標頭，可擴充想紀錄的內容
    header = [
        "timestamp", "frame_id", "person_id", "ground_truth_action",
        "detected_action", "confidence", "is_correct"
    ]
    log_writer.writerow(header)
    # --- 結束：日誌設定 ---

    try:
        for frame_id, keypoints_data, frame in get_keypoints_stream(video_source=camera_index):

            # 同時檢測攻擊和防禦動作
            attack_data, defense_data = detector.detect_comprehensive_actions(keypoints_data)

            # --- 開始：寫入日誌 ---
            # 遍歷每個被偵測到的人
            num_players = max(len(attack_data.players), len(defense_data.players))
            for person_id in range(num_players):

                # 找到對應玩家的攻擊和防禦數據
                attack_player = next((p for p in attack_data.players if p.player_id == person_id), None)
                defense_player = next((p for p in defense_data.players if p.player_id == person_id), None)

                detected_action = "idle"
                confidence = 0.0

                # 優先判斷高置信度的動作
                if attack_player and attack_player.is_attacking:
                    detected_action = attack_player.punch_type
                    confidence = attack_player.confidence
                elif defense_player and defense_player.is_defending:
                    detected_action = defense_player.defense_type
                    confidence = defense_player.confidence
                elif defense_player and defense_player.is_dodging:
                    detected_action = defense_player.dodge_direction
                    confidence = defense_player.confidence

                # 準備寫入的單行數據
                log_row = [
                    time.time(),
                    frame_id,
                    person_id,
                    "UNLABELED",  # ground_truth_action 留待人工標註
                    detected_action,
                    confidence,
                    "UNLABELED",  # is_correct 留待人工標註
                ]

                # 寫入 CSV
                log_writer.writerow(log_row)
            # --- 結束：寫入日誌 ---

            # 可視化結果
            result_frame = visualizer.draw_comprehensive_frame(frame, keypoints_data, attack_data, defense_data)

            # 在終端打印檢測結果
            for player_id in range(min(len(attack_data.players), len(defense_data.players))):
                # 查找攻擊數據
                attack_player = None
                for p in attack_data.players:
                    if p.player_id == player_id:
                        attack_player = p
                        break

                # 查找防禦數據
                defense_player = None
                for p in defense_data.players:
                    if p.player_id == player_id:
                        defense_player = p
                        break

                # 打印動作狀態
                actions = []

                if attack_player and attack_player.is_attacking:
                    actions.append(
                        f"ATTACK: {attack_player.punch_type} with {attack_player.attack_hand} (conf: {attack_player.confidence:.2f})")

                if defense_player and defense_player.is_defending:
                    actions.append(f"DEFENSE: {defense_player.defense_type} (conf: {defense_player.confidence:.2f})")

                if defense_player and defense_player.is_dodging:
                    actions.append(f"DODGE: {defense_player.dodge_direction}")

                if actions:
                    print(f"Player {player_id}: {' | '.join(actions)}")

            # 顯示結果
            cv2.imshow('Boxing Detection - Attack & Defense', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Detection stopped by user")
    finally:
        # --- 開始：關閉日誌檔案 ---
        if log_file:
            log_file.close()
            print(f"Log saved to: {log_filename}")
        # --- 結束：關閉日誌檔案 ---
        cv2.destroyAllWindows()
        print("Comprehensive detection completed")


if __name__ == "__main__":
    """主函數---僅用於測試動作檢測"""
    import argparse

    parser = argparse.ArgumentParser(description='Boxing Action Detection - Standalone Mode')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')

    parser.add_argument(
        '--action',
        type=str,
        default="general",
        help='Specify the detection mode for this test run (e.g., "training_session", "competition_mode")'
    )

    args = parser.parse_args()

    # 運行獨立檢測模式
    run_standalone_detection(camera_index=args.camera, ground_truth_action=args.action)
