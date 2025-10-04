# guard_detection.py - 與punch_detection.py相容的防禦檢測器

import numpy as np
import time
# import math
from collections import deque
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum
import cv2

from angle import calculate_normalized_angle, calculate_shoulder_width
from angle import (NOSE, NECK, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
                   LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, MID_HIP,
                   RIGHT_HIP, LEFT_HIP, RIGHT_KNEE, LEFT_KNEE,
                   RIGHT_ANKLE, LEFT_ANKLE, RIGHT_EAR, LEFT_EAR)


class DefenseType(Enum):
    """防禦類型枚舉"""
    IDLE = "idle"
    BLOCK = "block"  # 格擋 - 防直拳
    SLIP = "slip"  # 潛避 - 防鉤拳
    SHELL_GUARD = "shell_guard"  # 貝殼防禦 - 防上鉤拳


@dataclass
class DefenseConfig:
    """防禦動作配置"""
    name: str
    angle_range: Tuple[float, float]  # 手肘角度範圍
    position_requirement: str  # 位置要求描述
    effectiveness: float  # 防禦效果 0-1
    stamina_cost: float  # 體力消耗 0-1


@dataclass
class PlayerDefenseData:
    """與punch_detection.py相容的防禦數據"""
    player_id: int
    defense_type: str
    guard_hand: Optional[str]
    is_defending: bool
    is_dodging: bool
    dodge_direction: Optional[str]
    confidence: float
    effectiveness: float
    counters: List[str]  # 對抗的拳法類型
    timestamp: float


@dataclass
class DefenseFrameData:
    """防禦檢測幀數據"""
    frame_id: int
    timestamp: float
    players: List[PlayerDefenseData]


class BoxingDefenseDetector:
    """
    拳擊防禦檢測器 - 與punch_detection.py整合版本
    專門針對三種基本拳擊的對應防禦：
    - 直拳 -> 格擋 (Block)
    - 鉤拳 -> 潛避 (Slip)
    - 上鉤拳 -> 貝殼防禦(Shell_Guard)
    """

    def __init__(self, cooldown=1.0, history_len=20):
        # 狀態管理
        self.cooldown_frames  = {}  # 每個玩家的冷卻時間
        self.cooldown = cooldown
        self.last_defense = {}  # 每個玩家的上次防禦
        self.frame_count = 0

        # 身高基準管理（用於潛避檢測）
        self.neck_y_history = {}  # 每個玩家的頸部高度歷史
        self.base_neck_y = {}  # 每個玩家的基準頸部高度
        self.is_stable_stance = {}  # 每個玩家的穩定站姿狀態
        self.stable_frames = {}  # 每個玩家的穩定幀數
        self.min_stable_frames = 8  # 需要連續穩定幀數才更新基準
        self.history_len = history_len

        # 防禦配置
        self.defense_configs = {
            "block": DefenseConfig(
                name="格擋",
                angle_range=(80, 150),  # 手肘彎曲但伸展
                position_requirement="手向前伸出，形成盾牌",
                effectiveness=0.85,
                stamina_cost=0.1
            ),
            "slip": DefenseConfig(
                name="潛避",
                angle_range=(30, 120),  # 角度要求較寬泛
                position_requirement="身體下沉或側移",
                effectiveness=0.9,
                stamina_cost=0.2
            ),
            "shell_guard": DefenseConfig(
                name="貝殼防禦",
                angle_range=(30, 70),  # 貝殼防禦的角度更緊縮
                position_requirement="雙臂貼近身體，手肘向下保護肋骨",
                effectiveness=0.88,
                stamina_cost=0.12
            )
        }

        # 與punch_detection.py相同的關鍵點索引
        self.NOSE = 0
        self.NECK = 1
        self.RIGHT_SHOULDER = 2
        self.RIGHT_ELBOW = 3
        self.RIGHT_WRIST = 4
        self.LEFT_SHOULDER = 5
        self.LEFT_ELBOW = 6
        self.LEFT_WRIST = 7
        self.MID_HIP = 8

        print("=== Boxing Defense Detector Initialized ===")
        print("Defense Types: Block (vs Straight), Slip (vs Hook), Shell_Guard (vs Uppercut)")

    def detect_defense_actions(self, keypoints) -> DefenseFrameData:
        """主要防禦檢測函數 - 與punch_detection.py相容的接口"""
        self.frame_count += 1
        timestamp = time.time()

        players_data = []

        if keypoints is not None and len(keypoints) > 0:
            # 處理最多兩個人
            num_persons = min(len(keypoints), 2)

            for person_id in range(num_persons):
                defense_data = self._detect_player_defense(person_id, keypoints, timestamp)
                if defense_data:
                    players_data.append(defense_data)

        return DefenseFrameData(
            frame_id=self.frame_count,
            timestamp=timestamp,
            players=players_data
        )

    def _detect_player_defense(self, person_id: int, keypoints, timestamp: float) -> Optional[PlayerDefenseData]:
        """檢測單個玩家的防禦動作"""
        try:
            # 初始化玩家状态
            if person_id not in self.neck_y_history:
                self.neck_y_history[person_id] = deque(maxlen=self.history_len)
                self.base_neck_y[person_id] = None
                self.is_stable_stance[person_id] = False
                self.stable_frames[person_id] = 0
                self.cooldown_frames [person_id] = 0
                self.last_defense[person_id] = None

            # 更新基準線
            self._update_stance_baseline(person_id, keypoints)

            # 檢查冷卻時間
            if timestamp - self.cooldown_frames [person_id] < self.cooldown:
                return None

            # 使用angle.py的正規化方法檢測防禦
            defense_result = self._detect_defense(person_id, keypoints)

            if defense_result:
                self.cooldown_frames [person_id] = timestamp
                self.last_defense[person_id] = defense_result["type"]

                is_dodging = defense_result["type"] == DefenseType.SLIP.value
                dodge_direction = defense_result.get("direction") if is_dodging else None
                guard_hand = defense_result.get("hands")

                print(f"[Player {person_id}] {time.strftime('%H:%M:%S')} 檢測到防禦: {defense_result['name']} "
                      f"(置信度: {defense_result['confidence']:.2f})")

                return PlayerDefenseData(
                    player_id=person_id,
                    defense_type=defense_result["type"],
                    guard_hand=guard_hand,
                    is_defending=not is_dodging,
                    is_dodging=is_dodging,
                    dodge_direction=dodge_direction,
                    confidence=defense_result["confidence"],
                    effectiveness=defense_result["effectiveness"],
                    counters=defense_result.get("counters", []),
                    timestamp=timestamp
                )

            return None

        except Exception as e:
            print(f"Error detecting player {person_id} defense: {e}")
            return None

    def _get_keypoint(self, keypoints, person_id: int, index: int, confidence_threshold=0.3):
        """安全獲取關鍵點 - 適配雙人檢測"""
        try:
            if (keypoints is not None and
                    len(keypoints) > person_id and
                    keypoints[person_id].shape[0] > index and
                    keypoints[person_id][index, 2] > confidence_threshold):
                return keypoints[person_id][index, :2]
        except (IndexError, TypeError):
            pass
        return None

    def _calculate_angle(self, a, b, c):
        """計算三點夾角"""
        if any(p is None for p in [a, b, c]):
            return None

        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b

        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return None

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        return np.degrees(np.arccos(cosine_angle))

    def _update_stance_baseline(self, person_id: int, keypoints):
        """更新身體基準線（僅在穩定站姿時）"""
        neck = self._get_keypoint(keypoints, person_id, NECK)
        l_shoulder = self._get_keypoint(keypoints, person_id, LEFT_SHOULDER)
        r_shoulder = self._get_keypoint(keypoints, person_id, RIGHT_SHOULDER)
        l_wrist = self._get_keypoint(keypoints, person_id, LEFT_WRIST)
        r_wrist = self._get_keypoint(keypoints, person_id, RIGHT_WRIST)

        if not all(p is not None for p in [neck, l_shoulder, r_shoulder]):
            self.is_stable_stance[person_id] = False
            self.stable_frames[person_id] = 0
            return

        # 判斷是否為穩定站姿
        shoulder_y_diff = abs(l_shoulder[1] - r_shoulder[1])
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])

        hands_near_head = False
        if l_wrist is not None and l_wrist[1] < neck[1] - 20:
            hands_near_head = True
        if r_wrist is not None and r_wrist[1] < neck[1] - 20:
            hands_near_head = True

        # 判斷為穩定站姿的條件
        is_current_stable = (
                shoulder_y_diff < shoulder_width * 0.1 and  # 肩膀水平
                not hands_near_head  # 手不在头部附近
        )

        if is_current_stable:
            self.stable_frames[person_id] += 1
            if self.stable_frames[person_id] >= self.min_stable_frames:
                self.is_stable_stance[person_id] = True
                self.neck_y_history[person_id].append(neck[1])

                # 计算基准身高
                if len(self.neck_y_history[person_id]) >= 5:
                    self.base_neck_y[person_id] = np.median(self.neck_y_history[person_id])
        else:
            self.stable_frames[person_id] = 0
            self.is_stable_stance[person_id] = False

    def _detect_defense(self, person_id: int, keypoints):
        """防禦檢測函數"""
        # 獲取所有需要的關鍵點
        left_elbow_angle = calculate_normalized_angle(
            keypoints, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], person_id)
        right_elbow_angle = calculate_normalized_angle(
            keypoints, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], person_id)

        # 使用angle.py計算肩寬作为正規化基準
        shoulder_width = calculate_shoulder_width(keypoints, person_id)

        # 獲取關鍵點用於位置判斷
        keypoints_dict = {}
        for i in [NOSE, NECK, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                  LEFT_WRIST, RIGHT_WRIST, MID_HIP, RIGHT_EAR, LEFT_EAR]:
            keypoints_dict[i] = self._get_keypoint(keypoints, person_id, i)

        # 按優先級檢測防禦動作
        detected_defense = None

        # 1. 檢測規避（優先級最高）- 使用正規化身高變化
        slip_result = self._detect_slip(person_id, keypoints_dict, shoulder_width)
        if slip_result:
            detected_defense = slip_result

        # 2. 檢測緊密防禦 - 使用正規化角度
        if not detected_defense:
            shell_guard_result = self._detect_shell_guard(keypoints_dict, left_elbow_angle, right_elbow_angle,
                                                          shoulder_width)
            if shell_guard_result:
                detected_defense = shell_guard_result

        # 3. 檢測格擋 - 使用正規化距離和角度
        if not detected_defense:
            block_result = self._detect_block(keypoints_dict, left_elbow_angle, right_elbow_angle,
                                                         shoulder_width)
            if block_result:
                detected_defense = block_result

        return detected_defense

    def _detect_slip(self, person_id: int, keypoints_dict, shoulder_width):
        """檢測潛避動作 - 防禦鉤拳"""
        if (self.base_neck_y[person_id] is None or
                keypoints_dict[NECK] is None or
                shoulder_width is None or shoulder_width <= 0):
            return None

        neck = keypoints_dict[NECK]
        l_shoulder = keypoints_dict[LEFT_SHOULDER]
        r_shoulder = keypoints_dict[RIGHT_SHOULDER]

        if l_shoulder is None or r_shoulder is None:
            return None

        # 使用正規化的身高變化檢測
        height_drop = neck[1] - self.base_neck_y[person_id]

        # 條件：身高降低超過肩寬的25%（正規化閾值）
        normalized_drop_threshold = shoulder_width * 0.25
        significant_drop = height_drop > normalized_drop_threshold

        # 檢測側移 - 使用正規化距離
        body_center_x = (l_shoulder[0] + r_shoulder[0]) / 2
        neck_offset = abs(neck[0] - body_center_x)
        normalized_lateral_threshold = shoulder_width * 0.3
        significant_lateral = neck_offset > normalized_lateral_threshold

        if significant_drop or significant_lateral:
            # 計算置信度
            drop_score = min(height_drop / (shoulder_width * 0.4), 1.0) if height_drop > 0 else 0
            lateral_score = min(neck_offset / (shoulder_width * 0.5), 1.0) if neck_offset > 0 else 0

            confidence = max(drop_score, lateral_score) * 0.8

            # 判斷潛避方向
            direction = "down"
            if significant_lateral:
                direction = "left" if neck[0] < body_center_x else "right"

            return {
                "type": DefenseType.SLIP.value,
                "name": f"潜避 ({direction})",
                "confidence": confidence,
                "effectiveness": self.defense_configs["slip"].effectiveness,
                "direction": direction,
                "counters": ["hook_punch"]
            }

        return None

    def _detect_shell_guard(self, keypoints_dict, left_angle, right_angle, shoulder_width):
        """檢測貝殼防禦 - 防禦上鉤拳 """
        if left_angle is None or right_angle is None or shoulder_width is None or shoulder_width <= 0:
            return None

        config = self.defense_configs["shell_guard"]

        # 1. 角度檢查：雙臂必須緊縮
        angles_in_range = (config.angle_range[0] <= left_angle <= config.angle_range[1] and
                           config.angle_range[0] <= right_angle <= config.angle_range[1])

        if not angles_in_range:
            return None

        # 獲取所需關鍵點
        l_elbow = keypoints_dict[LEFT_ELBOW]
        r_elbow = keypoints_dict[RIGHT_ELBOW]
        l_shoulder = keypoints_dict[LEFT_SHOULDER]
        r_shoulder = keypoints_dict[RIGHT_SHOULDER]

        if any(p is None for p in [l_elbow, r_elbow, l_shoulder, r_shoulder]):
            return None

        # 2. 位置檢查：手肘必須緊貼身體
        # 計算手肘到對應肩膀的水平距離，並用肩寬進行標準化
        right_elbow_dist_norm = abs(r_elbow[0] - r_shoulder[0]) / shoulder_width
        left_elbow_dist_norm = abs(l_elbow[0] - l_shoulder[0]) / shoulder_width

        # 閾值可以設為例如 0.3 (即手肘水平距離小於肩寬的30%)
        right_elbow_close = right_elbow_dist_norm < 0.3
        left_elbow_close = left_elbow_dist_norm < 0.3

        if right_elbow_close and left_elbow_close:
            # 計算置信度
            # 角度分數
            optimal_angle = (config.angle_range[0] + config.angle_range[1]) / 2
            angle_score = 1.0 - (abs(left_angle - optimal_angle) + abs(right_angle - optimal_angle)) / 80.0
            angle_score = max(0, angle_score)

            # 位置分數 (越貼近分數越高)
            pos_score = 1.0 - (right_elbow_dist_norm + left_elbow_dist_norm) / 0.6
            pos_score = max(0, pos_score)

            confidence = (angle_score * 0.6 + pos_score * 0.4) * 0.95

            if confidence > 0.5: # 增加一個最低置信度門檻
                return {
                    "type": DefenseType.SHELL_GUARD.value,
                    "name": "貝殼防禦",
                    "confidence": confidence,
                    "effectiveness": config.effectiveness,
                    "hands": "both",
                    "counters": ["uppercut_punch"]
                }

        return None

    def _detect_block(self, keypoints_dict, left_angle, right_angle, shoulder_width):
        """檢測格擋動作 - 防禦直拳"""
        if shoulder_width is None or shoulder_width <= 0:
            return None

        config = self.defense_configs["block"]

        # 检测右手格挡
        right_block = self._detect_single_hand_block(
            keypoints_dict, "right", right_angle, config, shoulder_width
        )
        if right_block:
            return right_block

        # 检测左手格挡
        left_block = self._detect_single_hand_block(
            keypoints_dict, "left", left_angle, config, shoulder_width
        )
        if left_block:
            return left_block

        return None

    def _detect_single_hand_block(self, keypoints_dict, hand, angle, config, shoulder_width):
        """檢測單手格擋"""
        if angle is None or not (config.angle_range[0] <= angle <= config.angle_range[1]):
            return None

            # 選擇對應的關鍵點
        if hand == "right":
            wrist = keypoints_dict[RIGHT_WRIST]
            elbow = keypoints_dict[RIGHT_ELBOW]
            shoulder = keypoints_dict[RIGHT_SHOULDER]
        else:
            wrist = keypoints_dict[LEFT_WRIST]
            elbow = keypoints_dict[LEFT_ELBOW]
            shoulder = keypoints_dict[LEFT_SHOULDER]

        if any(p is None for p in [wrist, elbow, shoulder]):
            return None

            # 檢查手腕是否在手肘上方
        wrist_above_elbow = wrist[1] < elbow[1]
        if not wrist_above_elbow:
            return None

        # 使用正規化距離計算手腕到臉部的距離
        face_center = keypoints_dict[NOSE] if keypoints_dict[NOSE] is not None else keypoints_dict[RIGHT_EAR]
        if face_center is None:
            return None

        # 計算正規化距離
        dist_to_face = np.linalg.norm(np.array(wrist) - np.array(face_center))
        normalized_dist = dist_to_face / shoulder_width

        # 條件：正規化距離小於0.6（即小於60%的肩寬）
        if normalized_dist < 0.6:
            # 計算置信度
            angle_optimal = (config.angle_range[0] + config.angle_range[1]) / 2
            angle_score = 1.0 - abs(angle - angle_optimal) / 60.0
            angle_score = max(angle_score, 0)

            distance_score = 1.0 - normalized_dist / 0.6  # 距離越近 分數越高
            position_score = 0.8

            confidence = (angle_score * 0.4 + distance_score * 0.3 + position_score * 0.3) * 0.85

            return {
                "type": DefenseType.BLOCK.value,
                "name": f"{hand}手格挡",
                "confidence": confidence,
                "effectiveness": config.effectiveness,
                "hands": hand,
                "counters": ["straight_punch"]
            }

        return None

    def _detect_double_hand_block(self, keypoints_dict, left_angle, right_angle, config):
        """檢測雙手格擋"""
        if left_angle is None or right_angle is None:
            return None

        # 雙手都要在合適角度範圍內
        both_angles_good = (
                config.angle_range[0] <= left_angle <= config.angle_range[1] and
                config.angle_range[0] <= right_angle <= config.angle_range[1]
        )

        if not both_angles_good:
            return None

        l_wrist = keypoints_dict[self.LEFT_WRIST]
        r_wrist = keypoints_dict[self.RIGHT_WRIST]
        l_elbow = keypoints_dict[self.LEFT_ELBOW]
        r_elbow = keypoints_dict[self.RIGHT_ELBOW]
        neck = keypoints_dict[self.NECK]

        if any(p is None for p in [l_wrist, r_wrist, l_elbow, r_elbow, neck]):
            return None

        # 檢測條件
        both_wrists_up = l_wrist[1] < l_elbow[1] and r_wrist[1] < r_elbow[1]
        if not both_wrists_up:
            return None

        # 檢查雙手是否在頭部附近
        head_level_protection = (
                l_wrist[1] < neck[1] + 30 and r_wrist[1] < neck[1] + 30
        )

        if head_level_protection:
            # 計算置信度
            avg_angle = (left_angle + right_angle) / 2
            angle_score = 1.0 - abs(avg_angle - 120) / 60.0
            angle_score = max(angle_score, 0)

            position_score = 0.9
            confidence = (angle_score * 0.5 + position_score * 0.5) * 0.95

            return {
                "type": DefenseType.BLOCK.value,
                "name": "雙手格擋",
                "confidence": confidence,
                "effectiveness": config.effectiveness * 1.1,  # 雙手格擋效果更好
                "hands": "both",
                "counters": ["straight_punch"]  # 對抗直拳
            }

        return None

    def get_defense_stats(self, person_id: int = None):
        """獲取防禦統計信息"""
        if person_id is not None:
            return {
                "player_id": person_id,
                "base_neck_y": self.base_neck_y.get(person_id),
                "is_stable_stance": self.is_stable_stance.get(person_id, False),
                "stable_frames": self.stable_frames.get(person_id, 0),
                "neck_history_length": len(self.neck_y_history.get(person_id, [])),
                "last_defense": self.last_defense.get(person_id),
                "using_normalized_methods": True
            }
        else:
            return {
                "total_players": len(self.base_neck_y),
                "defense_configs": {name: config.name for name, config in self.defense_configs.items()},
                "using_angle_normalization": True,
                "using_distance_normalization": True
            }

    def reset(self, person_id: int = None):
        """重置檢測器狀態"""
        if person_id is not None:
            if person_id in self.neck_y_history:
                self.neck_y_history[person_id].clear()
                self.base_neck_y[person_id] = None
                self.is_stable_stance[person_id] = False
                self.stable_frames[person_id] = 0
                self.last_defense[person_id] = None
                self.cooldown_frames [person_id] = 0
        else:
            self.neck_y_history.clear()
            self.base_neck_y.clear()
            self.is_stable_stance.clear()
            self.stable_frames.clear()
            self.last_defense.clear()
            self.cooldown_frames .clear()
            self.frame_count = 0
        print(f"Defense detector reset {'for all players' if person_id is None else f'for player {person_id}'}")


class BoxingDefenseVisualizer:
    """防禦動作可視化器 - 與punch_detection.py風格保持一致"""

    def __init__(self, show_skeleton=True, show_debug=True):
        self.show_skeleton = show_skeleton
        self.show_debug = show_debug
        self.last_frame_time = time.time()

        # 與punch_detection.py相同的關鍵點索引
        self.NOSE = 0
        self.NECK = 1
        self.RIGHT_SHOULDER = 2
        self.RIGHT_ELBOW = 3
        self.RIGHT_WRIST = 4
        self.LEFT_SHOULDER = 5
        self.LEFT_ELBOW = 6
        self.LEFT_WRIST = 7
        self.MID_HIP = 8

    def draw_defense_frame(self, frame, keypoints, defense_data: DefenseFrameData):
        """繪製防禦動作信息"""
        if not self.show_debug:
            return frame

        result_frame = frame.copy()
        height, width = frame.shape[:2]

        # 繪製骨架
        if self.show_skeleton and keypoints is not None:
            for person_id in range(min(len(keypoints), 2)):
                self._draw_skeleton(result_frame, keypoints[person_id], person_id)

        # 繪製防禦動作信息
        for player_data in defense_data.players:
            self._draw_defense_info(result_frame, keypoints, player_data)

        # 繪製系統狀態
        self._draw_system_status(result_frame, defense_data, width, height)

        return result_frame

    def _draw_skeleton(self, frame, person_keypoints, person_id):
        """繪製人體骨架"""
        color = (0, 255, 0) if person_id == 0 else (255, 0, 0)

        skeleton_connections = [
            (self.NECK, self.RIGHT_SHOULDER), (self.NECK, self.LEFT_SHOULDER),
            (self.RIGHT_SHOULDER, self.RIGHT_ELBOW), (self.RIGHT_ELBOW, self.RIGHT_WRIST),
            (self.LEFT_SHOULDER, self.LEFT_ELBOW), (self.LEFT_ELBOW, self.LEFT_WRIST),
            (self.NECK, self.MID_HIP)
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

    def _draw_defense_info(self, frame, keypoints, player_data):
        """繪製防禦動作信息"""
        if keypoints is None or player_data.player_id >= len(keypoints):
            return

        person_keypoints = keypoints[player_data.player_id]
        color = (0, 255, 255)  # 青色用於防禦

        # 獲取頭部位置
        head_pos = None
        if person_keypoints[self.NOSE][2] > 0.3:
            head_pos = (int(person_keypoints[self.NOSE][0]), int(person_keypoints[self.NOSE][1]) - 50)
        elif person_keypoints[self.NECK][2] > 0.3:
            head_pos = (int(person_keypoints[self.NECK][0]), int(person_keypoints[self.NECK][1]) - 30)

        if head_pos:
            # 顯示防禦類型
            defense_text = player_data.defense_type.replace('_', ' ').upper()
            cv2.putText(frame, f"DEFENSE: {defense_text}", head_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 顯示防禦手
            if player_data.guard_hand:
                hand_text = f"Guard: {player_data.guard_hand.upper()}"
                cv2.putText(frame, hand_text,
                            (head_pos[0], head_pos[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 顯示閃避方向
            if player_data.is_dodging and player_data.dodge_direction:
                dodge_text = f"Dodge: {player_data.dodge_direction.upper()}"
                cv2.putText(frame, dodge_text,
                            (head_pos[0], head_pos[1] + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 顯示置信度
            conf_text = f"Conf: {player_data.confidence:.2f}"
            cv2.putText(frame, conf_text,
                        (head_pos[0], head_pos[1] + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_system_status(self, frame, defense_data, width, height):
        """繪製防禦系統狀態信息"""
        # 背景框
        cv2.rectangle(frame, (10, height - 120), (350, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, height - 120), (350, height - 10), (255, 255, 255), 2)

        # 幀信息
        cv2.putText(frame, f"Defense Frame: {defense_data.frame_id}",
                    (20, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 防禦玩家數量
        defending_count = sum(1 for p in defense_data.players if p.is_defending)
        dodging_count = sum(1 for p in defense_data.players if p.is_dodging)

        cv2.putText(frame, f"Defending: {defending_count}, Dodging: {dodging_count}",
                    (20, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 時間戳
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}",
                    (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 顯示可對抗的拳法
        counters_text = "Counters: Straight->Block, Hook->Slip, Uppercut->Cover"
        cv2.putText(frame, counters_text,
                    (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
