# id_tracker.py - 基於IoU的雙人身份追蹤模組

import numpy as np
# import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TrackedPerson:
    """追蹤到的人物數據"""
    person_id: int  # 追蹤ID (0=Player1, 1=Player2, 2+=其他)
    keypoints: np.ndarray  # 關鍵點數據 [25, 3]
    bounding_box: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    confidence: float  # 追蹤置信度
    frame_count: int  # 該ID已追蹤的幀數


class IDTracker:
    """
    基於IoU (Intersection over Union) 的身份追蹤器
    專為拳擊雙人對戰場景設計
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        """
        初始化追蹤器

        Args:
            iou_threshold: IoU匹配閾值，低於此值視為不同人
            max_age: 允許的最大消失幀數，超過後ID將被釋放
            min_hits: 新ID需要連續出現的最小幀數才視為有效
        """
        # 追蹤參數
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        # 追蹤狀態
        self.previous_boxes = []  # 前一幀的bounding boxes
        self.previous_ids = []  # 前一幀的ID列表
        self.next_id = 0  # 下一個可用的ID

        # ID管理
        self.id_ages = {}  # 每個ID的年齡（未匹配到的幀數）
        self.id_hits = {}  # 每個ID的命中次數
        self.id_confidences = {}  # 每個ID的歷史置信度

        # 固定玩家ID管理
        self.player1_id = None  # Player 1的固定ID
        self.player2_id = None  # Player 2的固定ID
        self.stable_id_threshold = 10  # 連續追蹤10幀後固定ID

        print("=== ID Tracker Initialized ===")
        print(f"IoU Threshold: {iou_threshold}")
        print(f"Max Age: {max_age} frames")
        print(f"Min Hits: {min_hits} frames")

    @staticmethod
    def _get_bounding_box(keypoints: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        計算關鍵點的邊界框

        Args:
            keypoints: [25, 3] 關鍵點數組

        Returns:
            (x_min, y_min, x_max, y_max) 或 None
        """
        # 僅考慮置信度大於0.1的點
        valid_mask = keypoints[:, 2] > 0.1
        valid_points = keypoints[valid_mask][:, :2]

        if len(valid_points) == 0:
            return None

        x_min = np.min(valid_points[:, 0])
        y_min = np.min(valid_points[:, 1])
        x_max = np.max(valid_points[:, 0])
        y_max = np.max(valid_points[:, 1])

        # 防止退化的邊界框
        if x_max - x_min < 10 or y_max - y_min < 10:
            return None

        return x_min, y_min, x_max, y_max

    @staticmethod
    def _calculate_iou(boxa: Tuple[float, float, float, float],
                       boxb: Tuple[float, float, float, float]) -> float:
        """
        計算兩個邊界框的IoU (Intersection over Union)

        Args:
            boxa: (x_min, y_min, x_max, y_max)
            boxb: (x_min, y_min, x_max, y_max)

        Returns:
            IoU值 (0.0 - 1.0)
        """
        # 計算交集區域
        x_left = max(boxa[0], boxb[0])
        y_top = max(boxa[1], boxb[1])
        x_right = min(boxa[2], boxb[2])
        y_bottom = min(boxa[3], boxb[3])

        # 檢查是否有交集
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # 計算交集面積
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # 計算兩個框的面積
        boxa_area = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1])
        boxb_area = (boxb[2] - boxb[0]) * (boxb[3] - boxb[1])

        # 計算聯集面積
        union_area = boxa_area + boxb_area - intersection_area

        # 防止除以零
        if union_area == 0:
            return 0.0

        # 計算IoU
        iou = intersection_area / union_area
        return iou

    @staticmethod
    def _calculate_box_confidence(keypoints: np.ndarray) -> float:
        """
        計算邊界框的置信度（基於關鍵點可見度）

        Args:
            keypoints: [25, 3] 關鍵點數組

        Returns:
            置信度 (0.0 - 1.0)
        """
        # 關鍵的上半身關鍵點
        key_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # NOSE, NECK, 雙肩, 雙肘, 雙腕

        valid_count = 0
        total_confidence = 0.0

        for idx in key_indices:
            if idx < len(keypoints):
                conf = keypoints[idx, 2]
                if conf > 0.1:
                    valid_count += 1
                    total_confidence += conf

        if valid_count == 0:
            return 0.0

        return total_confidence / len(key_indices)

    def track_and_assign_ids(self, current_keypoints_list: List[np.ndarray]) -> List[TrackedPerson]:
        """
        追蹤並分配ID給當前幀的所有骨架

        Args:
            current_keypoints_list: 當前幀的關鍵點列表，每個元素為 [25, 3] 數組

        Returns:
            TrackedPerson對象列表
        """
        # 步驟1: 計算當前幀所有骨架的邊界框
        current_boxes = []
        current_confidences = []

        for kp in current_keypoints_list:
            box = self._get_bounding_box(kp)
            conf = self._calculate_box_confidence(kp) if box else 0.0
            current_boxes.append(box)
            current_confidences.append(conf)

        # 步驟2: 與前一幀進行匹配
        matched_indices = {}  # {current_index: previous_index}
        current_ids = [None] * len(current_keypoints_list)

        if self.previous_boxes:
            # 建立IoU矩陣
            iou_matrix = np.zeros((len(current_boxes), len(self.previous_boxes)))

            for i, c_box in enumerate(current_boxes):
                if c_box is None:
                    continue
                for j, p_box in enumerate(self.previous_boxes):
                    if p_box is None:
                        continue
                    iou_matrix[i, j] = self._calculate_iou(c_box, p_box)

            # 貪婪匹配：找到IoU最高的配對
            for _ in range(min(len(current_boxes), len(self.previous_boxes))):
                max_iou = np.max(iou_matrix)

                if max_iou < self.iou_threshold:
                    break

                # 找到最大IoU的索引
                c_idx, p_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

                # 分配ID
                matched_id = self.previous_ids[p_idx]
                current_ids[c_idx] = matched_id
                matched_indices[c_idx] = p_idx

                # 更新ID狀態
                self.id_ages[matched_id] = 0  # 重置年齡
                self.id_hits[matched_id] = self.id_hits.get(matched_id, 0) + 1

                # 避免重複匹配
                iou_matrix[c_idx, :] = -1
                iou_matrix[:, p_idx] = -1

        # 步驟3: 分配新ID給未匹配的骨架
        for i in range(len(current_ids)):
            if current_ids[i] is None and current_boxes[i] is not None:
                current_ids[i] = self.next_id
                self.id_ages[self.next_id] = 0
                self.id_hits[self.next_id] = 1
                self.next_id += 1

        # 步驟4: 更新未匹配的舊ID年齡
        for old_id in self.previous_ids:
            if old_id not in current_ids:
                self.id_ages[old_id] = self.id_ages.get(old_id, 0) + 1

                # 移除過老的ID
                if self.id_ages[old_id] > self.max_age:
                    self._remove_id(old_id)

        # 步驟5: 固定Player1和Player2的ID
        self._stabilize_player_ids(current_ids, current_boxes)

        # 步驟6: 更新歷史數據
        self.previous_boxes = current_boxes
        self.previous_ids = current_ids

        # 步驟7: 構建輸出數據
        tracked_data = []
        for i, kp in enumerate(current_keypoints_list):
            if current_ids[i] is not None and current_boxes[i] is not None:
                # 判斷該ID是否有效（需要達到最小命中次數）
                is_valid = self.id_hits.get(current_ids[i], 0) >= self.min_hits

                if is_valid:
                    tracked_person = TrackedPerson(
                        person_id=current_ids[i],
                        keypoints=kp,
                        bounding_box=current_boxes[i],
                        confidence=current_confidences[i],
                        frame_count=self.id_hits.get(current_ids[i], 0)
                    )
                    tracked_data.append(tracked_person)

        return tracked_data

    def _stabilize_player_ids(self, current_ids: List[int], current_boxes: List):
        """
        穩定Player 1和Player 2的ID分配
        基於位置和追蹤穩定性
        """
        valid_ids = [id for id, box in zip(current_ids, current_boxes) if id is not None and box is not None]

        if len(valid_ids) == 0:
            return

        # 如果尚未分配固定玩家ID，根據x座標分配
        if self.player1_id is None and self.player2_id is None and len(valid_ids) >= 2:
            # 計算所有有效ID的中心x座標
            id_centers = {}
            for i, (id, box) in enumerate(zip(current_ids, current_boxes)):
                if id is not None and box is not None and self.id_hits.get(id, 0) >= self.stable_id_threshold:
                    center_x = (box[0] + box[2]) / 2
                    id_centers[id] = center_x

            # 如果有至少兩個穩定的ID，分配為Player1和Player2
            if len(id_centers) >= 2:
                sorted_ids = sorted(id_centers.items(), key=lambda x: x[1])
                self.player1_id = sorted_ids[0][0]  # 左邊的為Player1
                self.player2_id = sorted_ids[1][0]  # 右邊的為Player2
                print(f"Stabilized Player IDs: Player1={self.player1_id}, Player2={self.player2_id}")

    def _remove_id(self, id_to_remove: int):
        """移除一個ID的所有記錄"""
        if id_to_remove in self.id_ages:
            del self.id_ages[id_to_remove]
        if id_to_remove in self.id_hits:
            del self.id_hits[id_to_remove]
        if id_to_remove in self.id_confidences:
            del self.id_confidences[id_to_remove]

        # 如果是固定玩家ID，也需要清除
        if self.player1_id == id_to_remove:
            self.player1_id = None
        if self.player2_id == id_to_remove:
            self.player2_id = None

    def get_player_keypoints(self, tracked_data: List[TrackedPerson],
                             player_id: int) -> Optional[np.ndarray]:
        """
        從追蹤數據中提取特定玩家的關鍵點

        Args:
            tracked_data: track_and_assign_ids返回的數據
            player_id: 0=Player1, 1=Player2

        Returns:
            關鍵點數組 [25, 3] 或 None
        """
        # 根據player_id獲取對應的追蹤ID
        if player_id == 0:
            target_id = self.player1_id
        elif player_id == 1:
            target_id = self.player2_id
        else:
            return None

        if target_id is None:
            return None

        # 在追蹤數據中查找
        for person in tracked_data:
            if person.person_id == target_id:
                return person.keypoints

        return None

    def get_tracking_stats(self) -> Dict:
        """獲取追蹤統計信息"""
        return {
            "total_active_ids": len(self.id_ages),
            "player1_id": self.player1_id,
            "player2_id": self.player2_id,
            "player1_stable": self.player1_id is not None,
            "player2_stable": self.player2_id is not None,
            "id_hits": dict(self.id_hits),
            "id_ages": dict(self.id_ages)
        }

    def reset(self):
        """重置追蹤器狀態"""
        self.previous_boxes = []
        self.previous_ids = []
        self.next_id = 0
        self.id_ages.clear()
        self.id_hits.clear()
        self.id_confidences.clear()
        self.player1_id = None
        self.player2_id = None
        print("ID Tracker reset")


# ============ 使用範例 ============

def example_usage():
    """
    使用範例：整合到主程式
    """
    from pose_capture.openpose_api import get_keypoints_stream
    from punch_detection import BoxingActionDetector

    # 初始化
    tracker = IDTracker(iou_threshold=0.3)
    player1_detector = BoxingActionDetector()
    player2_detector = BoxingActionDetector()

    try:
        for frame_id, keypoints_data, frame in get_keypoints_stream(video_source=0):

            # 步驟1: 追蹤並分配ID
            tracked_data = tracker.track_and_assign_ids(keypoints_data)

            # 步驟2: 分離Player1和Player2的數據
            player1_kp = tracker.get_player_keypoints(tracked_data, player_id=0)
            player2_kp = tracker.get_player_keypoints(tracked_data, player_id=1)

            # 步驟3: 分別檢測動作
            if player1_kp is not None:
                # 需要將關鍵點包裝成列表形式傳入
                p1_attack_data, p1_defense_data = player1_detector.detect_comprehensive_actions([player1_kp])
                print(f"Player 1: {p1_attack_data}")

            if player2_kp is not None:
                p2_attack_data, p2_defense_data = player2_detector.detect_comprehensive_actions([player2_kp])
                print(f"Player 2: {p2_attack_data}")

            # 步驟4: 顯示追蹤狀態
            stats = tracker.get_tracking_stats()
            print(f"Tracking: {stats}")

    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    print("ID Tracker module loaded")
    print("This module provides person tracking functionality for boxing detection")
    print("\nUsage:")
    print("  from id_tracker import IDTracker")
    print("  tracker = IDTracker()")
    print("  tracked_data = tracker.track_and_assign_ids(keypoints_list)")
