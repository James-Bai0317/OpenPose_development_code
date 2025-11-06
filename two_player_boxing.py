# two_player_boxing.py - 雙人拳擊檢測整合主程式

import cv2
import time
import math
import csv
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# 導入現有模組
from pose_capture.openpose_api import get_keypoints_stream
from punch_detection import BoxingActionDetector, ComprehensiveVisualizer
from guard_detection import BoxingDefenseDetector
from id_tracker import IDTracker
from angle import calculate_shoulder_width, NECK, MID_HIP


@dataclass
class BattleResult:
    """戰鬥結果數據"""
    frame_id: int
    timestamp: float
    player1_action: str
    player2_action: str
    distance: float
    is_in_range: bool
    battle_outcome: str  # "P1_HIT", "P2_HIT", "P1_BLOCKED", "P2_BLOCKED", "BOTH_ATTACK", "NO_CONTACT"
    p1_confidence: float
    p2_confidence: float


class TwoPlayerBattleSystem:
    """雙人拳擊戰鬥系統"""

    def __init__(self, effective_distance_multiplier=2.5):
        """
        初始化雙人戰鬥系統

        Args:
            effective_distance_multiplier: 有效攻擊距離 = 肩寬 × 此係數
        """
        # 初始化追蹤器
        self.tracker = IDTracker(iou_threshold=0.3, max_age=30, min_hits=3)

        # 初始化檢測器（每個玩家一個實例）
        self.player1_attack_detector = BoxingActionDetector()
        self.player2_attack_detector = BoxingActionDetector()

        # 防禦檢測器（共用）
        self.defense_detector = BoxingDefenseDetector()

        # 可視化器
        self.visualizer = ComprehensiveVisualizer(show_skeleton=True, show_debug=True)

        # 戰鬥邏輯參數
        self.effective_distance_multiplier = effective_distance_multiplier

        # 動作歷史（用於時序判斷）
        self.action_history = []  # [(frame_id, player_id, action_type, timestamp)]
        self.history_window = 15  # 保留最近15幀的歷史

        # 統計信息
        self.frame_count = 0
        self.battle_events = []

        print("=== Two Player Battle System Initialized ===")
        print(f"Effective Distance: {effective_distance_multiplier}× shoulder width")

    @staticmethod
    def get_center_point(keypoints) -> Optional[Tuple[float, float]]:
        """
        獲取人物中心點（用於距離計算）
        使用NECK和MID_HIP的中點
        """
        try:
            if keypoints[NECK][2] > 0.3 and keypoints[MID_HIP][2] > 0.3:
                neck = keypoints[NECK][:2]
                mid_hip = keypoints[MID_HIP][:2]
                return (neck[0] + mid_hip[0]) / 2, (neck[1] + mid_hip[1]) / 2
        except (IndexError, TypeError):
            pass
        return None

    def calculate_players_distance(self, p1_kp, p2_kp) -> Optional[float]:
        """計算兩個玩家之間的距離"""
        p1_center = self.get_center_point(p1_kp)
        p2_center = self.get_center_point(p2_kp)

        if p1_center is None or p2_center is None:
            return None

        distance = math.sqrt(
            (p1_center[0] - p2_center[0]) ** 2 +
            (p1_center[1] - p2_center[1]) ** 2
        )
        return distance

    def check_battle_logic(self, p1_action_data, p2_action_data,
                           p1_defense_data, p2_defense_data,
                           p1_kp, p2_kp, frame_id, timestamp) -> BattleResult:
        """
        核心戰鬥邏輯判斷

        Returns:
            BattleResult對象，包含戰鬥結果
        """
        # 1. 計算距離
        distance = self.calculate_players_distance(p1_kp, p2_kp)

        # 2. 判斷是否在有效距離內
        shoulder_width = calculate_shoulder_width([p1_kp], 0)
        if shoulder_width is None or shoulder_width <= 0:
            shoulder_width = 100  # 預設值

        effective_distance = self.effective_distance_multiplier * shoulder_width
        is_in_range = distance is not None and distance <= effective_distance

        # 3. 提取動作類型
        p1_action = "idle"
        p1_conf = 0.0
        if p1_action_data and len(p1_action_data.players) > 0:
            player = p1_action_data.players[0]
            if player.is_attacking:
                p1_action = player.punch_type
                p1_conf = player.confidence

        p2_action = "idle"
        p2_conf = 0.0
        if p2_action_data and len(p2_action_data.players) > 0:
            player = p2_action_data.players[0]
            if player.is_attacking:
                p2_action = player.punch_type
                p2_conf = player.confidence

        # 4. 檢查防禦動作
        p1_defending = False
        p1_defense_type = None
        if p1_defense_data and len(p1_defense_data.players) > 0:
            player = p1_defense_data.players[0]
            if player.is_defending or player.is_dodging:
                p1_defending = True
                p1_defense_type = player.defense_type

        p2_defending = False
        p2_defense_type = None
        if p2_defense_data and len(p2_defense_data.players) > 0:
            player = p2_defense_data.players[0]
            if player.is_defending or player.is_dodging:
                p2_defending = True
                p2_defense_type = player.defense_type

        # 5. 戰鬥結果判定
        battle_outcome = "NO_CONTACT"

        if not is_in_range:
            battle_outcome = "TOO_FAR"
        else:
            # P1攻擊P2
            if p1_action in ['straight', 'hook', 'uppercut']:
                if p2_defending and self._check_defense_effective(p1_action, p2_defense_type):
                    battle_outcome = "P1_ATTACK_P2_BLOCKED"
                else:
                    battle_outcome = "P1_HIT_P2"

            # P2攻擊P1
            elif p2_action in ['straight', 'hook', 'uppercut']:
                if p1_defending and self._check_defense_effective(p2_action, p1_defense_type):
                    battle_outcome = "P2_ATTACK_P1_BLOCKED"
                else:
                    battle_outcome = "P2_HIT_P1"

            # 雙方同時攻擊
            elif (p1_action in ['straight', 'hook', 'uppercut'] and
                  p2_action in ['straight', 'hook', 'uppercut']):
                battle_outcome = "BOTH_ATTACK"

        # 6. 更新動作歷史
        self._update_action_history(frame_id, timestamp, p1_action, p2_action)

        # 7. 構建結果
        result = BattleResult(
            frame_id=frame_id,
            timestamp=timestamp,
            player1_action=p1_action,
            player2_action=p2_action,
            distance=distance if distance else 0.0,
            is_in_range=is_in_range,
            battle_outcome=battle_outcome,
            p1_confidence=p1_conf,
            p2_confidence=p2_conf
        )

        # 記錄戰鬥事件
        if battle_outcome not in ["NO_CONTACT", "TOO_FAR"]:
            self.battle_events.append(result)
            print(f"[Battle Event] Frame {frame_id}: {battle_outcome} "
                  f"(P1: {p1_action}, P2: {p2_action}, Dist: {distance:.1f}px)")

        return result

    @staticmethod
    def _check_defense_effective(attack_type: str, defense_type: str) -> bool:
        """
        檢查防禦是否有效對抗攻擊

        對應關係:
        - straight -> block
        - hook -> slip
        - uppercut -> shell_guard
        """
        defense_map = {
            'straight': 'block',
            'hook': 'slip',
            'uppercut': 'shell_guard'
        }

        expected_defense = defense_map.get(attack_type)
        return defense_type == expected_defense

    def _update_action_history(self, frame_id: int, timestamp: float,
                               p1_action: str, p2_action: str):
        """更新動作歷史"""
        self.action_history.append((frame_id, 0, p1_action, timestamp))
        self.action_history.append((frame_id, 1, p2_action, timestamp))

        # 保持歷史窗口大小
        if len(self.action_history) > self.history_window * 2:
            self.action_history = self.action_history[-self.history_window * 2:]

    def process_frame(self, frame_id: int, keypoints_data, frame):
        """
        處理單幀數據

        Returns:
            (result_frame, battle_result)
        """
        self.frame_count += 1
        timestamp = time.time()

        # 步驟1: 追蹤並分配ID
        tracked_data = self.tracker.track_and_assign_ids(keypoints_data)

        # 步驟2: 提取Player1和Player2的關鍵點
        player1_kp = self.tracker.get_player_keypoints(tracked_data, player_id=0)
        player2_kp = self.tracker.get_player_keypoints(tracked_data, player_id=1)

        # 初始化結果
        p1_attack_data = None
        p1_defense_data = None
        p2_attack_data = None
        p2_defense_data = None
        battle_result = None

        # 步驟3: 檢測Player1動作
        if player1_kp is not None:
            p1_attack_data, p1_defense_data = self.player1_attack_detector.detect_comprehensive_actions([player1_kp])

        # 步驟4: 檢測Player2動作
        if player2_kp is not None:
            p2_attack_data, p2_defense_data = self.player2_attack_detector.detect_comprehensive_actions([player2_kp])

        # 步驟5: 戰鬥邏輯判斷
        if player1_kp is not None and player2_kp is not None:
            battle_result = self.check_battle_logic(
                p1_attack_data, p2_attack_data,
                p1_defense_data, p2_defense_data,
                player1_kp, player2_kp,
                frame_id, timestamp
            )

        # 步驟6: 可視化
        result_frame = self._visualize_battle(frame, tracked_data, battle_result)

        return result_frame, battle_result

    def _visualize_battle(self, frame, tracked_data,
                          battle_result):
        """可視化戰鬥場景"""
        result_frame = frame.copy()
        height, width = frame.shape[:2]

        # 繪製追蹤信息
        for person in tracked_data:
            color = (0, 255, 0) if person.person_id == self.tracker.player1_id else (255, 0, 0)
            box = person.bounding_box

            # 繪製邊界框
            cv2.rectangle(result_frame,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 2)

            # 顯示Player標籤
            player_label = "P1" if person.person_id == self.tracker.player1_id else "P2"
            cv2.putText(result_frame, player_label,
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 繪製戰鬥結果
        if battle_result:
            self._draw_battle_result(result_frame, battle_result, width)

        # 繪製系統狀態
        self._draw_system_status(result_frame, height)

        return result_frame

    @staticmethod
    def _draw_battle_result(frame, battle_result, width):
        """繪製戰鬥結果信息"""
        # 背景框
        cv2.rectangle(frame, (width - 350, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 350, 10), (width - 10, 150), (255, 255, 255), 2)

        y_pos = 35

        # 戰鬥結果
        outcome_color = (0, 255, 0)
        if "HIT" in battle_result.battle_outcome:
            outcome_color = (0, 0, 255)
        elif "BLOCKED" in battle_result.battle_outcome:
            outcome_color = (0, 255, 255)

        cv2.putText(frame, f"Result: {battle_result.battle_outcome}",
                    (width - 340, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, outcome_color, 2)
        y_pos += 25

        # 玩家動作
        cv2.putText(frame, f"P1: {battle_result.player1_action} ({battle_result.p1_confidence:.2f})",
                    (width - 340, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20

        cv2.putText(frame, f"P2: {battle_result.player2_action} ({battle_result.p2_confidence:.2f})",
                    (width - 340, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20

        # 距離信息
        range_text = "IN RANGE" if battle_result.is_in_range else "OUT OF RANGE"
        range_color = (0, 255, 0) if battle_result.is_in_range else (128, 128, 128)
        cv2.putText(frame, f"Distance: {battle_result.distance:.1f}px ({range_text})",
                    (width - 340, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, range_color, 1)

    def _draw_system_status(self, frame, height):
        """繪製系統狀態"""
        stats = self.tracker.get_tracking_stats()

        cv2.rectangle(frame, (10, height - 100), (300, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, height - 100), (300, height - 10), (255, 255, 255), 2)

        cv2.putText(frame, f"Frame: {self.frame_count}",
                    (20, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"P1 ID: {stats['player1_id']} {'(Stable)' if stats['player1_stable'] else '(Unstable)'}",
                    (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"P2 ID: {stats['player2_id']} {'(Stable)' if stats['player2_stable'] else '(Unstable)'}",
                    (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Battle Events: {len(self.battle_events)}",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def get_statistics(self) -> Dict:
        """獲取統計信息"""
        return {
            "total_frames": self.frame_count,
            "total_battle_events": len(self.battle_events),
            "tracking_stats": self.tracker.get_tracking_stats()
        }


def run_two_player_detection(camera_index=0, save_log=True):
    """運行雙人拳擊檢測"""
    print("=== Two Player Boxing Detection System ===")

    # 初始化系統
    battle_system = TwoPlayerBattleSystem(effective_distance_multiplier=2.5)

    # 日誌設定
    log_file = None
    log_writer = None

    if save_log:
        log_filename = f"two_player_battle_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        log_file = open(log_filename, 'w', newline='', encoding='utf-8')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "timestamp", "frame_id", "player1_action", "player2_action",
            "distance", "is_in_range", "battle_outcome",
            "p1_confidence", "p2_confidence"
        ])
        print(f"Logging to: {log_filename}")

    try:
        for frame_id, keypoints_data, frame in get_keypoints_stream(video_source=camera_index):

            # 處理幀
            result_frame, battle_result = battle_system.process_frame(frame_id, keypoints_data, frame)

            # 記錄日誌
            if save_log and battle_result and log_writer:
                log_writer.writerow([
                    battle_result.timestamp,
                    battle_result.frame_id,
                    battle_result.player1_action,
                    battle_result.player2_action,
                    battle_result.distance,
                    battle_result.is_in_range,
                    battle_result.battle_outcome,
                    battle_result.p1_confidence,
                    battle_result.p2_confidence
                ])

            # 顯示結果
            cv2.imshow('Two Player Boxing Detection', result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                battle_system.tracker.reset()
                print("Tracker reset")

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        if log_file:
            log_file.close()
            print(f"Log saved")

        cv2.destroyAllWindows()

        # 顯示統計
        stats = battle_system.get_statistics()
        print("\n=== Final Statistics ===")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Battle Events: {stats['total_battle_events']}")
        print(f"Tracking: {stats['tracking_stats']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Two Player Boxing Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--no-log', action='store_true', help='Disable logging')

    args = parser.parse_args()

    run_two_player_detection(
        camera_index=args.camera,
        save_log=not args.no_log
    )
