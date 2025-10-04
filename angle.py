import math  # 數學函式庫
# import numpy as np

# BODY_25身體關節對應(19~24不知是否會用到，先打上，但可用性很低)
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

# 計算三點夾角（和水平線x軸的夾角） 以相對角度的概念來去實作，並提取各個關鍵點的座標位置，以cos-1=內積/長長的方式提取角度，並透過正規化消除人與攝影機的距離和人身高體型因素
def calculate_normalized_angle(keypoints, joint_indices, person_index):
    # 獲取三個關鍵點(a:頭點 b:中間點 c:末點)
    a_index, b_index, c_index = joint_indices
    # 提取三點座標
    a = keypoints[person_index][a_index][:2] if keypoints[person_index][a_index][2] > 0.5 else [0, 0]
    b = keypoints[person_index][b_index][:2] if keypoints[person_index][b_index][2] > 0.5 else [0, 0]
    c = keypoints[person_index][c_index][:2] if keypoints[person_index][c_index][2] > 0.5 else [0, 0]
    if a[0] == 0 or a[1] == 0 or b[0] == 0 or b[1] == 0 or c[0] == 0 or c[1] == 0:
        return None
    # 計算身體的比例標準（使用肩寬） (也可使用軀幹長度or臀寬)
    shoulder_width = calculate_shoulder_width(keypoints, person_index)
    if shoulder_width is None or shoulder_width < 1e-10:
        return None
    # 規一化向量（除上身體的比例標準）
    ba_norm = [(a[0] - b[0]) / shoulder_width, (a[1] - b[1]) / shoulder_width]
    bc_norm = [(c[0] - b[0]) / shoulder_width, (c[1] - b[1]) / shoulder_width]
    # 計算內積
    dot_product = ba_norm[0] * bc_norm[0] + ba_norm[1] * bc_norm[1]
    # 計算兩向量的norm
    mag_ba = math.sqrt(ba_norm[0] ** 2 + ba_norm[1] ** 2)
    mag_bc = math.sqrt(bc_norm[0] ** 2 + bc_norm[1] ** 2)
    if mag_ba < 1e-10 or mag_bc < 1e-10:
        return None
    # 計算cos角度
    cos_angle = dot_product / (mag_ba * mag_bc)
    # 避免floating point的問題
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    # 計算出弧度，並轉換為弳度
    angle_rad = math.acos(cos_angle)
    angle_deg = angle_rad * 180.0 / math.pi
    return angle_deg

# 計算肩寬作為身體比例的標準
def calculate_shoulder_width(keypoints, person_index):
    left_shoulder = keypoints[person_index][5][:2] if keypoints[person_index][5][2] > 0.5 else [0, 0]
    right_shoulder = keypoints[person_index][2][:2] if keypoints[person_index][2][2] > 0.5 else [0, 0]
    if left_shoulder[0] == 0 or right_shoulder[0] == 0:
        return None
    return math.sqrt((left_shoulder[0] - right_shoulder[0]) ** 2 +(left_shoulder[1] - right_shoulder[1]) ** 2)
