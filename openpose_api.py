import sys
import cv2
import os
import numpy as np
# from sys import platform
print(sys.version)
print(os.environ["PATH"])
# --- Import OpenPose ---
OPENPOSE_ROOT = r"C:\openpose-1.7.0"

pyopenpose_path = os.path.join(OPENPOSE_ROOT, "build", "python", "openpose", "Release")

# 改成 insert(0, ...)，把 OpenPose 路徑放在最前面
if pyopenpose_path not in sys.path:
    sys.path.insert(0, pyopenpose_path)

# DLL 路徑
os.environ["PATH"] = (os.path.join(OPENPOSE_ROOT, "build", "x64", "Release") + ";" + os.path.join(OPENPOSE_ROOT, "bin")
                      + ";" + os.environ["PATH"])

try:
    import pyopenpose as op
    print("✅ pyopenpose 載入成功！")
except Exception as e:
    print("❌ 載入失敗：", e)

# --- OpenPose Parameters ---
params = dict()
params["model_folder"] = "C:/openpose-1.7.0/models/"

# --- Initialize OpenPose ---
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# --- Video Capture ---
video_path = "C:/openpose-1.7.0/examples/media/video.avi"  # Or use 0 for webcam
cap = cv2.VideoCapture(0)

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create datum & process frame
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints = datum.poseKeypoints  # shape: [people, 25, 3]

    if keypoints is not None:
        for person_id in range(keypoints.shape[0]):
            kp_matrix = keypoints[person_id]  # Shape: (25, 3)

            # Save keypoints per frame per person
            filename = f"frame_{frame_id:05d}_person_{person_id+1}.csv"
            np.savetxt(filename, kp_matrix, delimiter=",", header="x,y,confidence", comments='')

    # Optional: Show output
    cv2.imshow("OpenPose Video", datum.cvOutputData)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# --- 用於其他動作識別模組呼叫的函數 ---
def get_keypoints_stream(video_source=0):
    """即時獲取關鍵點資料流的生成器函數 """
    """""
    cap.release()
    cv2.destroyAllWindows()

    Args:
            video_source: 視頻源 (0=攝像頭, 或視頻檔案路徑)

        Yields:
            tuple: (frame_id, keypoints, frame_image)
                - frame_id: 幀編號
                - keypoints: numpy array shape [people, 25, 3] 或 None
                - frame_image: 原始幀圖像
    """
    cap = cv2.VideoCapture(video_source)
    frame_id = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create datum & process frame
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            keypoints = datum.poseKeypoints

            yield frame_id, keypoints, datum.cvOutputData
            frame_id += 1

    finally:
        cap.release()


def initialize_openpose(model_folder_path="C:/openpose-1.7.0/models/"):
    """初始化 OpenPose (如果需要重新配置)"""
    global opWrapper
    params = dict()
    params["model_folder"] = model_folder_path
    opWrapper.configure(params)
    opWrapper.start()

