# OpenPose_development_code
這是供給於學弟妹們持續研究OpenPose的開源初始碼，可發揮各自的創意持續擴充相關模組，punch_detection.py為單人拳擊測試模組，目前開發總共三組拳擊動作--上鉤拳uppercut、直拳straight、鉤拳hook，在此程式上方設置參數run後按下q重置打開攝影機即可開始操作。guard_detection.py目前總共開發三個防禦動作--格擋block、潛避slip、貝殼防禦shell_guard，同樣在punch_detection.py按下run後可測試動作，同時額外提供openpose_api.py提供介面連接溝通OpenPose安裝包(為根據原始碼github提供基準程式再增加一些小函數)，angle.py為本人所編寫之程式，最主要為消除身體比例、與攝影機距離等外部因素而所產生的角度模組，提供一套正規化計算角度流程供OpenPose模組辨識各項動作。未來可往調整參數優化、雙人模組持續建構處理遮擋、2D轉3D以確保直拳等動作更好辨識、硬體效能提升(GPU、攝影機)等方向持續研究，作為教授研究所對於OpenPose合作計畫的一環貢獻。

以下為程式設計之相關流程圖：

<img width="660" height="2048" alt="image" src="https://github.com/user-attachments/assets/617befda-0a10-45dd-a908-9f190afe51e2" />

為angle.py的基本流程圖

<img width="1513" height="3840" alt="Untitled diagram _ Mermaid Chart-2025-10-04-082852" src="https://github.com/user-attachments/assets/905787a4-1f97-4493-be49-8d82e84e7cd9" />


為整體系統執行的流程圖
