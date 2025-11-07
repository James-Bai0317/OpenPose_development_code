# OpenPose_development_code
這是供給於學弟妹們持續研究OpenPose的開源初始碼，可發揮各自的創意持續擴充相關模組，punch_detection.py為單人拳擊測試模組，目前開發總共三組拳擊動作--上鉤拳uppercut、直拳straight、鉤拳hook，在此程式上方設置參數run後按下q重置打開攝影機即可開始操作。guard_detection.py目前總共開發三個防禦動作--格擋block、潛避slip、貝殼防禦shell_guard，同樣在punch_detection.py按下run後可測試動作，同時額外提供openpose_api.py提供介面連接溝通OpenPose安裝包(為根據原始碼github提供基準程式再增加一些小函數)，angle.py為本人所編寫之程式，最主要為消除身體比例、與攝影機距離等外部因素而所產生的角度模組，提供一套正規化計算角度流程供OpenPose模組辨識各項動作。未來可往調整參數優化、雙人模組持續建構處理遮擋、2D轉3D以確保直拳等動作更好辨識、硬體效能提升(GPU、攝影機)等方向持續研究，作為教授研究所對於OpenPose合作計畫的一環貢獻，而我與孫茂翔各自運用OpenPose作各自感興趣的開發，像小弟在下我是做有關於後處理的運用，孫學長是用OpenPose結合Unity先做單一拳種的辨識後做成可視覺化的Unity Game，給與各位參考

總可分為三種方法進行數據調整參數追蹤：1.初期開發調參階段，在終端機與攝影機即可詳細看到參數的呈現結果 2. 實驗數據收集階段： 此時數據收集量大，因此建議將終端機端關閉，改成以攝影機輸出與csv檔的數據收集為主，減少I/O端的端到端延遲時間 3. DEMO呈現結果： 若最後須在專題展或科展展示，須將以上清除乾淨，僅在畫面上讓評分委員看到受試者做該動作時頭上所顯示的動作與置信度即可，乾淨俐落地呈現最終成果

以下為程式設計之相關流程圖：(嘗試先用Mermaid簡單code製作生成，並不完善，最近會將整體程式流程圖使用GitMind自行製作繪出，這是此專案目前最後開發缺乏的初期部分)

<img width="660" height="2048" alt="image" src="https://github.com/user-attachments/assets/617befda-0a10-45dd-a908-9f190afe51e2" />

為angle.py的基本流程圖

<img width="1513" height="3840" alt="Untitled diagram _ Mermaid Chart-2025-10-04-082852" src="https://github.com/user-attachments/assets/905787a4-1f97-4493-be49-8d82e84e7cd9" />

<img width="888" height="2184" alt="image" src="https://github.com/user-attachments/assets/e3bb6ae4-536b-4b1f-8ccf-9fb0cbfc7601" />

為整體系統執行的流程圖
