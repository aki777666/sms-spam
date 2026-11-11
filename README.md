# 垃圾簡訊偵測器（SMS Spam Detector）

用 TF-IDF + 機器學習分類模型判斷英文簡訊是否為垃圾簡訊，並以 Gradio 做成網頁。

## 專案結構

```
sms-spam/
├── data/
│   └── spam.csv          ← 下載的資料集放這裡
├── spam_model.py         ← 共用邏輯：載入資料、前處理、模型、關鍵字解釋
├── train.py              ← 訓練、比較三種模型、存下最佳模型
├── app.py                ← Gradio 網頁
├── requirements.txt
└── artifacts/            ← 執行 train.py 後自動產生
    ├── model.joblib
    ├── metrics.json
    └── confusion_matrix.png
```

## 步驟

### 1. 安裝套件
```bash
pip install -r requirements.txt
```

### 2. 下載資料集
到 Kaggle 搜尋 **SMS Spam Collection Dataset**，下載 `spam.csv`，放到 `data/` 資料夾。

也可以用 UCI 原始版本（檔名 `SMSSpamCollection`，tab 分隔），程式兩種格式都支援：
```bash
python train.py data/SMSSpamCollection
```

### 3. 訓練模型
```bash
python train.py
```
終端機會顯示：
- 資料筆數與垃圾簡訊比例
- 基準線準確率（全部猜「正常」）
- 三種模型（Naive Bayes、邏輯迴歸、Linear SVM）的 Accuracy、Precision、Recall、F1
- 判錯的訊息

### 4. 啟動網頁
```bash
python app.py
```
瀏覽器打開 http://127.0.0.1:7860

網頁有三個分頁：
1. **線上偵測**：輸入簡訊 → 顯示機率與影響判斷的關鍵字
2. **模型評估**：三種模型比較表、混淆矩陣、為什麼不能只看準確率
3. **模型學到什麼**：最代表垃圾簡訊的字詞、判錯的例子

### 5. 執行單元測試
```bash
pip install pytest
python -m pytest -v
```
測試使用小型自建資料，不需要下載資料集（共 21 項）。

### 系統紀錄
網頁執行時會寫入 `app.log`，每次操作一行，包含操作編號、狀態（ok／empty／too_long／unknown／error）、
字數、垃圾機率與耗時。**不記錄簡訊內容**。

## 部署到 Hugging Face Spaces（SDK因2026費用政策改為Gradio須付費,若要實驗可考慮使用本機環境)

1. 註冊 Hugging Face 帳號 → 右上角 **New Space**
2. SDK 選 **Gradio**，硬體選免費的 CPU
3. 上傳這些檔案：`app.py`、`spam_model.py`、`requirements.txt`，以及整個 `artifacts/` 資料夾
   （不需要上傳 `data/` 和 `train.py`，網頁只用到訓練好的模型）
4. 等待自動建置完成，就會得到公開網址

## 技術重點（報告可用）

| 項目 | 做法 | 原因 |
|---|---|---|
| 前處理 | 網址、金額、電話號碼換成特殊記號 | 這些是垃圾簡訊的重要線索，刪掉會損失資訊 |
| 向量化 | TF-IDF，單字＋雙字詞（bigram） | 捕捉 "free entry"、"call now" 這類片語 |
| 資料切分 | `stratify` 分層抽樣 | 資料不平衡，確保測試集的垃圾比例和整體一致 |
| 不平衡處理 | 邏輯迴歸與 SVM 設 `class_weight="balanced"` | 讓模型不會因為正常訊息多就偏向猜正常 |
| 選模型 | 以 F1 分數，而非準確率 | 準確率會被多數類別灌水 |
| 可解釋性 | 顯示每個字詞的權重 | 讓使用者知道模型為什麼這樣判斷 |

## 限制
- 只支援英文
- 資料集年代較早，現今常見的詐騙型態（假物流、假投資）不一定抓得到
- 用詞巧妙、模仿一般對話的垃圾簡訊容易漏報
