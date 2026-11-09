"""共用邏輯：資料載入、建立模型、預測與關鍵字解釋。
train.py 和 app.py 都會 import 這個檔案。"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ---------- 1. 資料載入 ----------
def load_data(path: str) -> pd.DataFrame:
    """支援兩種常見格式，回傳欄位為 label(0/1)、text 的 DataFrame。
    - Kaggle 版 spam.csv：欄位 v1(ham/spam)、v2(內文)，編碼 latin-1
    - UCI 原始版 SMSSpamCollection：以 tab 分隔，無標題列
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, encoding="latin-1")
        df = df.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]
    else:
        df = pd.read_csv(p, sep="\t", header=None, names=["label", "text"],
                         quoting=3, encoding="utf-8")
    df = df.dropna()
    df["label"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


# ---------- 2. 前處理 ----------
def clean_text(text: str) -> str:
    """轉小寫，並把網址、電話、金額換成特殊記號。
    這些是垃圾簡訊的重要線索，所以「保留成記號」而不是刪掉。"""
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"[£$€]\s?\d+[\d,.]*", " moneytoken ", text)
    text = re.sub(r"\d{5,}", " phonetoken ", text)   # 5 位數以上視為電話／短碼
    text = re.sub(r"\d+", " numtoken ", text)
    return text


# ---------- 3. 三種候選模型 ----------
def build_models() -> dict:
    def tfidf():
        return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                               min_df=2, sublinear_tf=True)

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()),
                                 ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(
                                             C=10, class_weight="balanced",
                                             max_iter=2000))]),
        # LinearSVC 本身沒有機率輸出，用 CalibratedClassifierCV 包起來
        "Linear SVM": Pipeline([("tfidf", tfidf()),
                                ("clf", CalibratedClassifierCV(
                                    LinearSVC(C=1, class_weight="balanced"),
                                    cv=3))]),
    }


# ---------- 4. 關鍵字解釋 ----------
def _feature_weights(clf) -> np.ndarray:
    """取得每個字詞『偏向垃圾簡訊』的權重（正值 = 偏垃圾）。"""
    if isinstance(clf, MultinomialNB):
        return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    if isinstance(clf, LogisticRegression):
        return clf.coef_[0]
    if isinstance(clf, CalibratedClassifierCV):
        return np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_],
                       axis=0)
    raise ValueError("不支援的模型")


def explain(model: Pipeline, text: str, top_k: int = 8):
    """回傳這則訊息中影響判斷最大的字詞：[(字詞, 權重), ...]"""
    vec = model.named_steps["tfidf"]
    weights = _feature_weights(model.named_steps["clf"])
    x = vec.transform([text])
    idx = x.nonzero()[1]
    if len(idx) == 0:
        return []
    names = vec.get_feature_names_out()
    contrib = [(names[i], float(weights[i] * x[0, i])) for i in idx]
    contrib.sort(key=lambda t: abs(t[1]), reverse=True)
    return contrib[:top_k]


def top_spam_words(model: Pipeline, k: int = 20):
    """整個模型中最能代表垃圾簡訊的字詞"""
    vec = model.named_steps["tfidf"]
    weights = _feature_weights(model.named_steps["clf"])
    names = vec.get_feature_names_out()
    order = np.argsort(weights)[::-1][:k]
    return [(names[i], float(weights[i])) for i in order]


def predict(model: Pipeline, text: str) -> dict:
    proba = model.predict_proba([text])[0]
    return {"正常訊息 (ham)": float(proba[0]), "垃圾簡訊 (spam)": float(proba[1])}


# ---------- 5. 網頁使用的完整判斷流程（含輸入檢查） ----------
MAX_LEN = 1000   # 一般簡訊遠短於此；限制長度避免過長輸入拖慢服務


def analyze(model: Pipeline, text) -> dict:
    """回傳 {"status": ok/empty/too_long/unknown, ...}。
    不寫入任何檔案，也不保留使用者輸入。"""
    if text is None or not str(text).strip():
        return {"status": "empty", "message": "請輸入一則簡訊。"}
    text = str(text)
    if len(text) > MAX_LEN:
        return {"status": "too_long",
                "message": f"訊息過長（{len(text)} 字元），請限制在 {MAX_LEN} 字元以內。"}
    if model.named_steps["tfidf"].transform([text]).nnz == 0:
        return {"status": "unknown",
                "message": "這則訊息沒有模型認得的英文字詞，無法判斷（本系統只支援英文簡訊）。"}
    probs = predict(model, text)
    return {"status": "ok", "probs": probs,
            "spam_prob": probs["垃圾簡訊 (spam)"], "words": explain(model, text)}
