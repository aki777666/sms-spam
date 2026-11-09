"""Gradio 網頁：輸入簡訊 → 判斷是否為垃圾簡訊，並顯示關鍵字與模型比較。

先執行 python train.py 產生 artifacts/，再執行 python app.py
"""

import json
import logging
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from spam_model import analyze, top_spam_words

# ---------- 系統紀錄：只記錄長度、結果、耗時，不記錄簡訊內容 ----------
logging.basicConfig(
    level=logging.WARNING,          # 其他套件（如 httpx）只記錄警告以上，避免雜訊
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log", encoding="utf-8")])
log = logging.getLogger("spam-app")
log.setLevel(logging.INFO)          # 本系統自己的紀錄保留 INFO

ART = Path("artifacts")
try:
    model = joblib.load(ART / "model.joblib")          # 只在啟動時載入一次
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
except FileNotFoundError as e:
    log.error("startup_failed missing_file=%s hint=請先執行 python train.py", e.filename)
    sys.exit(1)
log.info("startup_ok best_model=%s", metrics["best_model"])


def classify(text: str):
    req = uuid.uuid4().hex[:8]                         # 同一次操作的紀錄用同一個編號
    t0 = time.perf_counter()
    try:
        r = analyze(model, text)
    except Exception:
        log.exception("req=%s status=error len=%d", req, len(text or ""))
        return None, [], "### ❌ 系統發生錯誤，請稍後再試。"
    ms = (time.perf_counter() - t0) * 1000
    if r["status"] != "ok":
        log.info("req=%s status=%s len=%d ms=%.1f", req, r["status"], len(text or ""), ms)
        return None, [], f"### ℹ️ {r['message']}"
    spam_p = r["spam_prob"]
    log.info("req=%s status=ok len=%d spam_prob=%.3f ms=%.1f",
             req, len(text), spam_p, ms)
    # HighlightedText 格式：(字詞, 類別)，依權重正負標成 spam / ham
    highlights = [(w, "偏垃圾" if s > 0 else "偏正常") for w, s in r["words"]]
    verdict = ("⚠️ 很可能是垃圾簡訊" if spam_p >= 0.5 else "✅ 看起來是正常訊息")
    return r["probs"], highlights, f"### {verdict}（垃圾機率 {spam_p:.1%}）"


# ---------- 模型比較表 ----------
compare_df = pd.DataFrame([
    {"模型": r["model"], "Accuracy": round(r["accuracy"], 3),
     "Precision": round(r["precision"], 3), "Recall": round(r["recall"], 3),
     "F1": round(r["f1"], 3)}
    for r in metrics["results"]
])
spam_words_df = pd.DataFrame(top_spam_words(model, 20), columns=["字詞", "權重"])
spam_words_df["權重"] = spam_words_df["權重"].round(3)
wrong_df = pd.DataFrame(metrics["misclassified"]).rename(
    columns={"text": "訊息", "true_label": "真實標籤"})

EXAMPLES = [
    "WINNER!! You have been selected to receive a £900 prize reward! "
    "Call 09061701461 to claim now.",
    "Hey, are we still meeting for lunch tomorrow at 12?",
    "URGENT! Your mobile number has won a free entry. Text WIN to 80086 now!",
    "Sorry I'll call you later, I'm in a meeting.",
    "Free ringtone! Reply TONE to 85233 to get yours. Std rates apply.",
]

with gr.Blocks() as demo:
    gr.Markdown("# 📱 垃圾簡訊偵測器\n輸入一則**英文**簡訊，模型會判斷它是不是垃圾簡訊。")

    with gr.Tab("線上偵測"):
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="簡訊內容（請勿貼上含個人資料的訊息）", lines=4,
                                 max_length=1000,
                                 placeholder="貼上一則英文簡訊…")
                btn = gr.Button("偵測", variant="primary")
                gr.Examples(EXAMPLES, inputs=inp)
            with gr.Column():
                verdict = gr.Markdown()
                out_label = gr.Label(label="預測機率")
                out_words = gr.HighlightedText(
                    label="影響判斷的關鍵字",
                    color_map={"偏垃圾": "red", "偏正常": "green"})
        btn.click(classify, inputs=inp, outputs=[out_label, out_words, verdict])
        inp.submit(classify, inputs=inp, outputs=[out_label, out_words, verdict])

    with gr.Tab("模型評估"):
        gr.Markdown(
            f"資料共 **{metrics['dataset_size']}** 則，其中垃圾簡訊 "
            f"**{metrics['spam_count']}** 則（約 "
            f"{metrics['spam_count'] / metrics['dataset_size']:.0%}）。\n\n"
            f"**為什麼不能只看準確率？** 如果模型把所有訊息都猜成「正常」，"
            f"準確率就有 **{metrics['baseline_accuracy']:.1%}**，但一則垃圾簡訊都沒抓到。"
            f"因此這裡以 **F1 分數**（兼顧 Precision 與 Recall）選出最佳模型："
            f"**{metrics['best_model']}**。")
        gr.Dataframe(compare_df, label="三種模型在測試集的表現")
        gr.Image(str(ART / "confusion_matrix.png"), label="最佳模型的混淆矩陣")
        gr.Markdown("- **Precision**：被判成垃圾的訊息中，真的是垃圾的比例（越高 = 越少誤報）\n"
                    "- **Recall**：所有垃圾簡訊中，被成功抓到的比例（越高 = 越少漏報）")

    with gr.Tab("模型學到什麼"):
        gr.Markdown("### 最能代表垃圾簡訊的 20 個字詞")
        gr.Dataframe(spam_words_df)
        gr.Markdown("### 模型判錯的例子\n觀察錯誤可以了解模型的限制。")
        gr.Dataframe(wrong_df)

if __name__ == "__main__":
    demo.launch()
