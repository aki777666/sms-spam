"""訓練與評估：比較三種模型，存下 F1 最好的那個。

用法：
    python train.py                  # 預設讀 data/spam.csv
    python train.py data/SMSSpamCollection
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from spam_model import build_models, load_data

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/spam.csv"
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

# ---------- 1. 載入資料 ----------
df = load_data(DATA_PATH)
n_spam = int(df["label"].sum())
print(f"共 {len(df)} 則（去除重複後），垃圾簡訊 {n_spam} 則，"
      f"佔 {n_spam / len(df):.1%}")

# stratify 讓訓練集和測試集的垃圾比例一致，資料不平衡時很重要
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

# ---------- 2. 基準線：全部猜「正常」 ----------
baseline_acc = accuracy_score(y_test, [0] * len(y_test))
print(f"\n基準線（全部猜正常）準確率：{baseline_acc:.2%}  ← 模型至少要比這好很多")

# ---------- 3. 訓練並比較 ----------
results = []
models = build_models()
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    results.append(r)

print(f"\n{'模型':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
for r in results:
    print(f"{r['model']:<22}{r['accuracy']:>10.3f}{r['precision']:>11.3f}"
          f"{r['recall']:>9.3f}{r['f1']:>8.3f}")

# ---------- 4. 以 F1 選最佳模型 ----------
best = max(results, key=lambda r: r["f1"])
best_model = models[best["model"]]
print(f"\n最佳模型：{best['model']}（F1 = {best['f1']:.3f}）")

tn, fp, fn, tp = [v for row in best["confusion_matrix"] for v in row]
print(f"  誤報（正常被判成垃圾）：{fp} 則")
print(f"  漏報（垃圾沒抓到）：{fn} 則")

# ---------- 5. 錯誤分析：列出判錯的訊息 ----------
pred = best_model.predict(X_test)
wrong = [(t, int(y)) for t, y, p in zip(X_test, y_test, pred) if y != p]
print(f"\n判錯的訊息（前 10 則）：")
for text, y in wrong[:10]:
    tag = "漏報 spam→ham" if y == 1 else "誤報 ham→spam"
    print(f"  [{tag}] {text[:90]}")

# ---------- 6. 存檔 ----------
joblib.dump(best_model, OUT / "model.joblib")

summary = {
    "dataset_size": len(df), "spam_count": n_spam,
    "test_size": len(y_test), "baseline_accuracy": baseline_acc,
    "best_model": best["model"], "results": results,
    "misclassified": [{"text": t, "true_label": "spam" if y else "ham"}
                      for t, y in wrong[:20]],
}
(OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred),
                       display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues",
                                                            colorbar=False)
ax.set_title(f"Confusion Matrix - {best['model']}")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=150)

print(f"\n已儲存：{OUT}/model.joblib、metrics.json、confusion_matrix.png")
