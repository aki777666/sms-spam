"""單元測試：python -m pytest -v
使用小型自建測試資料，不需要下載真實資料集。"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spam_model import (MAX_LEN, analyze, build_models, clean_text,  # noqa: E402
                        explain, load_data, predict)

SPAM = ["WINNER! You won a £900 prize. Call 09061701461 now",
        "Free entry to win cash! Text WIN to 80086",
        "URGENT your number won 2000 pounds claim at www.win-now.com",
        "Congratulations you are selected for a free ringtone reply 85233",
        "Claim your free prize now call 09050000301 std rates apply",
        "You have won a guaranteed £1000 cash award call 09066364589"]
HAM = ["Hey are we meeting for lunch at 12?", "Sorry I'll call you later",
       "Ok see you at home tonight", "Can you pick up milk on the way",
       "I'm running 10 minutes late", "Did you finish the homework",
       "Happy birthday! have a great day", "Let's watch a movie this weekend",
       "Call me when you get home", "Thanks for dinner yesterday"]


def make_rows():
    rows = [("spam", f"{t} ref{i}") for i in range(12) for t in SPAM]
    rows += [("ham", f"{t} x{i}") for i in range(10) for t in HAM]
    return rows


@pytest.fixture(scope="module")
def kaggle_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "spam.csv"
    with open(p, "w", encoding="latin-1", newline="") as f:
        w = csv.writer(f)
        w.writerow(["v1", "v2", "Unnamed: 2"])
        for label, text in make_rows():
            w.writerow([label, text, ""])
        w.writerow(["ham", "Sorry I'll call you later x0", ""])  # 刻意放一筆重複
    return p


@pytest.fixture(scope="module")
def models(kaggle_csv):
    df = load_data(str(kaggle_csv))
    ms = build_models()
    for m in ms.values():
        m.fit(df["text"], df["label"])
    return ms


# ---------- 資料載入 ----------
def test_kaggle_label_mapping_and_dedup(kaggle_csv):
    df = load_data(str(kaggle_csv))
    assert set(df["label"]) == {0, 1}
    assert df.loc[df["text"].str.startswith("WINNER"), "label"].eq(1).all()
    assert df.loc[df["text"].str.startswith("Sorry"), "label"].eq(0).all()
    assert df["text"].is_unique                       # 重複訊息已移除
    assert len(df) == 72 + 100


def test_uci_format(tmp_path):
    p = tmp_path / "SMSSpamCollection"
    p.write_text("spam\tFree entry now\nham\tSee you soon\n", encoding="utf-8")
    df = load_data(str(p))
    assert df["label"].tolist() == [1, 0]


# ---------- 前處理 ----------
def test_clean_text_keeps_spam_clues_as_tokens():
    t = clean_text("WIN £900! Call 09061701461 or visit www.x.com, reply 2")
    assert "moneytoken" in t and "phonetoken" in t and "urltoken" in t
    assert "numtoken" in t and "win" in t


# ---------- 預測 ----------
@pytest.mark.parametrize("name", ["Naive Bayes", "Logistic Regression", "Linear SVM"])
def test_probabilities_valid(models, name):
    p = predict(models[name], "Call now to claim your prize")
    assert abs(sum(p.values()) - 1) < 1e-9
    assert all(0 <= v <= 1 for v in p.values())


@pytest.mark.parametrize("name", ["Naive Bayes", "Logistic Regression", "Linear SVM"])
def test_obvious_cases(models, name):
    m = models[name]
    assert predict(m, "URGENT! You won a free prize, call 09061701461")["垃圾簡訊 (spam)"] > 0.5
    assert predict(m, "See you at home for dinner tonight")["垃圾簡訊 (spam)"] < 0.5


def test_same_input_same_output(models):
    m = models["Naive Bayes"]
    a = analyze(m, "Free entry to win cash")
    b = analyze(m, "Free entry to win cash")
    assert a == b


# ---------- 輸入檢查 ----------
@pytest.mark.parametrize("text", [None, "", "   ", "\n"])
def test_empty_input(models, text):
    assert analyze(models["Naive Bayes"], text)["status"] == "empty"


def test_too_long_input(models):
    r = analyze(models["Naive Bayes"], "free " * (MAX_LEN // 5 + 1))
    assert r["status"] == "too_long"


def test_exactly_max_length_is_accepted(models):
    text = ("call me " * MAX_LEN)[:MAX_LEN]
    assert analyze(models["Naive Bayes"], text)["status"] == "ok"


def test_non_english_input(models):
    assert analyze(models["Naive Bayes"], "你好，明天見")["status"] == "unknown"


# ---------- 關鍵字解釋 ----------
@pytest.mark.parametrize("name", ["Naive Bayes", "Logistic Regression", "Linear SVM"])
def test_explain_direction(models, name):
    words = dict(explain(models[name], "free prize call 09061701461"))
    assert words.get("prize", 0) > 0                  # 垃圾字詞應偏向垃圾
    words = dict(explain(models[name], "see you at home"))
    assert words.get("home", 0) < 0                   # 正常字詞應偏向正常


# ---------- 整個訓練流程 ----------
def test_train_script_end_to_end(kaggle_csv, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "train.py"), str(kaggle_csv)],
                   cwd=tmp_path, check=True, capture_output=True)
    art = tmp_path / "artifacts"
    assert (art / "model.joblib").exists() and (art / "confusion_matrix.png").exists()
    m = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
    assert m["best_model"] == max(m["results"], key=lambda r: r["f1"])["model"]
    assert m["spam_count"] == 72
