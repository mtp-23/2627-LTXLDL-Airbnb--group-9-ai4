"""Đánh giá LLM và baseline so với nhãn tay.

Nhiệm vụ duy nhất: nhận ba bảng dự đoán/nhãn (DataFrame) và tính chỉ số.
Không đọc/ghi tệp, không đọc cấu hình, không in kết quả. Đây là nơi DUY NHẤT
tính accuracy/F1/ma trận nhầm lẫn (kpi.py và viz.py chỉ dùng lại kết quả).

Cả hai mô hình được chấm trên cùng giao của nhãn tay đầy đủ và dự đoán có sẵn.
Cực tính khía cạnh bị bỏ qua: chỉ đánh giá khía cạnh nào được nhắc tới
(hợp của positive và negative).
"""
from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

LOGGER = logging.getLogger(__name__)
EXPECTED_TRUTH = 100


def _aspects(value) -> list[str]:
    """Đọc list, chuỗi JSON, chuỗi phân cách bằng dấu phẩy hoặc list object của LLM."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON aspect list: {value!r}") from exc
        else:
            value = value.split(",")
    if not isinstance(value, list):
        raise ValueError(f"Expected aspect list or comma-separated string, got {value!r}")
    labels = []
    for item in value:
        if isinstance(item, dict):
            if "category" not in item:
                raise ValueError(f"Aspect object has no category: {item!r}")
            item = item["category"]
        if not isinstance(item, str):
            raise ValueError(f"Invalid aspect: {item!r}")
        item = item.strip().lower()
        if item and item not in labels:
            labels.append(item)
    return labels


def _normalize(frame: pd.DataFrame, name: str, labels: list[str], sentiments: list[str]) -> pd.DataFrame:
    """Chuẩn hóa các schema: nhãn tay, cache LLM, CSV LLM và baseline."""
    frame = frame.copy()
    if "review_id" not in frame:
        raise ValueError(f"{name}: missing review_id column")
    frame["review_id"] = frame["review_id"].fillna("").astype(str).str.strip()
    if (frame["review_id"] == "").any():
        raise ValueError(f"{name}: blank review_id")
    if frame["review_id"].duplicated().any():
        if name == "ground truth":
            raise ValueError("ground truth contains duplicate review IDs")
        LOGGER.warning("%s: duplicate IDs; keeping first prediction", name)
        frame = frame.drop_duplicates("review_id", keep="first")
    sentiment_col = next((c for c in ("sentiment", "overall_sentiment", "label") if c in frame), None)
    if not sentiment_col or "language" not in frame:
        raise ValueError(f"{name}: missing sentiment or language column")
    frame["sentiment"] = frame[sentiment_col].fillna("").astype(str).str.strip().str.lower()
    frame["language"] = frame["language"].fillna("").astype(str).str.strip().str.lower()
    if "aspects" in frame:
        frame["aspects"] = frame["aspects"].map(_aspects)
    elif "aspects_positive" in frame and "aspects_negative" in frame:
        frame["aspects"] = [list(dict.fromkeys(_aspects(p) + _aspects(n)))
                            for p, n in zip(frame["aspects_positive"], frame["aspects_negative"])]
    else:
        raise ValueError(f"{name}: missing aspects or positive/negative aspect columns")
    unknown = sorted({a for row in frame["aspects"] for a in row} - set(labels))
    if unknown:
        raise ValueError(f"{name}: unknown aspect labels {unknown}; check config and input")
    invalid = sorted(set(frame["sentiment"]) - set(sentiments) - {""})
    if invalid:
        raise ValueError(f"{name}: invalid sentiment labels {invalid}")
    if name == "ground truth":
        text_col = next((c for c in ("text", "comments") if c in frame), None)
        if text_col is None:
            raise ValueError("ground truth: missing text/comments column")
        frame["text"] = frame[text_col].fillna("").astype(str)
    return frame


def sentiment_report(y_true, y_pred, sentiments: list[str]) -> pd.DataFrame:
    """Ma trận nhầm lẫn, precision/recall/F1 từng lớp, macro-F1 và accuracy (dạng dòng dài)."""
    cm = confusion_matrix(y_true, y_pred, labels=sentiments)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=sentiments, zero_division=0)
    n = len(y_true)
    rows = [{"kind": "confusion", "true_label": t, "llm_label": p, "metric": "review_count",
             "value": int(cm[i, j]), "sample_size": n}
            for i, t in enumerate(sentiments) for j, p in enumerate(sentiments)]
    for i, k in enumerate(sentiments):
        rows += [{"kind": "class", "true_label": k, "llm_label": "", "metric": metric,
                  "value": round(float(v), 4), "sample_size": int(support[i])}
                 for metric, v in (("precision", prec[i]), ("recall", rec[i]), ("f1", f1[i]))]
    rows += [{"kind": "total", "true_label": "", "llm_label": "", "metric": "macro_f1",
              "value": f1_score(y_true, y_pred, labels=sentiments, average="macro", zero_division=0), "sample_size": n},
             {"kind": "total", "true_label": "", "llm_label": "", "metric": "accuracy",
              "value": accuracy_score(y_true, y_pred), "sample_size": n}]
    return pd.DataFrame(rows)


def evaluate(truth: pd.DataFrame, llm: pd.DataFrame, baseline: pd.DataFrame,
             labels: list[str], sentiments: list[str]) -> dict:
    """Chấm LLM và baseline trên nhãn tay.

    Trả về dict: metrics (bảng chỉ số 2 mô hình), errors (bảng lỗi của LLM),
    sentiment_report (cho hình LLM), llm_sentiment_f1 (macro-F1 của LLM), n (số review được chấm).
    """
    if (not isinstance(labels, list) or not labels or
            any(not isinstance(x, str) or not x or x != x.strip().lower() for x in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError("aspects must be a non-empty list of unique lowercase labels")
    truth = _normalize(truth, "ground truth", labels, sentiments)
    llm_df = _normalize(llm, "LLM", labels, sentiments)
    baseline = _normalize(baseline, "baseline", labels, sentiments)
    if len(truth) != EXPECTED_TRUTH:
        LOGGER.warning("Expected %d ground-truth reviews, found %d", EXPECTED_TRUTH, len(truth))
    for name, predictions in (("LLM", llm_df), ("baseline", baseline)):
        missing = sorted(set(truth.review_id) - set(predictions.review_id))
        if missing:
            LOGGER.warning("%s missing %d of %d ground-truth IDs: %s", name, len(missing), len(truth), ", ".join(missing))
    complete = truth.sentiment.ne("") & truth.language.ne("") & truth.text.str.strip().ne("")
    if not complete.all():
        LOGGER.warning("Excluding %d incomplete ground-truth rows: %s", int((~complete).sum()), ", ".join(truth.loc[~complete, "review_id"]))
    truth = truth.loc[complete, ["review_id", "text", "sentiment", "aspects", "language"]]
    joined = truth.merge(llm_df[["review_id", "sentiment", "aspects", "language"]], on="review_id",
                         how="inner", suffixes=("_true", "_llm"), validate="one_to_one")
    joined = joined.merge(
        baseline[["review_id", "sentiment", "aspects", "language"]].rename(
            columns={"sentiment": "sentiment_baseline", "aspects": "aspects_baseline", "language": "language_baseline"}),
        on="review_id", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No fully labeled reviews have predictions from both models")
    for method in ("llm", "baseline"):
        missing_values = joined[f"sentiment_{method}"].eq("") | joined[f"language_{method}"].eq("")
        if missing_values.any():
            raise ValueError(f"{method}: missing sentiment or language for IDs: " + ", ".join(joined.loc[missing_values, "review_id"]))

    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(joined["aspects_true"])
    rows = []
    for method, title in (("llm", "LLM"), ("baseline", "Baseline")):
        s = joined[f"sentiment_{method}"]
        a = mlb.transform(joined[f"aspects_{method}"])
        rows.append({
            "Method": title,
            "Sentiment Acc": accuracy_score(joined.sentiment_true, s),
            "Sentiment Macro-F1": f1_score(joined.sentiment_true, s, labels=sentiments, average="macro", zero_division=0),
            "Sentiment Weighted-F1": f1_score(joined.sentiment_true, s, labels=sentiments, average="weighted", zero_division=0),
            "Aspect Micro-F1": f1_score(y_true, a, average="micro", zero_division=0),
            "Aspect Macro-F1": f1_score(y_true, a, average="macro", zero_division=0),
            "Aspect Samples-F1": f1_score(y_true, a, average="samples", zero_division=0),
            "Language Acc": accuracy_score(joined.language_true, joined[f"language_{method}"]),
        })
    metrics = pd.DataFrame(rows)

    wrong = (joined.sentiment_llm != joined.sentiment_true) | pd.Series(
        [set(a) != set(b) for a, b in zip(joined.aspects_llm, joined.aspects_true)], index=joined.index)
    errors = joined[wrong].rename(columns={
        "sentiment_true": "true_sentiment", "sentiment_llm": "llm_sentiment", "sentiment_baseline": "baseline_sentiment",
        "aspects_true": "true_aspects", "aspects_llm": "llm_aspects", "aspects_baseline": "baseline_aspects"}).copy()
    for col in ("true_aspects", "llm_aspects", "baseline_aspects"):
        errors[col] = errors[col].map(lambda x: json.dumps(x, ensure_ascii=False))
    errors = errors[["review_id", "text", "true_sentiment", "llm_sentiment", "baseline_sentiment",
                     "true_aspects", "llm_aspects", "baseline_aspects"]]

    report = sentiment_report(joined.sentiment_true, joined.sentiment_llm, sentiments)
    return {
        "metrics": metrics,
        "errors": errors,
        "sentiment_report": report,
        "llm_sentiment_f1": float(metrics.loc[metrics.Method == "LLM", "Sentiment Macro-F1"].iloc[0]),
        "n": len(joined),
        "n_truth": len(complete),
    }
