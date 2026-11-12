"""Compare cached LLM and regex baseline predictions with hand labels.

Both models are scored on the same intersection of complete ground truth and
available predictions. Aspect polarity is discarded: the task is to evaluate
which categories are mentioned, using the union of positive and negative.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)
SENTIMENTS = ["positive", "mixed", "negative"]


def _path(value: str | Path) -> Path:
    return PROJECT_ROOT / Path(value)


def _aspects(value) -> list[str]:
    """Read a list, a JSON list, a comma-separated string, or LLM objects."""
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


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation input not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {path}, line {number}") from exc
        return pd.DataFrame(rows)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported input format: {path}")


def _normalize(frame: pd.DataFrame, name: str, labels: list[str]) -> pd.DataFrame:
    """Normalize the actual hand-label, LLM cache, LLM CSV and baseline schemas."""
    frame = frame.copy()
    if "review_id" not in frame:
        raise ValueError(f"{name}: missing review_id column")
    frame["review_id"] = frame["review_id"].fillna("").astype(str).str.strip()
    if (frame["review_id"] == "").any():
        raise ValueError(f"{name}: blank review_id")
    if frame["review_id"].duplicated().any():
        if name == "ground truth":
            raise ValueError("ground truth contains duplicate review IDs")
        LOGGER.warning("%s: duplicate IDs; keeping first prediction, as the LLM CSV export does", name)
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
    invalid = sorted(set(frame["sentiment"]) - set(SENTIMENTS) - {""})
    if invalid:
        raise ValueError(f"{name}: invalid sentiment labels {invalid}")
    if name == "ground truth":
        text_col = next((c for c in ("text", "comments") if c in frame), None)
        if text_col is None:
            raise ValueError("ground truth: missing text/comments column")
        frame["text"] = frame[text_col].fillna("").astype(str)
    return frame


def _llm_input(llm: dict, processed: Path) -> Path:
    """Prefer the configured cache; otherwise use the processed LLM CSV."""
    configured = llm.get("cache") or llm.get("output") or llm.get("output_csv")
    if configured:
        return _path(configured)
    cache = processed / "llm_cache.jsonl"
    return cache if cache.is_file() else processed / "llm_results.csv"


def run_evaluation(config: dict) -> pd.DataFrame:
    """Evaluate matched labeled reviews, write both reports, and return metrics."""
    llm = config.get("llm") or {}
    labels = llm.get("aspects")
    if (not isinstance(labels, list) or not labels or
            any(not isinstance(x, str) or not x or x != x.strip().lower() for x in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError("config['llm']['aspects'] must be a non-empty list of unique lowercase labels")
    processed = _path(config.get("processed_dir", "data/processed"))
    truth_path = _path(config.get("evaluation", {}).get("ground_truth", "data/samples/hand_labeled.csv"))
    llm_path = _llm_input(llm, processed)
    baseline_path = _path(llm.get("baseline_output", processed / "baseline_results.jsonl"))
    truth = _normalize(_read(truth_path), "ground truth", labels)
    llm_df = _normalize(_read(llm_path), "LLM", labels)
    baseline = _normalize(_read(baseline_path), "baseline", labels)
    if len(truth) != 100:
        LOGGER.warning("Expected 100 ground-truth reviews, found %d", len(truth))
    for name, predictions in (("LLM", llm_df), ("baseline", baseline)):
        missing = sorted(set(truth.review_id) - set(predictions.review_id))
        if missing:
            LOGGER.warning("%s missing %d of %d ground-truth IDs: %s", name, len(missing), len(truth), ", ".join(missing))
    complete = truth.sentiment.ne("") & truth.language.ne("") & truth.text.str.strip().ne("")
    if not complete.all():
        LOGGER.warning("Excluding %d incomplete ground-truth rows: %s", int((~complete).sum()), ", ".join(truth.loc[~complete, "review_id"]))
    truth = truth.loc[complete, ["review_id", "text", "sentiment", "aspects", "language"]]
    joined = truth.merge(llm_df[["review_id", "sentiment", "aspects", "language"]], on="review_id", how="inner", suffixes=("_true", "_llm"), validate="one_to_one")
    joined = joined.merge(baseline[["review_id", "sentiment", "aspects", "language"]].rename(columns={"sentiment":"sentiment_baseline", "aspects":"aspects_baseline", "language":"language_baseline"}), on="review_id", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No fully labeled reviews have predictions from both models")
    for model in ("llm", "baseline"):
        missing_values = joined[f"sentiment_{model}"].eq("") | joined[f"language_{model}"].eq("")
        if missing_values.any():
            raise ValueError(f"{model}: missing sentiment or language for IDs: " + ", ".join(joined.loc[missing_values, "review_id"]))
    print(f"Evaluating {len(joined)} common, fully labeled reviews out of {len(complete)} ground-truth IDs.")
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(joined["aspects_true"])
    rows = []
    for model, title in (("llm", "LLM"), ("baseline", "Baseline")):
        s = joined[f"sentiment_{model}"]
        a = mlb.transform(joined[f"aspects_{model}"])
        rows.append({
            "Model": title,
            "Sentiment Acc": accuracy_score(joined.sentiment_true, s),
            "Sentiment Macro-F1": f1_score(joined.sentiment_true, s, labels=SENTIMENTS, average="macro", zero_division=0),
            "Sentiment Weighted-F1": f1_score(joined.sentiment_true, s, labels=SENTIMENTS, average="weighted", zero_division=0),
            "Aspect Micro-F1": f1_score(y_true, a, average="micro", zero_division=0),
            "Aspect Macro-F1": f1_score(y_true, a, average="macro", zero_division=0),
            "Aspect Samples-F1": f1_score(y_true, a, average="samples", zero_division=0),
            "Language Acc": accuracy_score(joined.language_true, joined[f"language_{model}"]),
        })
    metrics = pd.DataFrame(rows)
    display = ["Model", "Sentiment Acc", "Sentiment Macro-F1", "Aspect Micro-F1", "Aspect Macro-F1", "Language Acc"]
    print(metrics[display].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    processed.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(processed / "evaluation_metrics.csv", index=False, encoding="utf-8")
    errors = joined[(joined.sentiment_llm != joined.sentiment_true) | pd.Series([set(a) != set(b) for a, b in zip(joined.aspects_llm, joined.aspects_true)], index=joined.index)].copy()
    errors = errors.rename(columns={"sentiment_true":"true_sentiment", "sentiment_llm":"llm_sentiment", "sentiment_baseline":"baseline_sentiment", "aspects_true":"true_aspects", "aspects_llm":"llm_aspects", "aspects_baseline":"baseline_aspects"})
    columns = ["review_id", "text", "true_sentiment", "llm_sentiment", "baseline_sentiment", "true_aspects", "llm_aspects", "baseline_aspects"]
    for col in ("true_aspects", "llm_aspects", "baseline_aspects"):
        errors[col] = errors[col].map(lambda x: json.dumps(x, ensure_ascii=False))
    errors[columns].to_csv(processed / "llm_error_analysis.csv", index=False, encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LLM and baseline predictions")
    parser.add_argument("--config", required=True, help="City YAML configuration")
    args = parser.parse_args()
    import yaml
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config YAML must contain a mapping")
    run_evaluation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
