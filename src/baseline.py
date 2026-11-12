"""Explainable regex baseline for QA-filtered Airbnb reviews.

JSONL records contain review_id, sentiment, aspects and language. Paths are
relative to the project root unless absolute. No network or LLM is used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def words(terms: str) -> re.Pattern:
    """Compile alternatives as Unicode-aware, case-insensitive whole words."""
    return re.compile(r"\b(?:" + terms + r")\b", re.IGNORECASE)


# English and Spanish (the local language of Santiago), plus Vietnamese labels.
ASPECT_PATTERNS = {
    "location": words(r"location|located|neighbou?rhood|nearby|metro|subway|ubicaci[oó]n|ubicado|ubicada|barrio|cerca|vị trí"),
    "cleanliness": words(r"clean|cleanliness|dirty|dirt|spotless|limpio|limpia|limpieza|sucio|sucia|sạch|bẩn"),
    "host": words(r"host|hostess|landlord|anfitri[oó]n|anfitriona|dueño|dueña|propietario|propietaria|chủ nhà"),
    "value": words(r"value|price|priced|cost|affordable|expensive|cheap|precio|valor|caro|cara|barato|barata|giá|đắt|rẻ"),
    "amenities": words(r"amenit(?:y|ies)|wifi|wi-fi|internet|kitchen|bed|bathroom|shower|air conditioning|heating|facilities|cocina|cama|baño|ducha|calefacci[oó]n|servicios|tiện nghi"),
    "noise": words(r"noise|noisy|quiet|loud|traffic|ruido|ruidoso|ruidosa|silencioso|silenciosa|tranquilo|tranquila|ồn|yên tĩnh"),
    "safety": words(r"safe|safety|secure|security|unsafe|seguro|segura|seguridad|peligroso|peligrosa|an toàn|nguy hiểm"),
}

# Stopword counts provide weak evidence; ties and no evidence default to English.
LANGUAGE_WORDS = {
    "en": words(r"the|and|was|with|our|we|very|place|host|stay"),
    "es": words(r"el|la|los|las|una|un|muy|con|para|estaba|lugar|excelente|ubicación"),
    "fr": words(r"le|les|une|des|avec|pour|très|était|nous|séjour"),
    "de": words(r"der|die|das|und|mit|sehr|war|wir|unterkunft"),
    "pt": words(r"o|os|as|uma|muito|com|para|estava|não|localização|estadia"),
    "it": words(r"il|gli|una|con|per|molto|era|siamo|soggiorno"),
    "vi": words(r"và|của|rất|không|được|chúng tôi|nơi|ở|sạch sẽ"),
}
LANGUAGE_MARKERS = {
    "vi": re.compile(r"[ăđơưảạấầẩẫậắằẳẵặếềểễệốồổỗộớờởỡợứừửữựỷỹỵ]", re.I),
    "de": re.compile(r"[ß]", re.I),
    "es": re.compile(r"[ñ¿¡]", re.I),
    "pt": re.compile(r"[ãõ]", re.I),
}

POSITIVE = words(r"good|great|excellent|amazing|wonderful|perfect|clean|quiet|safe|helpful|comfortable|friendly|beautiful|love(?:d)?|bueno|buena|excelente|genial|maravilloso|maravillosa|limpio|limpia|cómodo|cómoda|recomendable")
NEGATIVE = words(r"bad|poor|terrible|awful|dirty|noisy|unsafe|uncomfortable|rude|broken|disappointing|malo|mala|terrible|sucio|sucia|ruidoso|ruidosa|incómodo|incómoda|horrible")
NEGATED_POSITIVE = words(r"(?:not|never|no|hardly|barely|wasn't|weren't|isn't|isnt|wasnt|no es|no estaba)\s+(?:clean|quiet|safe|good|great|helpful|comfortable|bueno|buena|limpio|limpia|cómodo|cómoda|seguro|segura)")


def detect_language(text: str) -> str:
    """Count stopwords and distinctive character evidence, defaulting to en."""
    scores = {lang: len(pattern.findall(text)) for lang, pattern in LANGUAGE_WORDS.items()}
    for lang, pattern in LANGUAGE_MARKERS.items():
        if pattern.search(text):
            scores[lang] += 1
    best = max(scores.values())
    winners = [lang for lang, score in scores.items() if score == best]
    return winners[0] if best and len(winners) == 1 else "en"


def extract_aspects(text: str, aspects: list[str]) -> list[str]:
    """Return configured canonical labels once, in configuration order."""
    return list(dict.fromkeys(label for label in aspects if ASPECT_PATTERNS[label].search(text)))


def classify_sentiment(text: str) -> str:
    """Count hits; a three-to-one lead is dominant, otherwise both means mixed."""
    flipped = list(NEGATED_POSITIVE.finditer(text))
    # Do not count the positive word within a negated phrase twice.
    positive = sum(not any(m.start() >= n.start() and m.end() <= n.end() for n in flipped)
                   for m in POSITIVE.finditer(text))
    negative = len(list(NEGATIVE.finditer(text))) + len(flipped)
    if positive and negative:
        if positive >= 3 * negative:
            return "positive"
        if negative >= 3 * positive:
            return "negative"
        return "mixed"
    if negative:
        return "negative"
    # A no-hit review defaults to positive, as an explicit baseline assumption.
    return "positive"


def _csv_rows(file: Path):
    """Read either plain or gzipped CSV with IDs preserved as strings."""
    opener = gzip.open if file.suffix == ".gz" else open
    with opener(file, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {file}")
        yield reader.fieldnames
        yield from reader


def _project_path(value: str) -> Path:
    return PROJECT_ROOT / Path(value)


def run_baseline(config: dict) -> list[dict]:
    """Analyze unique QA-valid reviews and write the configured JSONL output.

    QA has already selected assigned reviews in raw_dir/phan_cong. Missing
    comment flags are applied across snapshots, matching llm.py; latest review
    contents win for duplicate IDs. This baseline processes all clean reviews.
    """
    llm = config.get("llm") or {}
    aspects = llm.get("aspects")
    if not isinstance(aspects, list) or not aspects or any(not isinstance(a, str) for a in aspects):
        raise ValueError("config['llm']['aspects'] must be a non-empty list of labels")
    unknown = set(aspects) - ASPECT_PATTERNS.keys()
    if unknown:
        raise ValueError(f"No baseline patterns for configured aspects: {sorted(unknown)}")
    snapshots = config.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("config['snapshots'] must be a non-empty list")
    raw = _project_path(config["raw_dir"])
    processed = _project_path(config["processed_dir"])
    output = _project_path(llm.get("baseline_output", str(processed / "baseline_results.jsonl")))
    if output.suffix.lower() != ".jsonl":
        raise ValueError("llm.baseline_output must be a JSONL path")
    reviews, invalid = {}, set()
    for snapshot in snapshots:
        review_path = raw / str(snapshot) / "reviews.csv.gz"
        flag_path = processed / "flagged" / str(snapshot) / "reviews_flags.csv.gz"
        rows = _csv_rows(review_path)
        columns = next(rows)
        id_column = "id" if "id" in columns else "review_id"
        if id_column not in columns or "comments" not in columns:
            raise ValueError(f"{review_path} must contain id (or review_id) and comments")
        for row in rows:
            rid = (row.get(id_column) or "").strip()
            comment = row.get("comments") or ""
            if rid and comment.strip():
                reviews[rid] = comment
        flags = _csv_rows(flag_path)
        columns = next(flags)
        if not {"id", "missing_comments"}.issubset(columns):
            raise ValueError(f"{flag_path} must contain id and missing_comments")
        for flag in flags:
            if (flag.get("missing_comments") or "").strip().lower() in {"true", "1", "t"}:
                invalid.add((flag.get("id") or "").strip())
    results = [
        {"review_id": rid, "sentiment": classify_sentiment(comment),
         "aspects": extract_aspects(comment, aspects), "language": detect_language(comment)}
        for rid, comment in reviews.items() if rid not in invalid
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the rule-based review baseline")
    parser.add_argument("--config", required=True, help="Path to a city YAML configuration")
    args = parser.parse_args()
    import yaml  # Only the CLI needs PyYAML; the entrypoint accepts a loaded dict.
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config YAML must contain a mapping")
    results = run_baseline(config)
    print(f"Processed {len(results)} reviews.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
