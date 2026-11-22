"""Phương pháp đối chứng không dùng LLM: regex giải thích được cho review.

Nhiệm vụ duy nhất: biến văn bản review thành sentiment, aspects, language bằng
luật. Không đọc/ghi tệp, không đọc cấu hình, không gọi mạng hay LLM.
"""
from __future__ import annotations

import re
from typing import Mapping

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

def analyze(reviews: Mapping[str, str], aspects: list[str]) -> list[dict]:
    """Phân tích {review_id: comments}; trả về [{review_id, sentiment, aspects, language}]."""
    if not isinstance(aspects, list) or not aspects or any(not isinstance(a, str) for a in aspects):
        raise ValueError("aspects must be a non-empty list of labels")
    unknown = set(aspects) - ASPECT_PATTERNS.keys()
    if unknown:
        raise ValueError(f"No baseline patterns for configured aspects: {sorted(unknown)}")
    return [
        {"review_id": rid, "sentiment": classify_sentiment(text),
         "aspects": extract_aspects(text, aspects), "language": detect_language(text)}
        for rid, text in reviews.items()
    ]
