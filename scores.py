"""Inference-time scores for text and responses from a frozen classifier."""
import hashlib
import math
import random
import re

STOPWORDS = set("a an and are as at be but by for from how if in is it of on or so the then to was were while with".split())
GENERIC = set("review positive negative language movie favorably unfavorably describes overall uses emphasizes frames way label text input message sms email".split())
TASK_WORDS = {
    "SST-2": "classify sentiment",
    "Rotten Tomatoes": "classify",
    "TREC": "abbreviation answer classify description entity human location numeric question type",
    "Jigsaw": "classify comment non-toxic toxic",
    "AG News": "article business classify news science science/technology sports technology world",
}
DELETION_RATES = (0, .05, .10, .20, .30, .40, .50, .60)


def tokens(text):
    words = [s.lower().strip(" _-") for s in re.findall(r"[A-Za-z0-9_\-']+", text)]
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def clause_similarity(clause, summary):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not clause.strip() or not summary.strip():
        return 0.0
    try:
        vectors = TfidfVectorizer().fit_transform([clause, summary])
    except ValueError:
        return 0.0
    a, b = vectors[0], vectors[1]
    denominator = math.sqrt(a.multiply(a).sum()) * math.sqrt(b.multiply(b).sum())
    return float(a.multiply(b).sum()) / denominator if denominator else 0.0


def grounding_drift(text, summary, dataset):
    excluded = GENERIC.union(TASK_WORDS[dataset].split())
    if not summary.strip():
        return 1.0
    source = set(tokens(text))
    eligible = [word for word in tokens(summary) if word not in excluded]
    lexical = sum(word in source for word in eligible) / len(eligible) if eligible else 0.0
    marked = re.sub(r"\b(however|although|though|but|yet)\b", "|", text, flags=re.I)
    clauses = [part.strip() for part in re.split(r"[|.;:]", marked) if part.strip()] or [text]
    tfidf = max(clause_similarity(clause, summary) for clause in clauses)
    return max(0.0, min(1.0, 1.0 - max(lexical, tfidf)))


def deletion_views(text):
    spans = list(re.finditer(r"\S+", text))
    if not spans:
        raise ValueError("Expected nonempty text")
    order = list(range(len(spans)))
    key = hashlib.sha256(f"2026\0{text}".encode()).hexdigest()
    random.Random(key).shuffle(order)
    views = []
    for rate in DELETION_RATES:
        removed = set(order[:int(len(spans) * rate)])
        pieces, cursor = [], 0
        for i, span in enumerate(spans):
            pieces.append(text[cursor:span.start()])
            if i not in removed:
                pieces.append(span.group())
            cursor = span.end()
        views.append("".join(pieces) + text[cursor:])
    return views


def bbcal_score(labels):
    """Triggered-positive score from eight parsed labels in deletion order."""
    if isinstance(labels, (str, bytes)) or len(labels) != 8:
        raise ValueError("BBCaL-FPS requires eight ordered labels")
    for i, label in enumerate(labels):
        if not label:
            return 1
        if label != labels[0]:
            return -(i - 1)
    return -7
