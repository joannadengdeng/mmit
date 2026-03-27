"""VQA evaluation metrics.

Includes VQA v2 soft accuracy, exact match, token F1, ANLS, and contains match.

References:
  - VQA v2: https://visualqa.org/evaluation.html
  - ANLS: https://arxiv.org/abs/1907.00490
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional


_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "were": "we're", "weve": "we've", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")


def normalize_answer(answer: str) -> str:
    """Apply VQA v2 standard normalisation to an answer string."""
    answer = answer.lower()

    # Expand contractions
    words = answer.split()
    words = [_CONTRACTIONS.get(w, w) for w in words]
    answer = " ".join(words)

    # Remove punctuation
    answer = "".join(ch for ch in answer if ch not in _PUNCT)

    # Remove articles
    words = answer.split()
    words = [w for w in words if w not in _ARTICLES]
    answer = " ".join(words)

    # Normalise whitespace + trailing/leading spaces
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def vqa_accuracy(
    prediction: str,
    ground_truths: List[str],
) -> float:
    """Compute soft VQA accuracy for a single prediction.

    VQA v2 metric: min(#humans_agreeing / 3, 1.0) for exact-match.
    ``ground_truths`` is the list of up to 10 human answers.
    """
    norm_pred = normalize_answer(prediction)
    count = sum(1 for gt in ground_truths if normalize_answer(gt) == norm_pred)
    return min(count / 3.0, 1.0)


def exact_match(
    prediction: str,
    ground_truths: List[str],
) -> float:
    """1.0 if normalised prediction matches any ground truth, else 0.0."""
    norm_pred = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(gt) == norm_pred for gt in ground_truths) else 0.0


def token_f1(
    prediction: str,
    ground_truths: List[str],
) -> float:
    """Token-level F1 between prediction and best-matching ground truth.

    Splits on whitespace after normalisation, computes precision/recall
    on the token multisets, returns the max F1 across all ground truths.
    """
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def anls_score(
    prediction: str,
    ground_truths: List[str],
    threshold: float = 0.5,
) -> float:
    """Average Normalized Levenshtein Similarity (ANLS).

    Used in TextVQA / DocVQA evaluations.  Returns the best NLS across
    all ground truths, with values below *threshold* clamped to 0.
    """
    norm_pred = normalize_answer(prediction)
    best = 0.0
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if not norm_pred and not norm_gt:
            best = max(best, 1.0)
            continue
        max_len = max(len(norm_pred), len(norm_gt))
        if max_len == 0:
            best = max(best, 1.0)
            continue
        dist = _levenshtein(norm_pred, norm_gt)
        nls = 1.0 - dist / max_len
        if nls >= threshold:
            best = max(best, nls)
    return best


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def contains_match(
    prediction: str,
    ground_truths: List[str],
) -> float:
    """1.0 if any ground truth is contained in prediction (or vice versa)."""
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if not norm_gt:
            continue
        if norm_gt in norm_pred or norm_pred in norm_gt:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Registry: metric_name -> (function, display_label)
# ---------------------------------------------------------------------------
METRIC_REGISTRY: Dict[str, tuple] = {
    "exact_match": (exact_match, "Exact Match"),
    "vqa_accuracy": (vqa_accuracy, "VQA Accuracy (soft)"),
    "token_f1": (token_f1, "Token F1"),
    "anls": (anls_score, "ANLS"),
    "contains": (contains_match, "Contains Match"),
}


def aggregate_vqa_accuracy(results: List[Dict]) -> float:
    """Average VQA accuracy over a list of result dicts.

    Each dict must have keys ``"prediction"`` and ``"ground_truths"``
    (a list of strings).
    """
    if not results:
        return 0.0
    total = sum(
        vqa_accuracy(r["prediction"], r["ground_truths"])
        for r in results
    )
    return total / len(results)
