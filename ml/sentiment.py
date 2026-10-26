# ── ml/sentiment.py ────────────────────────────────────────────────────────
#
# WHAT THIS DOES:
#   Analyzes financial news headlines for a ticker and returns:
#     - aggregate sentiment: POSITIVE / NEGATIVE / NEUTRAL
#     - confidence score  : 0.0 – 1.0
#     - per-headline breakdown
#
# TWO MODES:
#   1. FinBERT (transformers)  → state-of-the-art financial NLP (~400MB download)
#   2. Rule-based fallback     → keyword scoring, no dependencies
#
# HOW FINBERT WORKS:
#   FinBERT is BERT fine-tuned on 10,000 financial news sentences.
#   It maps headlines to [POSITIVE, NEGATIVE, NEUTRAL] probabilities.
#   Much better than generic sentiment models for financial text because
#   words like "bearish", "downgrade", "rally" carry specific meaning.
#
# The rule-based fallback uses a curated financial keyword dictionary
#   and is good enough for demo / testing without GPU/internet.
# ───────────────────────────────────────────────────────────────────────────

import re
from typing import List, Dict, Optional

# ── Fake news headlines per ticker (for demo) ─────────────────────────────────
SAMPLE_HEADLINES: Dict[str, List[str]] = {
    "AAPL": [
        "Apple reports record iPhone sales beating expectations",
        "Apple faces antitrust investigation in EU",
        "Apple stock upgraded to Buy at Goldman Sachs",
        "Supply chain disruptions may impact Apple Q4 guidance",
        "Apple launches AI-powered features across product lineup",
    ],
    "TSLA": [
        "Tesla beats delivery estimates for third consecutive quarter",
        "Tesla recalls 200,000 vehicles over software issue",
        "Tesla Cybertruck demand exceeds initial projections",
        "Elon Musk sells $2 billion worth of Tesla shares",
        "Tesla expands Supercharger network to new markets",
    ],
    "GOOGL": [
        "Alphabet advertising revenue surges on AI-driven growth",
        "Google faces $5 billion antitrust fine in Europe",
        "Google Cloud grows 28% beating analyst estimates",
        "YouTube Premium subscribers reach 100 million milestone",
        "Google DeepMind achieves breakthrough in protein folding",
    ],
    "INFY": [
        "Infosys raises annual revenue guidance on strong deal wins",
        "Infosys wins $1.5 billion multi-year deal from European bank",
        "Infosys margins under pressure due to wage hike cycle",
        "Infosys digital services revenue grows 15% year-on-year",
        "Infosys announces special dividend amid strong cash generation",
    ],
}

# ── Keyword lexicon for rule-based fallback ────────────────────────────────────
POSITIVE_WORDS = {
    "beats", "beat", "record", "surges", "surge", "upgraded", "upgrade",
    "growth", "grows", "grows", "strong", "wins", "win", "breakthrough",
    "milestone", "raises", "raise", "expands", "expand", "profit",
    "bullish", "rally", "recovery", "outperform", "dividend", "positive",
    "exceeds", "exceed", "above", "higher", "better", "improving",
}

NEGATIVE_WORDS = {
    "falls", "fall", "drops", "drop", "investigation", "antitrust",
    "recall", "recalls", "sells", "fine", "pressure", "disruption",
    "issue", "decline", "bearish", "downgrade", "miss", "misses",
    "below", "lower", "worse", "loss", "concern", "risk", "warning",
    "cut", "cuts", "layoff", "layoffs", "debt", "lawsuit",
}


def _rule_based_score(text: str) -> Dict:
    """
    Simple keyword-based sentiment scoring.
    Returns {"label": str, "score": float, "pos": float, "neg": float}
    """
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos_hits = len(words & POSITIVE_WORDS)
    neg_hits = len(words & NEGATIVE_WORDS)
    total = pos_hits + neg_hits + 1e-9

    pos_score = pos_hits / total
    neg_score = neg_hits / total
    neu_score = 1.0 - pos_score - neg_score + 1e-9

    if pos_hits > neg_hits:
        label, confidence = "POSITIVE", round(min(0.55 + 0.1 * pos_hits, 0.92), 2)
    elif neg_hits > pos_hits:
        label, confidence = "NEGATIVE", round(min(0.55 + 0.1 * neg_hits, 0.92), 2)
    else:
        label, confidence = "NEUTRAL", 0.60

    return {
        "label":      label,
        "score":      confidence,
        "positive":   round(pos_score, 3),
        "negative":   round(neg_score, 3),
        "neutral":    round(max(0, 1 - pos_score - neg_score), 3),
    }


# ── FinBERT wrapper ────────────────────────────────────────────────────────────
class SentimentAnalyzer:
    """
    Wraps FinBERT (transformers) with a rule-based fallback.

    Usage:
        analyzer = SentimentAnalyzer()
        analyzer.load()                         # optional — loads FinBERT
        result = analyzer.analyze_ticker("AAPL")
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        self._pipeline = None
        self._loaded   = False

    def load(self) -> bool:
        """
        Attempt to load FinBERT. Falls back to rule-based if unavailable.
        Returns True if FinBERT loaded, False if using fallback.
        """
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model     = self.MODEL_NAME,
                tokenizer = self.MODEL_NAME,
                top_k     = None,       # return all class scores
            )
            self._loaded = True
            print("[SentimentAnalyzer] FinBERT loaded successfully.")
            return True
        except Exception as e:
            print(f"[SentimentAnalyzer] FinBERT unavailable ({e}). Using rule-based fallback.")
            self._loaded = False
            return False

    def _analyze_one(self, text: str) -> Dict:
        """Score a single headline."""
        if self._pipeline:
            try:
                results = self._pipeline(text[:512])[0]   # truncate to BERT max
                best = max(results, key=lambda x: x["score"])
                label_map = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}
                scores = {r["label"].lower(): r["score"] for r in results}
                return {
                    "label":    label_map.get(best["label"].lower(), "NEUTRAL"),
                    "score":    round(best["score"], 3),
                    "positive": round(scores.get("positive", 0), 3),
                    "negative": round(scores.get("negative", 0), 3),
                    "neutral":  round(scores.get("neutral", 0), 3),
                }
            except Exception:
                pass   # fall through to rule-based
        return _rule_based_score(text)

    def analyze_ticker(self, ticker: str,
                       headlines: Optional[List[str]] = None) -> Dict:
        """
        Analyze all headlines for a ticker.
        Returns aggregate sentiment + per-headline breakdown.
        """
        if headlines is None:
            headlines = SAMPLE_HEADLINES.get(ticker, [f"{ticker} stock activity normal"])

        breakdown = []
        for h in headlines:
            result = self._analyze_one(h)
            breakdown.append({"headline": h, **result})

        # Aggregate: weighted vote
        pos = sum(1 for b in breakdown if b["label"] == "POSITIVE")
        neg = sum(1 for b in breakdown if b["label"] == "NEGATIVE")
        neu = sum(1 for b in breakdown if b["label"] == "NEUTRAL")
        total = len(breakdown) or 1

        if pos > neg and pos > neu:
            agg_label = "POSITIVE"
            agg_score = round(pos / total, 2)
        elif neg > pos and neg > neu:
            agg_label = "NEGATIVE"
            agg_score = round(neg / total, 2)
        else:
            agg_label = "NEUTRAL"
            agg_score = round(neu / total, 2)

        # Sentiment score in [-1, +1] range (useful for signal engine)
        sentiment_score = round((pos - neg) / total, 3)

        return {
            "ticker":          ticker,
            "aggregate_label": agg_label,
            "aggregate_score": agg_score,
            "sentiment_score": sentiment_score,   # -1 to +1
            "headlines_count": len(breakdown),
            "positive_count":  pos,
            "negative_count":  neg,
            "neutral_count":   neu,
            "model":           "FinBERT" if self._loaded else "rule-based",
            "breakdown":       breakdown,
        }


# Singleton — import this everywhere
analyzer = SentimentAnalyzer()