# ── ml/rag.py ──────────────────────────────────────────────────────────────
#
# WHAT THIS DOES (RAG = Retrieval-Augmented Generation, retrieval half):
#   Instead of scoring 5 hardcoded headlines, the system now SEARCHES a real
#   news corpus (Benzinga headlines, 2009-2020) and hands the most relevant
#   recent headlines for a stock to FinBERT.
#
# HOW IT WORKS:
#   1. INDEX (once): read news/analyst_ratings_processed.csv, keep only our
#      4 stocks (~5,000 headlines), turn every headline into a 384-number
#      "meaning vector" with a small embedding model (all-MiniLM-L6-v2).
#      Vectors are cached in news/rag_cache.npz so later starts are instant.
#   2. RETRIEVE (every request): take the stock's most recent headlines,
#      embed a question like "earnings, guidance, analyst rating changes,
#      risks", and return the headlines whose meaning is closest to it
#      (cosine similarity, plain numpy - no FAISS, nothing for Windows to block).
#
# NO NEW INSTALLS: uses torch + transformers, which you already have.
# ───────────────────────────────────────────────────────────────────────────

import csv
import os
from typing import Dict, List, Optional

import numpy as np

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_DIR   = os.path.join(ROOT, "news")
CSV_PATH   = os.path.join(NEWS_DIR, "analyst_ratings_processed.csv")
CACHE_PATH = os.path.join(NEWS_DIR, "rag_cache.npz")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# our ticker -> tickers used inside the dataset
TICKER_ALIASES: Dict[str, List[str]] = {
    "AAPL":  ["AAPL"],
    "TSLA":  ["TSLA"],
    "GOOGL": ["GOOGL", "GOOG"],
    "INFY":  ["INFY"],
}

DEFAULT_QUERY = ("earnings results, guidance, analyst rating or price target "
                 "change, risks, major company news")

# Words that show a headline is really ABOUT the company (many headlines are
# tagged with a ticker but are market-wide roundups).
COMPANY_TERMS: Dict[str, List[str]] = {
    "AAPL":  ["apple", "aapl", "iphone", "ipad", "tim cook"],
    "TSLA":  ["tesla", "tsla", "musk", "cybertruck", "model 3"],
    "GOOGL": ["alphabet", "google", "googl", "goog", "youtube", "waymo"],
    "INFY":  ["infosys", "infy"],
}
COMPANY_NAME = {"AAPL": "Apple", "TSLA": "Tesla", "GOOGL": "Alphabet Google",
                "INFY": "Infosys"}
MENTION_BONUS = 0.30     # added to the similarity when the headline names the company


def _mean_pool_embedder(model_name: str = EMBED_MODEL):
    """Return embed(texts) -> L2-normalised float32 matrix, using only
    torch + transformers (no sentence-transformers package needed)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    def embed(texts: List[str], batch: int = 64) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tok(texts[i:i + batch], padding=True, truncation=True,
                          max_length=64, return_tensors="pt")
                hidden = model(**enc).last_hidden_state            # [b, t, 384]
                mask   = enc["attention_mask"].unsqueeze(-1).float()
                vec    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vec    = torch.nn.functional.normalize(vec, dim=1)
                out.append(vec.numpy().astype(np.float32))
        return np.vstack(out)

    return embed


class NewsRAG:
    def __init__(self, csv_path: str = CSV_PATH, cache_path: str = CACHE_PATH):
        self.csv_path   = csv_path
        self.cache_path = cache_path
        self.ready      = False
        self.error: Optional[str] = None
        self._embed     = None
        self._titles: np.ndarray = np.array([])
        self._dates:  np.ndarray = np.array([])
        self._stocks: np.ndarray = np.array([])
        self._emb:    np.ndarray = np.zeros((0, 1), dtype=np.float32)

    # ── indexing ─────────────────────────────────────────────────────────────
    def _read_corpus(self):
        """Filter the 1.4M-row CSV down to our stocks, drop duplicate titles."""
        alias_to_ticker = {a: t for t, al in TICKER_ALIASES.items() for a in al}
        seen, rows = set(), []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tk = alias_to_ticker.get(r["stock"])
                if not tk:
                    continue
                title = (r["title"] or "").strip()
                key = (tk, title.lower())
                if not title or key in seen:
                    continue
                seen.add(key)
                rows.append((tk, r["date"][:10], title))
        rows.sort(key=lambda x: x[1], reverse=True)       # newest first
        return rows

    def load(self, embedder=None) -> Dict:
        """Build (or load from cache) the index. Safe to call repeatedly."""
        if self.ready:
            return self.status()
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(
                    f"News file not found: {self.csv_path}. Put "
                    f"analyst_ratings_processed.csv inside the 'news' folder.")

            if os.path.exists(self.cache_path):
                z = np.load(self.cache_path, allow_pickle=False)
                self._stocks, self._dates = z["stocks"], z["dates"]
                self._titles, self._emb   = z["titles"], z["emb"]
            else:
                rows = self._read_corpus()
                if not rows:
                    raise ValueError("No headlines for AAPL/TSLA/GOOGL/INFY in the CSV.")
                self._stocks = np.array([r[0] for r in rows])
                self._dates  = np.array([r[1] for r in rows])
                self._titles = np.array([r[2] for r in rows])
                embed = embedder or _mean_pool_embedder()
                self._emb = embed(list(self._titles))
                np.savez_compressed(self.cache_path, stocks=self._stocks,
                                    dates=self._dates, titles=self._titles,
                                    emb=self._emb)
            self._embed = embedder
            self.ready, self.error = True, None
        except Exception as e:                      # never crash the server
            self.ready, self.error = False, str(e)
        return self.status()

    def status(self) -> Dict:
        per = {t: int((self._stocks == t).sum()) for t in TICKER_ALIASES} \
              if self.ready else {}
        return {"ready": self.ready, "error": self.error,
                "headlines_indexed": int(len(self._titles)) if self.ready else 0,
                "per_ticker": per}

    # ── retrieval ────────────────────────────────────────────────────────────
    def retrieve(self, ticker: str, query: str = DEFAULT_QUERY,
                 k: int = 8, recent_window: int = 400) -> Optional[List[Dict]]:
        """
        Top-k headlines for `ticker`: look at its `recent_window` newest
        headlines, then keep the k closest in meaning to `query`.
        Returns None if the index is not ready or the ticker has no news
        (caller then falls back to the sample headlines).
        """
        if not self.ready:
            return None
        idx = np.where(self._stocks == ticker)[0]          # already newest-first
        if len(idx) == 0:
            return None
        idx = idx[:recent_window]

        if self._embed is None:
            self._embed = _mean_pool_embedder()
        q = self._embed([f"{COMPANY_NAME.get(ticker, ticker)}: {query}"])[0]
        sims = self._emb[idx] @ q                          # cosine (vectors are unit length)

        # bonus for headlines that actually mention the company
        terms = COMPANY_TERMS.get(ticker, [ticker.lower()])
        titles = [str(self._titles[i]) for i in idx]
        about  = np.array([any(t in ti.lower() for t in terms) for ti in titles])
        scores = sims + MENTION_BONUS * about

        order = np.argsort(-scores)[:k]
        hits = [{"title": titles[i],
                 "date":  str(self._dates[idx[i]]),
                 "similarity": round(float(sims[i]), 3),
                 "about_company": bool(about[i])} for i in order]
        hits.sort(key=lambda h: h["date"], reverse=True)   # show newest first
        return hits


# Singleton
rag = NewsRAG()