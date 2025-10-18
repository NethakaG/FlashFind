import re
from collections import defaultdict

def summarize_textrank(text, top_sentences=5):
    # ultra-light TextRank-ish frequency summary
    words = re.findall(r"\w+", text.lower())
    freq = defaultdict(int)
    for w in words:
        freq[w] += 1
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    scores = []
    for s in sentences:
        score = sum(freq.get(w.lower(), 0) for w in re.findall(r"\w+", s))
        scores.append((score, s))
    best = [s for _, s in sorted(scores, reverse=True)[:top_sentences]]
    return " ".join(best)

def highlight_hits(sentence, query):
    q = re.escape(query)
    return re.sub(f"({q})", r"<mark>\1</mark>", sentence, flags=re.I)
