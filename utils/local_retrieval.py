"""Deterministic lexical retrieval for explicitly offline, extractive QA."""
import re
import math
from collections import Counter

STOP = set("which what is are was were the a an and or its it has have had of for in on to with does do how by from this that listed document record focal limitation lowest highest tell me about".split())

def terms(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}

class LocalRetriever:
    def __init__(self, documents, k=4):
        self.documents = documents
        self.k = k

    def invoke(self, query):
        wanted = terms(query)
        if not wanted:
            return []
        frequency = Counter(t for d in self.documents for t in terms(d.page_content))
        def score(doc):
            found = wanted & terms(doc.page_content)
            return sum(math.log(1 + len(self.documents) / frequency[t]) for t in found)
        ranked = sorted(self.documents, key=score, reverse=True)
        return [d for d in ranked[:self.k] if score(d) > 0]
