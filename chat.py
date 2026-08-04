import json
import os
import sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-5-mini")
TOP_K = 5

with open("data/chunks.json", encoding="utf-8") as f: chunks = json.load(f)

embeddings = np.load("data/embeddings.npy").astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

def retrieve(query, k=TOP_K):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    q = np.asarray(response.data[0].embedding, dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = embeddings @ q
    idx = np.argsort(scores)[::-1][:k]
    return [{"score":float(scores[i]), **chunks[i]} for i in idx]

def answer(query):
    results = retrieve(query)
    context = "\n\n".join([f"[SOURCE {n}: {r['paper']}, pages {r['page_start']}-{r['page_end']}]\n{r['text']}" for n,r in enumerate(results,1)])

    prompt = f"""Answer the question using only the supplied excerpts from Kevin S. McCarthy's published papers.

Rules:
- Do not use outside knowledge.
- If the excerpts do not contain enough information, say so.
- Give a direct scientific answer.
- Refer to sources using labels such as [SOURCE 1].
- Do not invent paper titles, page numbers, numerical results, or conclusions.

EXCERPTS:
{context}

QUESTION:
{query}
"""

    response = client.responses.create(model=ANSWER_MODEL, input=prompt)
    return response.output_text, results

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    text, results = answer(query)

    print("\nANSWER\n")
    print(text)
    print("\nRETRIEVED SOURCES")
    for n,r in enumerate(results,1):
        print(f"[SOURCE {n}] {r['paper']}, pages {r['page_start']}-{r['page_end']}, similarity={r['score']:.3f}")
