import json
import sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

with open("data/chunks.json", encoding="utf-8") as f: chunks = json.load(f)

embeddings = np.load("data/embeddings.npy").astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

def search(query, k=5):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    q = np.asarray(response.data[0].embedding, dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = embeddings @ q
    idx = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), chunks[i]) for i in idx]

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    for rank,(score,chunk) in enumerate(search(query),1):
        print(f"\n--- Result {rank}: score={score:.3f} ---")
        print(f"{chunk['paper']}, pages {chunk['page_start']}-{chunk['page_end']}")
        print(chunk["text"][:1200])
