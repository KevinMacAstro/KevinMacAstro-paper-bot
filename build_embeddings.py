import json
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()
client = OpenAI()

with open("data/chunks.json", encoding="utf-8") as f: chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
batch_size = 10
embeddings = []

for start in range(0, len(texts), batch_size):
    batch = texts[start:start+batch_size]
    while True:
        try:
            response = client.embeddings.create(model="text-embedding-3-small", input=batch)
            embeddings.extend(item.embedding for item in response.data)
            print(f"{min(start+len(batch),len(texts))}/{len(texts)}")
            break
        except RateLimitError:
            print("Rate limit reached; waiting 60 seconds...")
            time.sleep(60)

embeddings = np.asarray(embeddings, dtype=np.float32)
np.save("data/embeddings.npy", embeddings)
print(f"Saved data/embeddings.npy with shape {embeddings.shape}")

