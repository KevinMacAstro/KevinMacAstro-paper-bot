import json
import numpy as np
from openai import OpenAI

client = OpenAI()

with open("data/chunks.json") as f:
    chunks = json.load(f)

embeddings = []

for i, chunk in enumerate(chunks):
    print(f"{i+1}/{len(chunks)}")

    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk["text"]
    )

    embeddings.append(r.data[0].embedding)

embeddings = np.asarray(embeddings, dtype=np.float32)

np.save("data/embeddings.npy", embeddings)

print("Saved", embeddings.shape)


