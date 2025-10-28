import cv2
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

base = Path(__file__).resolve().parents[1]
feature_file = base / "resized_filenames.pkl"

if not feature_file.exists():
    raise FileNotFoundError(f"❌ Could not find feature file: {feature_file}")

with open(feature_file, "rb") as f:
    data = pickle.load(f)

files, _, descs = zip(*data)
N = len(files)
print(f"📂 Loaded {N} frames for feature matching")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
scores = np.zeros((N, N), dtype=np.float32)

print("⚙️  Building feature-match graph...")
for i in tqdm(range(N)):
    if descs[i] is None:
        continue
    for j in range(i + 1, N):
        if descs[j] is None:
            continue
        matches = bf.match(descs[i], descs[j])
        score = len(matches)
        scores[i, j] = scores[j, i] = score

print("✅ Finished computing match scores")

max_score = np.max(scores)
if max_score == 0:
    raise ValueError("No valid feature matches found between frames.")
cost = 1.0 - (scores / (max_score + 1e-8))

out_file = base / "pairwise_cost.npy"
np.save(out_file, cost)

print(f"💾 Pairwise cost matrix saved to: {out_file}")
print("✅ build_graph.py complete.")
