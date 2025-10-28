import numpy as np
import pickle
import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
coarse_file = base / "coarse_order.pkl"
cost_file = base / "pairwise_cost.npy"

if not coarse_file.exists() or not cost_file.exists():
    raise FileNotFoundError("❌ Missing input files (coarse_order.pkl or pairwise_cost.npy).")

with open(coarse_file, "rb") as f:
    coarse_order = pickle.load(f)

cost = np.load(cost_file)
N = len(coarse_order)

print(f"📂 Loaded coarse order ({N} frames) and cost matrix {cost.shape}")

final_order = []
used = set()
current = 0
final_order.append(current)
used.add(current)

print("⚙️  Assembling final sequence using ORB cost minimization...")
for _ in range(N - 1):
    next_idx = np.argmin(cost[current])
    while next_idx in used:
        cost[current, next_idx] = np.inf
        next_idx = np.argmin(cost[current])
    final_order.append(next_idx)
    used.add(next_idx)
    current = next_idx

ordered_filenames = [coarse_order[i] for i in final_order]

out_file = base / "final_sequence.json"
with open(out_file, "w") as f:
    json.dump(ordered_filenames, f, indent=2)

print(f"🎯 Final frame sequence saved to: {out_file}")
