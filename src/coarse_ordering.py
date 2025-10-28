import numpy as np
import pickle
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from pathlib import Path

# ------------------------------------------------------
# Custom Chi-square distance for histogram comparison
# ------------------------------------------------------
def chi2_distance(a, b):
    eps = 1e-10
    return 0.5 * np.sum(((a - b) ** 2) / (a + b + eps))

# ------------------------------------------------------
# Load precomputed features
# ------------------------------------------------------
base = Path(__file__).resolve().parents[1]
feature_file = base / "resized_filenames.pkl"

if not feature_file.exists():
    raise FileNotFoundError(f"❌ Could not find feature file: {feature_file}")

with open(feature_file, "rb") as f:
    data = pickle.load(f)

print(f"📂 Loaded {len(data)} feature entries from {feature_file.name}")

# ------------------------------------------------------
# Separate file paths, histograms, and descriptors
# ------------------------------------------------------
files, hists, descs = [], [], []
for entry in data:
    if len(entry) >= 3 and isinstance(entry[1], np.ndarray):
        files.append(entry[0])
        hists.append(entry[1])
        descs.append(entry[2])
    elif len(entry) == 2 and isinstance(entry[1], np.ndarray):
        files.append(entry[0])
        hists.append(entry[1])
        descs.append(None)

print(f"⚙️  Valid histograms: {len(hists)} / {len(data)}")

# ------------------------------------------------------
# Stack histograms into a 2D NumPy array
# ------------------------------------------------------
hists = np.stack([np.ravel(h).astype(np.float32) for h in hists])
print("✅ Histogram matrix shape:", hists.shape)

# ------------------------------------------------------
# Compute pairwise Chi-square distance matrix
# ------------------------------------------------------
print("⚙️  Computing Chi-square distance matrix...")
D = cdist(hists, hists, metric=chi2_distance)
print("✅ Distance matrix shape:", D.shape)

# ------------------------------------------------------
# Convert distances to similarity matrix (Gaussian kernel)
# ------------------------------------------------------
sigma = np.median(D)
W = np.exp(-D / (sigma + 1e-8))

# ------------------------------------------------------
# Spectral seriation using Fiedler vector
# ------------------------------------------------------
deg = np.sum(W, axis=1)
L = np.diag(deg) - W
vals, vecs = eigh(L)
order = np.argsort(vecs[:, 1])

# ------------------------------------------------------
# Save ordered list of frame filenames
# ------------------------------------------------------
ordered_files = [files[i] for i in order]
out_file = base / "coarse_order.pkl"

with open(out_file, "wb") as f:
    pickle.dump(ordered_files, f)

print(f"🎯 Coarse ordering complete! Saved to: {out_file}")
