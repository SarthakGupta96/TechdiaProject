import cv2, os, glob, numpy as np
from tqdm import tqdm
import multiprocessing as mp
import pickle

# Ensure deterministic behavior
cv2.setNumThreads(0)

def resize_keep_aspect(img, short_side=480):
    if img is None:
        return None
    h, w = img.shape[:2]
    if min(h, w) <= short_side:
        return img
    scale = short_side / min(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def compute_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compute_orb(img):
    orb = cv2.ORB_create(nfeatures=1000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, des = orb.detectAndCompute(gray, None)
    return des

def process_file(fname):
    img = cv2.imread(fname)
    if img is None:
        return None  # skip bad frames
    img_small = resize_keep_aspect(img, short_side=480)
    hist = compute_histogram(img_small)
    orb = compute_orb(img_small)
    return (fname, hist, orb)

if __name__ == "__main__":
    files = sorted(glob.glob("frames/*.png"))
    print(f"📸 Found {len(files)} frames")

    pool = mp.Pool(mp.cpu_count())
    results = []
    for res in tqdm(pool.imap(process_file, files), total=len(files)):
        if res is not None:
            results.append(res)
    pool.close()
    pool.join()

    print(f"✅ Successfully processed {len(results)} frames")

    with open("resized_filenames.pkl", "wb") as f:
        pickle.dump(results, f)

    print("💾 Features saved to resized_filenames.pkl")
