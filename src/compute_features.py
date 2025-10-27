import cv2, os, glob, numpy as np
from tqdm import tqdm
import multiprocessing as mp

def resize_keep_aspect(img, short_side=480):
    h,w = img.shape[:2]
    if min(h,w) <= short_side:
        return img
    if h < w:
        scale = short_side / h
    else:
        scale = short_side / w
    return cv2.resize(img, (int(w*scale), int(h*scale)))

def compute_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_bins, s_bins = 32, 32
    hist = cv2.calcHist([hsv],[0,1],None,[h_bins,s_bins],[0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compute_orb(img):
    orb = cv2.ORB_create(nfeatures=1000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, des = orb.detectAndCompute(gray, None)
    return des

def process_file(fname):
    img = cv2.imread(fname)
    img_small = resize_keep_aspect(img, short_side=480)
    return fname, compute_histogram(img_small), compute_orb(img_small)

if __name__ == "__main__":
    files = sorted(glob.glob("frames/*.png"))
    pool = mp.Pool(mp.cpu_count())
    results = list(tqdm(pool.imap(process_file, files), total=len(files)))
    pool.close()
    # save features as .npz or .npy per-frame for later use
    import pickle
    with open("resized_filenames.pkl","wb") as f:
        pickle.dump(results, f)
