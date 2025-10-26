import cv2, os, argparse
os.makedirs("frames", exist_ok=True)

def extract(video_path, out_dir="frames"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        name = os.path.join(out_dir, f"frame_{i:04d}.png")
        cv2.imwrite(name, frame)
        i += 1
    cap.release()
    print("Extracted", i, "frames at", fps, "fps")
    return fps, i

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv)>1 else "data/jumbled_video.mp4"
    extract(video)
