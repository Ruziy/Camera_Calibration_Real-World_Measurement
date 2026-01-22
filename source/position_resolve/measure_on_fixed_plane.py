import cv2
import numpy as np
import pickle
import time
from pathlib import Path

CAMERA_ID = 0
IMAGE_RES = (1280, 720)
BASE_DIR = Path(__file__).resolve().parent
CALIB_DIR = BASE_DIR.parent / "calibrate_step" / "output"
CAM_MTX_PATH = CALIB_DIR / "camera_matrix.txt"
DIST_PATH = CALIB_DIR / "distortion_coefficients.txt"
PLANE_POSE_PATH = CALIB_DIR / "plane_pose.pkl"

SHOW_RESULT_MS = 900   
AUTO_RESET = True      

clicked = []
last_result_text = None
last_result_until = 0

def mouse_cb(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        print(f"Clicked: {(x, y)}")

def undistort_points(pts, K, dist):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K) 
    return und.reshape(-1, 2)

def intersect_ray_with_plane(u, v, K, R, t):
    Kinv = np.linalg.inv(K)
    ray = Kinv @ np.array([u, v, 1.0], dtype=np.float64)

    Rt = R.T
    a = Rt @ ray
    b = Rt @ t.reshape(3)

    if abs(a[2]) < 1e-12:
        return None

    s = b[2] / a[2]
    Xb = s * a - b
    return Xb  

def main():
    global clicked, last_result_text, last_result_until

    K = np.loadtxt(CAM_MTX_PATH)
    dist = np.loadtxt(DIST_PATH).reshape(-1, 1)

    with open(PLANE_POSE_PATH, "rb") as f:
        plane = pickle.load(f)

    R = plane["R"]
    t = plane["t"]
    print("Loaded plane pose. Camera must be fixed since calibration.")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_RES[1])

    for _ in range(20):
        cap.read()

    win = "Measure: click 2 points (start,end). 'r' reset, 'q' quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()

        cv2.putText(vis, "Click 2 points. r=reset, q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        for i, (x, y) in enumerate(clicked[:2]):
            cv2.circle(vis, (x, y), 6, (0,255,0), -1)
            cv2.putText(vis, str(i+1), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        now_ms = int(time.time() * 1000)
        if len(clicked) >= 2 and last_result_until <= now_ms:
            pts_und = undistort_points(clicked[:2], K, dist)

            P1 = intersect_ray_with_plane(pts_und[0, 0], pts_und[0, 1], K, R, t)
            P2 = intersect_ray_with_plane(pts_und[1, 0], pts_und[1, 1], K, R, t)

            if P1 is None or P2 is None:
                last_result_text = "Intersection failed"
            else:
                length_cm = float(np.linalg.norm(P2[:2] - P1[:2]))
                last_result_text = f"Length: {length_cm:.2f} cm"
                print(last_result_text)

            last_result_until = now_ms + SHOW_RESULT_MS

        if last_result_until > now_ms and len(clicked) >= 2:
            p1 = tuple(map(int, clicked[0]))
            p2 = tuple(map(int, clicked[1]))
            cv2.line(vis, p1, p2, (0, 255, 0), 2)
            if last_result_text:
                cv2.putText(vis, last_result_text,
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if AUTO_RESET and last_result_until > 0 and last_result_until <= now_ms:
            clicked = []
            last_result_text = None
            last_result_until = 0

        cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF

        if k == ord('q') or k == 27:
            break
        if k == ord('r'):
            clicked = []
            last_result_text = None
            last_result_until = 0
            print("Reset.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
