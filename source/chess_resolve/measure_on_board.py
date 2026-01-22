import cv2
import numpy as np
from pathlib import Path

IMAGE_PATH = "test.jpg"
BASE_DIR = Path(__file__).resolve().parent
CALIB_DIR = BASE_DIR.parent / "calibrate_step" / "output"
CAM_MTX_PATH = CALIB_DIR / "camera_matrix.txt"
DIST_PATH = CALIB_DIR / "distortion_coefficients.txt"
CHESSBOARD_SIZE = (9, 6)  
SQUARE_SIZE_CM = 2.0       
USE_SB = True              

clicked = []
def mouse_cb(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        print(f"Clicked: {(x, y)}")

def load_calib():
    K = np.loadtxt(CAM_MTX_PATH)
    dist = np.loadtxt(DIST_PATH).reshape(-1, 1)
    return K, dist

def make_object_points():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_CM
    return objp

def undistort_points(pts, K, dist):
    # pts: (N,2)
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)  
    return und.reshape(-1, 2)

def intersect_ray_with_plane(u, v, K, R, t):
    Kinv = np.linalg.inv(K)
    ray = Kinv @ np.array([u, v, 1.0], dtype=np.float64)  
    Rt = R.T
    a = Rt @ ray
    b = Rt @ t.reshape(3)

    if abs(a[2]) < 1e-9:
        return None

    s = b[2] / a[2]
    Xb = s * a - b
    return Xb  

def main():
    global clicked

    K, dist = load_calib()
    objp = make_object_points()
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError(f"Cannot read {IMAGE_PATH}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if USE_SB:
        found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, None)
    else:
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if not found:
        print("Chessboard NOT found. Увеличьте контраст/размер, уберите блики, сделайте шахматку крупнее в кадре.")
        return

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("solvePnP failed.")
        return

    R, _ = cv2.Rodrigues(rvec)
    vis = img.copy()
    cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, found)
    cv2.putText(vis, "Click 2 points on ruler (start, end). Press 'r' reset, 'q' quit.",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    win = "Measure"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        frame = vis.copy()
        for i, (x, y) in enumerate(clicked):
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow(win, frame)
        k = cv2.waitKey(20) & 0xFF

        if k == ord('q') or k == 27:
            cv2.destroyAllWindows()
            return
        if k == ord('r'):
            clicked = []
            print("Reset.")
        if len(clicked) >= 2:
            break

    cv2.destroyAllWindows()
    pts_und = undistort_points(clicked, K, dist)
    P1 = intersect_ray_with_plane(pts_und[0,0], pts_und[0,1], K, R, tvec)
    P2 = intersect_ray_with_plane(pts_und[1,0], pts_und[1,1], K, R, tvec)

    if P1 is None or P2 is None:
        print("Ray-plane intersection failed.")
        return

    length_cm = float(np.linalg.norm(P2[:2] - P1[:2]))

    print("\n--- RESULT ---")
    print(f"P1 board (cm): {P1}")
    print(f"P2 board (cm): {P2}")
    print(f"Length (cm): {length_cm:.2f}")

if __name__ == "__main__":
    main()
