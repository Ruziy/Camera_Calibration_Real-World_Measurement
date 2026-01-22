import cv2
import numpy as np
import pickle
import os
from pathlib import Path

CAMERA_ID = 0
IMAGE_RES = (1280, 720)

CHESSBOARD_SIZE = (9, 6)  
SQUARE_SIZE_CM = 2.0

BASE_DIR = Path(__file__).resolve().parent
CALIB_DIR = BASE_DIR.parent / "calibrate_step" / "output"
CAM_MTX_PATH = CALIB_DIR / "camera_matrix.txt"
DIST_PATH = CALIB_DIR / "distortion_coefficients.txt"
OUT_PLANE_PATH = CALIB_DIR / "plane_pose.pkl"

def make_object_points():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_CM
    return objp

def main():
    K = np.loadtxt(CAM_MTX_PATH)
    dist = np.loadtxt(DIST_PATH).reshape(-1, 1)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_RES[1])

    for _ in range(20):
        cap.read()

    objp = make_object_points()
    win = "Plane calibration: put chessboard on WORK PLANE, press 's' to solvePnP, 'q' to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, None)
        vis = frame.copy()
        cv2.putText(vis, "Press 's' to save",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)   
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, True)
            cv2.putText(vis, "Chessboard found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(vis, "Chessboard NOT found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF

        if k == ord('q') or k == 27:
            break

        if k == ord('s'):
            if not found:
                print("Cannot solve: chessboard not found.")
                continue

            ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                print("solvePnP failed")
                continue

            R, _ = cv2.Rodrigues(rvec)
            data = {
                "R": R.astype(np.float64),
                "t": tvec.astype(np.float64),
                "CHESSBOARD_SIZE": CHESSBOARD_SIZE,
                "SQUARE_SIZE_CM": float(SQUARE_SIZE_CM),
                "IMAGE_RES_USED": (int(frame.shape[1]), int(frame.shape[0]))
            }

            os.makedirs(os.path.dirname(OUT_PLANE_PATH), exist_ok=True)
            with open(OUT_PLANE_PATH, "wb") as f:
                pickle.dump(data, f)

            print(f"Saved plane pose to: {OUT_PLANE_PATH}")
            print("Do NOT move camera after this. Remove chessboard for production.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
