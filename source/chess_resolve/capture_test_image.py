import cv2
import os
from datetime import datetime

CAMERA_ID = 0
OUT_PATH = "test.jpg"          
PREVIEW_WINDOW = "Preview (press 's' to save test.jpg, 'q' to quit)"

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={CAMERA_ID}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(15):
        cap.read()

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

    saved = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        h, w = frame.shape[:2]
        info = f"Actual frame: {w}x{h} | Press 's' to save"
        cv2.putText(frame, info, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(PREVIEW_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        if key == ord('s'):
            cv2.imwrite(OUT_PATH, frame)
            print(f"Saved: {OUT_PATH} (resolution {w}x{h})")
            saved = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if not saved:
        print("No image saved.")

if __name__ == "__main__":
    main()
