import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if ret:
    print("Actual resolution:", frame.shape[1], frame.shape[0])
else:
    print("The frame was not received")

cap.release()
