import cv2
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    print(ok, frame.shape if ok else "fail")
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()