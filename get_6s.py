import cv2

cap = cv2.VideoCapture('../1785643223859.mp4')
# 6000 ms = 6 seconds, which is where 10.3 was visible
cap.set(cv2.CAP_PROP_POS_MSEC, 6000)
ret, frame = cap.read()
if ret:
    cv2.imwrite('debug_6sec.jpg', frame)
    print("Saved debug_6sec.jpg")

cap.set(cv2.CAP_PROP_POS_MSEC, 16000)
ret, frame = cap.read()
if ret:
    cv2.imwrite('debug_16sec.jpg', frame)
    print("Saved debug_16sec.jpg")

cap.release()
