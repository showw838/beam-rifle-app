import cv2
import os

cap = cv2.VideoCapture('../1785643223859.mp4')
frame_count = 0

os.makedirs('frames', exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    if frame_count % 30 == 0:
        cv2.imwrite(f'frames/frame_{frame_count}.jpg', frame)

cap.release()
print("Frames extracted.")
