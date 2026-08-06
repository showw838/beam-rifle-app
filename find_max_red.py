import cv2
import numpy as np

cap = cv2.VideoCapture('../1785643223859.mp4')
max_red = 0
best_frame = None
best_frame_idx = 0

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    count += 1
    # Check every 10 frames to save time
    if count % 10 != 0: continue
    
    h, w = frame.shape[:2]
    
    # The lower device is located roughly:
    # y: h/2 to h*3/4
    # x: w/4 to w*3/4
    # The score panel is on the left side of this device:
    # y: h/2 + 20 to h/2 + 150
    # x: w/4 + 20 to w/2
    roi = frame[int(h*0.5):int(h*0.65), int(w*0.25):int(w*0.45)]
    
    # Calculate red intensity: R - (G+B)/2
    b, g, r = cv2.split(roi)
    red_diff = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
    
    red_score = cv2.sumElems(red_diff)[0]
    
    if red_score > max_red:
        max_red = red_score
        best_frame = frame.copy()
        best_frame_idx = count
        
    if count > 3000: # Limit search to first 100 seconds to be safe
        break

if best_frame is not None:
    cv2.imwrite('debug_best.jpg', best_frame)
    print(f"Best frame was {best_frame_idx} with red score {max_red}")
cap.release()
