import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('../1785643223859.mp4')
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # We want a frame where the score might be visible.
    # Let's save frame 100 just to see what the mask looks like.
    if frame_count == 100:
        cv2.imwrite('debug_raw.jpg', frame)
        
        # Same color threshold logic
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        cv2.imwrite('debug_mask.jpg', mask)
        
        # Left half mask
        h, w = frame.shape[:2]
        left_mask = mask[:, :w//2]
        inverted_left = cv2.bitwise_not(left_mask)
        cv2.imwrite('debug_ocr_input.jpg', inverted_left)
        
        break

cap.release()
print("Debug frames extracted.")
