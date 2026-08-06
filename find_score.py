import cv2
import glob

# Find a frame that has the score
files = glob.glob('frames/*.jpg')
for f in sorted(files):
    img = cv2.imread(f)
    if img is None: continue
    h, w = img.shape[:2]
    # Check the display area (roughly middle left)
    roi = img[h//2-100:h//2+100, w//4:w//2]
    
    # Check for strong red
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = (160, 100, 100)
    upper = (180, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    
    if cv2.countNonZero(mask) > 100:
        print(f"Found red in {f}")
        cv2.imwrite('debug_found.jpg', img)
        break
