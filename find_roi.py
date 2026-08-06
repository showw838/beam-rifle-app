import cv2

img = cv2.imread('frames/frame_330.jpg')
# Let's save a resized version to see coordinates if needed, but I can just use a python script to crop and OCR it.
# Actually I will just crop the center of the image roughly where the display is.
h, w = img.shape[:2]

# The display seems to be in the center
# Let's crop a box in the middle.
# I'll just write a script to show me where the bright pixels are.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_bright.jpg', mask)
print(f"Image shape: {w}x{h}")
