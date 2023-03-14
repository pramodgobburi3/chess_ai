import cv2
import numpy as np

image = cv2.imread('sample.JPG')
im_id = "0befefb1-f385-4f59-90a5-6f744a1dd7fe"
pts = np.array([[0.17578125,0.12109375],[0.89453125,0.125],[0.88671875,0.85546875],[0.15625,0.83203125],[0.17578125,0.12109375]])

mod_pts = []
for [x,y] in pts:
    mod_pts.append([x *  265, y * 265])

pts = np.array(mod_pts)
print(pts)

print(image.shape)

# Blue color in BGR
color = (255, 0, 0)
 
# Line thickness of 2 px
thickness = 2

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Find Canny edges
edged = cv2.Canny(gray, 30, 200)

contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

im_contours = cv2.drawContours(image, contours, -1, (0,255,0), 3)

new_pts = np.array([[47.21,32], [237, 33], [234,227], [41,220],[47,32]])
polylines = cv2.polylines(image, [new_pts], True, color, thickness)

cv2.imshow('contours', im_contours)
cv2.imshow('polylines', polylines)

cv2.waitKey(0)
  
cv2.destroyAllWindows()