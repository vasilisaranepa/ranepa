import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl

img = cv2.imread('images/plate6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(img_filter, 30, 200)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True)

    if len(approx) == 4:
        pos = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

(z, v) = np.where(mask == 255)
(z1, v1) = (np.min(z), np.min(v))
(z2, v2) = (np.max(z), np.max(v))
crop = gray[z1:z2, v1:v2]

text = easyocr.Reader(['en'])
text = text.readtext(crop)

res = text[0][-2]
final_image = cv2.putText(img, res, (z1, v2 - 100), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5)
final_image = cv2.rectangle(img, (v1, z1), (v2, z2), (0, 255, 0), 5)

pl.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
pl.show()
