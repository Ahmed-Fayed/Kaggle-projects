# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:45:38 2021

@author: Ahmed Fayed
"""

from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import requests
import matplotlib.pyplot as plt


original = Image.open(requests.get('https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg', stream=True).raw)
tampered = Image.open(requests.get('https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png', stream=True).raw) 

# plt.imshow(original)
# plt.imshow(tampered)
# plt.show()


# the file formate of the source file
print('Original image format: ', original.format)
print('Tampered image format: ', tampered.format)

# Image size
print('Original image size: ', original.size)
print('Tampered image size: ', tampered.size)


original = original.resize((250, 160))
print(original.size)
original.save('pan_card_tampering/image/original.png')#Save image
tampered = tampered.resize((250,160))
print(tampered.size)
tampered.save('pan_card_tampering/image/tampered.png')#Saves image



tampered = Image.open('pan_card_tampering/image/tampered.png')
tampered.save('pan_card_tampering/image/tampered.png')#can do png to jpg


original = cv2.imread('PAN_card_tampering/image/original.png')
tampered = cv2.imread('PAN_card_tampering/image/tampered.png')

original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

# plt.imshow(original_gray)


(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))



thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
plt.imshow(thresh)


th2 = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
plt.imshow(th2)
    


th3 = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
plt.imshow(th3)



    
contours, h= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(original, contours, -1, (0,255,0), 3)
plt.imshow(original)

cv2.drawContours(tampered, contours, -1, (0,255,0), 3)
plt.imshow(tampered)

# contours = imutils.grab_contours(contours)





# for c in contours:
#     # applying contours on image
#     (x, y, w, h) = cv2.boundingRect(c)
#     cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)

# print('Original Format Image')
# plt.imshow(Image.fromarray(original))
# plt.imshow(Image.fromarray(tampered))

# All the black in ( diff image ) in the image is the diffrence between two images
plt.imshow(diff)

plt.imshow(thresh)









