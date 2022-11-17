from re import I
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

#parte de lectura de las imagenes, escala de grises y enfoque 

img = cv.imread('Placas_1.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))

#parte de aplicación de filtros y  localización 
bfilter = cv.bilateralFilter(gray, 11, 17, 17) #parte de reducción de sonido 
edged = cv.Canny(bfilter, 30, 200) #Parte de deteccion de los bordes
plt.imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))

#parte de los contornos y aplicación de la mascara 
keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv.drawContours(mask, [location], 0,255, -1)
new_image = cv.bitwise_and(img, img, mask=mask)

plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))

# parte del codigo donde se utiliza la libreria de OCR para leer las placas

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)


#parte del renderizado y seleccion de la placa 

text = result[0][-2]
font = cv.FONT_HERSHEY_SIMPLEX
res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))