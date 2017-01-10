# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 00:25:53 2017

@author: tlacson
"""

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(r'C:\apps\Anaconda3\envs\ClassifyingFish\pkgs\opencv3-3.1.0-py35_0\Library\etc\haarcascades\haarcascade_eye.xml')


img = cv2.imread(r'c:\data\projects\ClassifyingFish\images\maxresdefault.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()