from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as cv


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

out = cv.VideoWriter(
    'output.avi',
    cv.VideoWriter_fourcc(*'MJPG'),
    15.,
    (940,680))

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

while(True):

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=min(900, frame.shape[1]))
    
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
        padding=(8, 8), scale=1.05) 

    for (x, y, w, h) in rects:
        cv.rectangle(ret, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.85)
    
    for (xA, yA, xB, yB) in pick:
        cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        out.write(frame.astype('uint8'))
   
    cv.imshow('frame',frame)
   
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv.destroyAllWindows()
cv.waitKey(1)