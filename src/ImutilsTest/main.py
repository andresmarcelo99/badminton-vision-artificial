from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as cv
from pyfirmata import Arduino, util

board = Arduino('COM4')

iterator = util.Iterator(board)
iterator.start()

pin = 7

def isInRange(x, y, z):
    print(x,y,z)
    if x>200 and x<220 and y>165 and y<190 and z>330 and z<355:
        print('PERSON DETECTED!!!! ')
        return True
    else:
        return False

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

out = cv.VideoWriter(
    'output.avi',
    cv.VideoWriter_fourcc(*'MJPG'),
    15,
    (940,680))

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

isDetecting = False

while(True):

    board.digital[pin].write(1 if isDetecting else 0) 
   
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=min(900, frame.shape[1]))
    

    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
        padding=(8, 8), scale=1.05) 

    if(len(rects)>0 and isInRange(rects[0][0], rects[0][2], rects[0][3])):
        isDetecting=True
    else:
        isDetecting=False

    for (x, y, w, h) in rects:
        # print(x, y,w, h)
        cv.rectangle(ret, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    for (xA, yA, xB, yB) in pick:
        cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        board.digital[pin].write(1) 
        out.write(frame.astype('uint8'))

    board.digital[pin].write(0) 
    cv.imshow('frame',frame)
   
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv.destroyAllWindows()
cv.waitKey(1)