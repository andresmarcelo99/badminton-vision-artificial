from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as cv
from pyfirmata import Arduino, util

board = Arduino('COM3')

iterator = util.Iterator(board)
iterator.start()

pin = 7


# def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=200):
#     '''(ndarray, 3-tuple, int, int) -> void
#     draw gridlines on img
#     line_color:
#         BGR representation of colour
#     thickness:
#         line thickness
#     type:
#         8, 4 or cv2.LINE_AA
#     pxstep:
#         grid line frequency in pixels
#     '''
#     x = pxstep
#     y = pxstep
#     while x < img.shape[1]:
#         cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
#         x += pxstep

#     while y < img.shape[0]:
#         cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
#         y += pxstep

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

    if(len(rects)>0):
        isDetecting=True
    else:
        isDetecting=False

    for (x, y, w, h) in rects:
        cv.rectangle(ret, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.85)
    
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