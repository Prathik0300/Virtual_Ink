import cv2 as cv
import numpy as np
from caliberation import caliberation

Lower,Upper = caliberation()
video = cv.VideoCapture(0)
x0,y0 = -1,-1
color = (204,51,255)
temp = np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
while(cv.waitKey(1)==-1):
    isTrue,frame = video.read()
    frame = cv.flip(frame,1)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,np.array(Lower),np.array(Upper))
    #cv.imshow("First mask",mask)
    kernel = np.ones((5,5),np.uint8)
    erode = cv.erode(mask,kernel,iterations=1)
    #cv.imshow("erode",erode)
    frame = np.array(frame)
    contour,hierarchy = cv.findContours(erode,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    if contour:
        c = max(contour,key = cv.contourArea)
        x,y,w,h = cv.boundingRect(c)
        if x0==-1:
            x0,y0 = x+w//2,y+h//2
        else:
            cv.line(temp,(x0,y0),(x+w//2,y+h//2),color,3)
            #cv.imshow("original temp",temp)
            x0,y0 = x+w//2,y+h//2
    else:
        x0,y0=-1,-1
    
    mask=cv.cvtColor(temp,cv.COLOR_BGR2HSV)
    #cv.imshow("second mask",mask)
    mask = cv.inRange(mask,np.array([1,1,1]),np.array([255,255,255]))
    #cv.imshow("third mask",mask)
    temp = cv.bitwise_and(temp,temp,mask=mask)
    #cv.imshow("temp",temp)
    mask = cv.bitwise_not(mask)
    #cv.imshow("fourth mask",mask)
    frame = cv.bitwise_and(frame,frame,mask=mask)
    #cv.imshow("new frame",frame)
    frame = cv.add(frame,temp)
    cv.imshow("frame",frame)
video.release()
cv.destroyAllWindows()