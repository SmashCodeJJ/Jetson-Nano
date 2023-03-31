import cv2 as cv
import numpy as np

# camera setup for CSI IMX219 (can adjust resolution to increase pfs)
# if it is webcam, we can remove this function, modify videocapture(0)
def gstreamer_pipeline(
    capture_width=960,
    capture_height=540,
    display_width=960,
    display_height=540,
    framerate=90,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

capture = cv.VideoCapture(gstreamer_pipeline(flip_method=0), cv.CAP_GSTREAMER)

ok, frame = capture.read()
print(frame)


lower_b = (65, 43, 46) # (65, 43, 46) # green (36, 25, 25) 
upper_b = (110, 255, 255) # (110, 255, 255)# green (70, 255,255) 

height, width = frame.shape[0:2]
screen_center = width / 2
offset = 50
dire = ""
while ok:
    
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(hsv_frame, lower_b, upper_b)
    mask2 = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask3 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)
    
   
    contours, _ = cv.findContours(mask3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    maxArea = 0
    maxIndex = 0
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        if area > maxArea:
            maxArea = area
            maxIndex = i
    
	
    cv.drawContours(frame, contours, maxIndex, (255, 255, 0), 2)
   
    x, y, w, h = cv.boundingRect(contours[maxIndex])
    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    center_x = int(x + w/2)
    center_y = int(y + h/2)
    cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    
    if center_x < screen_center - offset:
              
        dire = "turn left"
        print(dire)
    elif screen_center - offset <= center_x <= screen_center + offset:
        
        dire = "keep"
        
        print(dire)
    elif center_x > screen_center + offset:
        
        dire = "turn right"
        print(dire)

    cv.imshow("mask4", mask3)
    cv.imshow("frame", frame)
    cv.waitKey(1)
    ok, frame = capture.read()

