import cv2
cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
while True:
    ret,frames=cap.read()
    gray=cv2.cvtColor(frames,cv2.COLOR_RGB2GRAY)
    faces=cascade.detectMultiScale(frames,1.1,4)
    for x,y,a,b in faces:
        cv2.rectangle(frames,(x,y),(x+a,y+b),(255,255,0),3)
    cv2.imshow("video",frames)
    key=cv2.waitKey(20)
    if key==27:
        break
        
cv2.waitKey(0)