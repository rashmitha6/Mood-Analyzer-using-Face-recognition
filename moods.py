import cv2

from deepface import DeepFace

cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)

while True:
    ret,frames=cap.read()
    
    gray=cv2.cvtColor(frames,cv2.COLOR_RGB2GRAY)
    
    faces=cascade.detectMultiScale(frames,2.0,5)


for x,y,a,b in faces:
    
    cv2.rectangle(frames,(x,y),(x+a,y+b),(255,255,0),3)
    
    cv2.imshow("video",frames)

    key =cv2.waitKey(20)
     
    if key==27:
         
         break


a=DeepFace.analyze(frames)

print(a['dominant_emotion'])