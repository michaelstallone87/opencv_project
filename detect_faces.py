import cv2

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rtsp://admin:123456@192.168.0.101:554/live/ch0')

faceClassif = cv2.CascadeClassifier('/home/michael/python/haarcascade_frontalface_default.xml')

while True:
  ret,frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceClassif.detectMultiScale(gray, 1.3, 5)

  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

  cv2.imshow('Câmera',frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()