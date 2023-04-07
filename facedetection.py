import cv2  #OpenCV help us in image processing of python.
cap = cv2.VideoCapture(0)
faseCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # haar cascade frontal face recognizer to
# detect the face from our webcam.

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )

    faces = faseCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
    )                           # we will tell python that here is the face, draw a rectangle around it.

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2 )    # That means when we move
        # our face they will also move with the face

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
