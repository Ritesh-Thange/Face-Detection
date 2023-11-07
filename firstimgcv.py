import cv2 as cv

#____________________________________________________________________image showing______________________________________________________________#

# img = cv.imread('Photo/gun.webp')
# cv.imshow('gun', img)
# cv.waitKey(0)

#_____________________________________________________________________video playing____________________________________________________________ #

# capture = cv.VideoCapture('video/video1.mp4')
# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video',frame)
#     if cv.waitKey(20) & 0xFF==ord('e'):
#         break

# capture.release()
# cv.distroyAllWindows()

#____________________________________________________________________resize image_______________________________________________________________#

# img = cv.imread('Photo/tree.jpg')

# def rescaleFrame(frame, scale):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width,height)

#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


# r_img = rescaleFrame(img , scale=0.288)
# cv.imshow('tree',r_img) 
# cv.waitKey(0)

#____________________________________________________________________resize video_________________________________________________________________#

# capture = cv.VideoCapture('video/video1.mp4')

# def rescaleFrame(frame, scale):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width,height)

#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


# while True:
#     isTrue, frame = capture.read()
#     r_vid = rescaleFrame(frame, scale = 0.5)
#     cv.imshow('resize video is', r_vid)
#     if cv.waitKey(20) & 0xFF==ord('e'):
#         break

# capture.release()
# cv.distroyAllWindows()

#___________________________________________________________________live video with resize________________________________________________________#

# capture = cv.VideoCapture(0)

# def rescaleFrame(frame, scale):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width,height)

#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


# while True:
#     isTrue, frame = capture.read()
#     r_vid = rescaleFrame(frame, scale = 1.5)
#     cv.imshow('resize video is', r_vid)
#     if cv.waitKey(20) & 0xFF==ord('e'):
#         break

# capture.release()
# cv.distroyAllWindows()

#___________________________________________________________________face detection using img ________________________________________________#


# img = cv.imread('Photo/groupjpg.jpg')
# def rescaleFrame(frame, scale):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width,height)
#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
# r_img = rescaleFrame(img , scale=2)
# face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# faces = face_cascade.detectMultiScale(r_img,scaleFactor=1.19,minNeighbors=5)
# print("Total Faces:",len(faces))
# # print("Face Coordinates:",faces)
# for x,y,w,h in faces:
#     img=cv.rectangle(r_img,(x,y),(x+w,y+h),(0,255,0),3)
# cv.imshow("Family",r_img)
# cv.waitKey(0)
 
 #___________________________________________________________________face cam detect ________________________________________________________#

face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv.VideoCapture(0)
while True:
    ret,frame=capture.read()
    faces = face_cascade.detectMultiScale(frame,scaleFactor=1.19,minNeighbors=8)
    for x,y,w,h in faces:
        frame=cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv.imshow('detect',frame)
    if cv.waitKey(1) & 0xFF == ord(' '):
        break
capture.release()


# import cv2

# # Load the pre-trained Haar cascade file for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load the image
# image = cv2.imread('photo/64.jpg')

# # Convert the image to grayscale for face detection
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Iterate over the detected faces
# for (x, y, w, h) in faces:
#     # Draw a rectangle around the face
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Add the name of the person detected (you can modify this based on your needs)
#     name = "Person"
#     cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# # Display the image with faces and names
# cv2.imshow("Face Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
