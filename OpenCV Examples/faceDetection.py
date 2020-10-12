import cv2

def detectFaces(imgPath):

    # Using OpenCV Cascades
    faceCascades = cv2.CascadeClassifier("Resources/haarsascade_frontalface_default.xml")

    img = cv2.imread(imgPath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BG2GRAY)

    faces = faceCascades.detectMultiScalse(img_gray,1.1,4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(255,0,0),thickness=2)

    cv2.imshow("Output", img)
    cv2.waitkey(0)