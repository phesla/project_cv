import cv2
import sys
import cnn
from imutils.object_detection import non_max_suppression


yolo = cnn.YOLO_TF()
yolo.disp_console = False
yolo.imshow = True



#cascPath = 'cascades/fullbody_cascade.xml'
#peopleCascade = cv2.CascadeClassifier(cascPath)

#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video_capture = cv2.VideoCapture(0)

def getFrame():
    retval, img = video_capture.read()
    return img

def hog_detect_people(frame):
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return 0

n_image = 0;

while True:
    
    ret, frame = video_capture.read()

    coords = yolo.detect_from_cvmat(frame)
      
    if len(coords):
        for obj in coords:
            cv2.rectangle(frame, (obj[0] - obj[2], obj[1] - obj[3]), (obj[0] + obj[2], obj[1] + obj[3]), (255, 0, 0), 2)
        cv2.imwrite('img/' + str(n_image) + '.jpg', frame)
        print(coords)

    cv2.imshow('haar', frame)
        
    n_image += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
