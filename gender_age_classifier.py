import numpy as np
import cv2
import sys
import dlib

# labels
# age_list = ['(0, 7)','(8, 13)','(14, 19)','(20,33)','(34,45)','(46,60)','(61,73)','(74, 100)']
age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

# face detect model
face_detector = dlib.get_frontal_face_detector()

# gender classifier model
gender_classifier_model = "models/gender_net.caffemodel"
gender_classifier_config = "models/deploy_gender.prototxt"

# age classifier model
age_classifier_model = "models/age_net.caffemodel"
age_classifier_config = "models/deploy_age.prototxt"

# create nets
gender_classifier_net = cv2.dnn.readNet(gender_classifier_model, gender_classifier_config)
age_classifier_net = cv2.dnn.readNet(age_classifier_model, age_classifier_config)

# image list
imgs_list = ["imgs/01.jpg","imgs/02.jpg","imgs/03.jpg","imgs/04.jpg","imgs/05.jpg","imgs/06.jpg",\
             "imgs/07.jpg","imgs/08.jpg","imgs/09.jpg","imgs/10.jpg","imgs/11.jpg","imgs/12.jpg",\
             "imgs/13.jpg","imgs/14.jpg","imgs/15.jpg"]

for img_path in imgs_list:

    img = cv2.imread(img_path)
    
    if img is None:
        sys.exit()

    # to detect faces
    faces = face_detector(img)

    for face in faces:

        x1,y1,x2,y2 = face.left(), face.top(), face.right(), face.bottom()

        face_img = img[y1:y2, x1:x2].copy()

        # create blob object to classify gender and age group
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227,227), \
            mean=(78.4263377603, 87.7689143744, 114.895847746), \
            swapRB=False, crop=False)

        # classify gender
        gender_classifier_net.setInput(blob)
        gender_out = gender_classifier_net.forward()
        gender = gender_list[gender_out[0].argmax()]

        # classify age
        age_classifier_net.setInput(blob)
        age_out = age_classifier_net.forward()
        age = age_list[age_out[0].argmax()]

        # draw rectangle around the face object
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        label = f"{gender}, {age}"
        cv2.putText(img, label, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print(label)

    cv2.imshow('img', img)
    if cv2.waitKey() == ord(" "):
        continue
    
cv2.destroyAllWindows()

