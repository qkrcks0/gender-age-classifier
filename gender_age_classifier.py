import numpy as np
import cv2
import sys

# labels
age_list = ['(0, 3)','(4, 8)','(9, 15)','(16, 25)','(26, 35)','(36, 45)','(46, 59)','(60, 100)']
gender_list = ['Male', 'Female']

# face detect model
face_detect_model = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_detect_config = "models/deploy.prototxt"

# gender classifier model
gender_classifier_model = "models/gender_net.caffemodel"
gender_classifier_config = "models/deploy_gender.prototxt"

# age classifier model
age_classifier_model = "models/age_net.caffemodel"
age_classifier_config = "models/deploy_age.prototxt"

face_detect_net = cv2.dnn.readNet(face_detect_model, face_detect_config)
gender_classifier_net = cv2.dnn.readNet(gender_classifier_model, gender_classifier_config)
age_classifier_net = cv2.dnn.readNet(age_classifier_model, age_classifier_config)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open faile!")
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    blob1 = cv2.dnn.blobFromImage(frame, scalefactor=1, size=(300, 300),\
                                  mean=(104, 177, 123))
    net.setInput(blob1)
    out = net.forward()

    face_detected = out[0, 0, :, :]
    (h,w) = frame.shape[:2]

    for i in range(face_detected.shape[0]):
        confidence = face_detected[i, 2]
        if confidence < 0.5:
            break

        x1 = int(face_detected[i, 3] * w)
        y1 = int(face_detected[i, 4] * h)
        x2 = int(face_detected[i, 5] * w)
        y2 = int(face_detected[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = f'Face: {confidence:4.2f}'
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    blob2 = cv2.dnn.blobFromImage(frame, scalefactor=1, size=(227,227), \
            mean=(74.4263377603, 87.7689143744, 114.895847746), \
            swapRB=False, crop=False)
    
    # gender_classifier_net.setInput(blob)
    # gender_out = gender_classifier_net.forward()
    # gender = 123







