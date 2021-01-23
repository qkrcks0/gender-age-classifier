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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open faile!")
    sys.exit()

face_detect_net = cv2.dnn.readNet(face_detect_model, face_detect_config)
gender_classifier_net = cv2.dnn.readNet(gender_classifier_model, gender_classifier_config)
age_classifier_net = cv2.dnn.readNet(age_classifier_model, age_classifier_config)









