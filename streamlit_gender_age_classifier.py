import numpy as np
import cv2
import dlib
import streamlit as st
import pandas as pd
from PIL import Image

# labels
age_list = ['(0, 3)','(4, 7)','(8, 13)','(15, 21)','(25, 33)','(38, 44)','(48, 54)','(61, 100)']
gender_list = ['Male', 'Female']

@st.cache(allow_output_mutation=True)
def load_models():

    # face detect model
    face_detector = dlib.get_frontal_face_detector()

    # landmarks detect model
    sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

    # gender classifier model
    gender_classifier_model = "models/gender_net.caffemodel"
    gender_classifier_config = "models/deploy_gender.prototxt"

    # age classifier model
    age_classifier_model = "models/age_net.caffemodel"
    age_classifier_config = "models/deploy_age.prototxt"

    # create nets
    gender_classifier_net = cv2.dnn.readNetFromCaffe(gender_classifier_config, gender_classifier_model)
    age_classifier_net = cv2.dnn.readNetFromCaffe(age_classifier_config, age_classifier_model)

    return face_detector, sp, gender_classifier_net, age_classifier_net

def predict(img):

    # to detect faces
    faces = face_detector(img)
    age = []
    gender = []

    for face in faces:

        x1,y1,x2,y2 = face.left(), face.top(), face.right(), face.bottom()
        
        s = sp(img, face)

        # face_img = img[y1:y2, x1:x2].copy()
        face_img = dlib.get_face_chip(img, s, size=227, padding=0.4)

        # create blob object to classify gender and age group
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227,227), \
            mean=(78.4263377603, 87.7689143744, 114.895847746), \
            swapRB=False, crop=False)

        # classify gender
        gender_classifier_net.setInput(blob)
        gender_out = gender_classifier_net.forward()
        # gender = gender_list[gender_out[0].argmax()]
        gender.append(gender_out[0])
        
        # classify age
        age_classifier_net.setInput(blob)
        age_out = age_classifier_net.forward()
        # age = age_list[age_out[0].argmax()]
        age.append(age_out[0])

        # draw rectangle around the face object
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        labels = f"{gender_list[gender_out[0].argmax()]}, {age_list[age_out[0].argmax()]}"
        cv2.putText(img, labels, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    return img, gender, age

st.write("# Gender and Age Classifier")

face_detector, sp, gender_classifier_net, age_classifier_net = load_models()

uploaded_file = st.file_uploader("Choose an image to Classify...")
if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))

    st.image(img, caption='Uploaded Image.', use_column_width=True)

    st.write("")

    dst_img, gender, age = predict(img)

    st.write("# Result")
    st.image(dst_img, use_column_width=True)

    for i in range(len(gender)):

        st.write("")
        gender_df = pd.DataFrame({
            "gender":gender_list,
            "confidence":gender[i]
        })
        gender_df = gender_df.set_index("gender")
        st.write(f"Gender{i+1}")
        st.bar_chart(gender_df)

    for i in range(len(age)):

        age_df = pd.DataFrame({
            "age":age_list,
            "confidence":age[i]
        })
        age_df = age_df.set_index("age")
        st.write(f"Age{i+1}")
        st.bar_chart(age_df)


