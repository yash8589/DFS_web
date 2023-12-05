import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights, resnet50
import torch.optim as optim
import pandas as pd
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import numpy as np
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

model = load_model('t3_5_epoch.h5')

# find the index of the maximum element in an array
def find_max(arr):
    maxi = max(arr)
    for i in range(len(arr)):
        if arr[i] == maxi:
            return i
        
train_ds= ['all_benign',
            'all_early',
            'all_pre',
            'all_pro',
            'brain_glioma',
            'brain_menin',
            'brain_tumor',
            'breast_benign',
            'breast_malignant',
            'cervix_dyk',
            'cervix_koc',
            'cervix_mep',
            'cervix_pab',
            'cervix_sfi',
            'colon_aca',
            'colon_bnt',
            'kidney_normal',
            'kidney_tumor',
            'lung_aca',
            'lung_bnt',
            'lung_scc',
            'lymph_cll',
            'lymph_fl',
            'lymph_mcl',
            'oral_normal',
            'oral_scc']

# predict the class from an imput image using our trained model
def predict_image_class(model, img, show = True):
    img = cv2.imread(img)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    classes = model.predict(img)
    print(classes)
    # maxi = max(classes)
    # # find the class with maximum probability and print it
    # l = []
    # for i in range(5):
    #     if classes[0][i] == maxi:
    #         max_ind = i
    #         break
    #     print(train_ds.class_names[max_ind])

    max_ind = find_max(classes[0])
    print(train_ds[max_ind])


st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Streamlit page title and description
st.title("Image Classification App")
st.write("Upload an image, and we'll predict its class.")

# Create a file uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction when the "Predict" button is clicked
    if st.button("Predict"):
        try:
            # Load and preprocess the uploaded image
            st.write("Processing the image...")
            # img = image.load_img(uploaded_image, target_size=(224, 224))  # Adjust target_size to match your model's input size
            # img_array = image.img_to_array(img)
            # img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
            # img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

            # # Make predictions using the loaded model
            # predictions = model.predict(img_array)

            # # Decode the predictions if your model has categorical labels
            # class_labels = ['brain_glioma', 'brain_menin', 'brain_tumor', 'oral_normal', 'oral_scc']  # Replace with your class labels
            # predicted_class = predict_image_class(model,uploaded_image)
            img = image.load_img(uploaded_image, target_size=(224, 224))
            # img = cv2.resize(uploaded_image,(224,224))
            img = np.reshape(img,[1,224,224,3])
            classes = model.predict(img)
            # print(classes)

            max_ind = find_max(classes[0])
            predicted_class = train_ds[max_ind]
            print(train_ds[max_ind])

            # Display the prediction
            st.success(f"Prediction: {predicted_class}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add a brief description and instructions
st.write("### Instructions:")
st.write("1. Upload an image (in JPG, PNG, or JPEG format) using the 'Choose an image...' button.")
st.write("2. Click the 'Predict' button to classify the image into one of the predefined classes.")
st.write("3. The predicted class will be displayed on the screen.")

# Provide additional information or links to model details, data sources, etc.
st.write("### About This Model:")
st.write("This model was trained to classify images into specific categories.")
st.write("For more details about the model and the categories, please visit our [model documentation].")

# You can also add a link to the model documentation here.
# st.markdown("[Model Documentation](https://your_model_documentation_url)")

# Add a footer with contact information or copyright details
st.write("Contact us at yash.agarwal@research.iiit.ac.in or syed.i@research.iiit.ac.in")
st.write("© 2023 Data Foundation system..")
