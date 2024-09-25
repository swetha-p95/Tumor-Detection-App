import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('cnn_tumor.h5')

def make_prediction(img,model):
    # img=cv2.imread(img)
    # img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res:
        print("Tumor Detected")
    else:
        print("No Tumor")
    return res


st.title("Tumour Detetction App")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    uploaded_image = Image.open(uploaded_image)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    # Classify the uploaded image
    st.write("Classifying the image...")
    res = make_prediction(uploaded_image,model)
    # Display the prediction result
    if res:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor")
    
    
