import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import requests


def kmeans_segmentation(image, k):
    # Convert image to RGB if it is in another mode
    img = np.array(image)
    img_reshaped = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img_reshaped)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(img.shape)
    return segmented_img.astype(np.uint8)


def app():
    st.set_page_config(page_title='CoTAI - S7 Postclass')
    st.title('Image Segmentation using KMeans')

    # Upload image or URL
    k_value = st.slider("Select number of clusters (k)", 2, 10, 5)
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter image URL")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif image_url:
        image = Image.open(requests.get(image_url, stream=True).raw)
    else:
        image = None

    if image is not None:
        left_col, right_col = st.columns(2)
        left_col.image(image, caption='Original Image', use_column_width=True)

        # Apply KMeans segmentation
        segmented_image = kmeans_segmentation(image, k_value)

        right_col.image(segmented_image,
                        caption='Segmented Image', use_column_width=True)
    else:
        st.write("Please upload an image or enter an image URL.")


if __name__ == "__main__":
    app()
