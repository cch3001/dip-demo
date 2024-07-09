import numpy as np
import cv2
import streamlit as st
from PIL import Image

def image_formation_model(f, x0, y0, sigma):
    g = f.copy()
    nr, nc = f.shape[:2]
    illumination = np.zeros([nr, nc], dtype='float32')
    for x in range(nr):
        for y in range(nc):
            illumination[x, y] = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) /
                                        (2 * sigma * sigma))
    for x in range(nr):
        for y in range(nc):
            for k in range(3):
                val = round(illumination[x, y] * f[x, y, k])
                g[x, y, k] = np.uint8(val)
    return g

def main():
    st.title("Image Formation Model with Streamlit")

    # File uploader for image selection
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        img = np.array(image)
        
        # Check if the image is grayscale and convert it to RGB
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        nr, nc = img.shape[:2]  #gat image wi

        # Sliders for parameters
        x0 = st.sidebar.slider("x0", 0, nr, nr // 2)
        y0 = st.sidebar.slider("y0", 0, nc, nc // 2)
        sigma = st.sidebar.slider("sigma", 1, 500, 200)

        img2 = image_formation_model(img, x0, y0, sigma)

        # Display images
        st.image([img, img2], caption=['Original Image', 'Image Formation Model'], use_column_width=True)
    else:
        st.info("Please upload an image file.")

if __name__ == "__main__":
    main()
