import numpy as np
import cv2
import streamlit as st

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

    # Load the image
    img_path = "Monet.bmp"
    img = cv2.imread(img_path, -1)
    if img is None:
        st.error("Image not found!")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nr, nc = img.shape[:2]

    # Sliders for parameters
    x0 = st.sidebar.slider("x0", 0, nr, nr)
    y0 = st.sidebar.slider("y0", 0, nc, nc)
    sigma = st.sidebar.slider("sigma", 1, 500, 200)

    img2 = image_formation_model(img, x0, y0, sigma)

    # Display images
    st.image([img, img2], caption=['Original Image', 'Image Formation Model'], use_column_width=True)

if __name__ == "__main__":
    main()
