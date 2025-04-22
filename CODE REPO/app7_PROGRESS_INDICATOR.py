import os.path as osp
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from RRDBNet_arch import RRDBNet
import time

# Function to perform image super-resolution
def super_resolve(model, input_image, progress_bar):
    img = np.array(input_image)

    # Convert image to RGB if it has an alpha channel
    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        start_time = time.time()
        total_time = 0
        for i in range(101):
            progress_bar.progress(i)
            if i < 100:
                time.sleep(total_time / 100)  # Simulate processing time based on progress
            if i == 100:
                progress_bar.empty()  # Remove progress bar after completion
            output = model(img_LR).data.squeeze().float().clamp_(0, 1).numpy()
            end_time = time.time()
            total_time += end_time - start_time
        st.write(f"Total time taken for image generation: {total_time:.2f} seconds")

    # Convert output to image format
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output

# Main Streamlit app
def main():
    st.title("Image Super-Resolution App")

    # Model and device setup
    model_path = 'models/RRDB_ESRGAN_x4.pth'
    device = torch.device('cpu')  # Use CPU for inference

    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    st.write('Model path:', model_path)
    st.write('Testing...')

    # File uploader for low-resolution image
    st.sidebar.header("Upload Low-Resolution Image")
    uploaded_file = st.sidebar.file_uploader("Choose a low-resolution image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Low-Resolution Image', use_column_width=True)
        st.write("Image uploaded successfully.")

        # Perform super-resolution
        st.subheader("Processing Image...")
        progress_bar = st.progress(0)
        output_image = super_resolve(model, Image.open(uploaded_file), progress_bar)

        # Display output image
        st.subheader("Super-Resolved Image")
        st.image(output_image, caption='Super-Resolved Image', use_column_width=True)

        # Save output image
        output_path = 'super_resolved_image.png'
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        # Display before-and-after comparison
        st.subheader("Before and After Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption='Original Image', use_column_width=True)
        with col2:
            st.image(output_image, caption='Super-Resolved Image', use_column_width=True)

        # Create download button for the super-resolved image
        with open(output_path, 'rb') as f:
            data = f.read()
        st.download_button(label="Download super-resolved image", data=data, file_name='super_resolved_image.png')

if __name__ == "__main__":
    main()
