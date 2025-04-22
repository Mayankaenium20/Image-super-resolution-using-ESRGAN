import os.path as osp
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from RRDBNet_arch import RRDBNet

# Function to perform image super-resolution
def super_resolve(model, input_image):
    try:
        img = np.array(input_image)

        # Check if the image has an alpha channel and convert to RGB if necessary
        if img.shape[-1] == 4:
            img = img[..., :3]

        # Check if the image is grayscale and convert to RGB if necessary
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert image to float and transpose to PyTorch format
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().clamp_(0, 1).numpy()

        # Convert output to image format
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    
    
# Main Streamlit app
def main():
    st.title("Image Super-Resolution App")

    # Model and device setup
    model_path = 'models/RRDB_ESRGAN_x4.pth'
    device = torch.device('cpu')  # Use CPU for inference

    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # File uploader for low-resolution image
    st.sidebar.header("Upload Low-Resolution Image")
    uploaded_file = st.sidebar.file_uploader("Choose a low-resolution image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Low-Resolution Image', use_column_width=True)
        st.write("Image uploaded successfully.")

        # Perform super-resolution
        output_image = super_resolve(model, Image.open(uploaded_file))

        # Display output image
        st.image(output_image, caption='Super-Resolved Image', use_column_width=True)

        # Save output image
        output_path = 'super_resolved_image.png'
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        # Create download link for the super-resolved image
        st.markdown(f"Download super-resolved image: [{output_path}](./{output_path})")

if __name__ == "__main__":
    main()
