# Image Super-Resolution using ESRGAN

This project demonstrates the implementation of an **Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)** to upscale low-resolution images into high-resolution outputs. ESRGAN leverages deep learning techniques, adversarial training, and residual-in-residual dense blocks (RRDB) to produce perceptually accurate and visually realistic images.

## Objective

To develop an image super-resolution system that enhances the quality of low-resolution images for applications in photography, surveillance, and medical imaging using ESRGAN.

## Key Concepts

- **ESRGAN Architecture**:
  - **Generator**: Uses stacked RRDB blocks to transform low-res images into high-res.
  - **Discriminator**: Distinguishes between real and generated high-res images using adversarial training.
  - **Loss Functions**: Combines pixel-wise loss (MSE) with perceptual and adversarial losses for better quality.
  - **Dense Blocks**: Improve feature reuse and gradient flow for deeper learning.

## Performance Metrics

| Image     | PSNR (dB) | SSIM   |
|-----------|-----------|--------|
| IMAGE_1   | 28.84     | 0.9416 |
| IMAGE_2   | 25.37     | 0.8303 |
| IMAGE_3   | 27.04     | 0.9555 |

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image reconstruction quality.
- **SSIM (Structural Similarity Index)**: Measures perceived structural similarity.

## Results

The ESRGAN model significantly improves image resolution and visual quality compared to traditional interpolation methods and even previous deep learning models like SRGAN and SRCNN.

## Tools & Technologies

- Python
- PyTorch
- OpenCV
- ESRGAN architecture
- Jupyter Notebooks for experimentation and visualization

## References

- Ledig, C. et al. (2017). Photo-Realistic Single Image Super-Resolution Using a GAN.
- Zhang, Y. et al. (2018). Residual Dense Network for Super-Resolution.
- Tai, Y. et al. (2017). MemNet: A Persistent Memory Network for Image Restoration.
- Lim, B. et al. (2017). Enhanced Deep Residual Networks for Single Image Super-Resolution.

## Contributors

- Mayank Baber (BC201)
- Mayuresh Chougule (BC211)
---

## License

This project is licensed under the [MIT License](LICENSE).
