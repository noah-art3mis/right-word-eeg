import cv2
import numpy as np
import matplotlib.pyplot as plt


def fft(path):
    # Load image in grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Compute FFT and shift zero-frequency component to center
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Get image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Center

    # Create a low-pass filter (Circular mask)
    radius = 50  # Change this for different levels of low-pass filtering
    low_pass_mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(low_pass_mask, (ccol, crow), radius, 1, thickness=-1)

    # Apply mask and inverse shift
    low_frequencies = dft_shift * low_pass_mask
    low_frequencies_shifted = np.fft.ifftshift(low_frequencies)
    low_frequencies_img = np.fft.ifft2(low_frequencies_shifted)
    low_frequencies_img = np.abs(low_frequencies_img)

    # Create a high-pass filter (1 - low-pass mask)
    high_pass_mask = 1 - low_pass_mask
    high_frequencies = dft_shift * high_pass_mask
    high_frequencies_shifted = np.fft.ifftshift(high_frequencies)
    high_frequencies_img = np.fft.ifft2(high_frequencies_shifted)
    high_frequencies_img = np.abs(high_frequencies_img)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap="gray"), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(low_frequencies_img, cmap="gray"), plt.title(
        "Low Frequencies"
    )
    plt.subplot(1, 3, 3), plt.imshow(high_frequencies_img, cmap="gray"), plt.title(
        "High Frequencies"
    )
    plt.show()
