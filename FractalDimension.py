import numpy as np
import cv2

# https://github.com/brian-xu/FractalDimension?tab=readme-ov-file

import numpy as np


def fractal_dimension(image: np.ndarray) -> np.float64:
    """Calculates the fractal dimension of an image using a modified box-counting algorithm.

    Args:
        image: A 2D numpy array (grayscale image, square format).

    Returns:
        D: The fractal dimension (Df).
    """
    M = image.shape[0]  # Image dimensions (assumed square)
    G_min = int(image.min())  # Convert to Python int to prevent uint8 overflow
    G_max = int(image.max())  # Convert to Python int
    G = G_max - G_min + 1  # Number of gray levels (typically 256)
    prev = -1  # For plateau detection
    r_Nr = []

    for L in range(2, (M // 2) + 1):
        h = max(1, int(G) // max(1, (M // L)))  # Ensure valid height calculation
        N_r = 0
        r = L / M
        for i in range(0, M, L):
            boxes = [[] for _ in range((G + h - 1) // h)]  # Ensure unique lists

            for row in image[i : i + L]:
                for pixel in row[i : i + L]:
                    height = (int(pixel) - G_min) // h  # Convert pixel to int
                    boxes[int(height)].append(pixel)

            boxes_array = [np.array(b, dtype=np.float64) for b in boxes if len(b) > 0]
            stddev = np.sqrt(
                np.array([np.var(b) for b in boxes_array], dtype=np.float64)
            )

            nBox_r = 2 * (stddev // h) + 1
            N_r += np.sum(nBox_r)

        if N_r != prev:
            r_Nr.append([r, N_r])
            prev = N_r

    x = np.array([np.log(1 / point[0]) for point in r_Nr], dtype=np.float64)
    y = np.array([np.log(point[1]) for point in r_Nr], dtype=np.float64)

    D = np.polyfit(x, y, 1)[0]  # Compute slope of best fit line
    return D


def crop_to_square(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read.")

    # Get dimensions
    height, width = img.shape[:2]
    size = min(height, width)  # Smallest dimension

    # Compute the center crop coordinates
    start_x = (width - size) // 2
    start_y = (height - size) // 2

    # Crop the image
    cropped_img = img[start_y : start_y + size, start_x : start_x + size]

    # Save the squared image
    cv2.imwrite(output_path, cropped_img)


def convert_to_greyscale(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read.")

    # Convert to greyscale
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save the processed image
    cv2.imwrite(output_path, greyscale_img)


import numpy as np
import pylab as pl

# https://plos.figshare.com/articles/code/Python()_code_of_the_class_Fractal_that_calculates_the_fractal_dimension_through_the_box_counting_/22935765?file=40671303


def fractal_dimension():
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    image = rgb2gray(pl.imread("Sierpinski.png"))

    # finding all the non-zero pixels
    pixels = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:
                pixels.append((i, j))

    Lx = image.shape[1]
    Ly = image.shape[0]
    print(Lx, Ly)
    pixels = pl.array(pixels)
    print(pixels.shape)

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    Ns = []
    # looping over several scales
    for scale in scales:
        print("======= Scale :", scale)
        # computing the histogram
        H, edges = np.histogramdd(
            pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale))
        )
        Ns.append(np.sum(H > 0))

    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

    pl.plot(np.log(scales), np.log(Ns), "o", mfc="none")
    pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
    pl.xlabel("log epsilon")
    pl.ylabel("log N")
    print(
        "The Hausdorff dimension is", -coeffs[0]
    )  # the fractal dimension is the OPPOSITE of the fitting coefficient
