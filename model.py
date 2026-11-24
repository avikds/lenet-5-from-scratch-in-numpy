"""
LeNet-5 from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - scale_input
import numpy as np

def scale_input(images):
    """
    Convert MNIST images into LeNet-5 input format.

    Parameters:
        images: NumPy array of shape (n, 28, 28), containing
                grayscale pixel values from 0 (white background)
                to 255 (full black ink).

    Returns:
        A float64 NumPy array of shape (n, 1, 32, 32),
        with a 2-pixel background border and pixel values
        scaled from -0.1 to 1.175.
    """

    # Convert input to a NumPy array without modifying the original.
    images = np.asarray(images)

    # Validate the input dimensions.
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Input images must have shape (n, 28, 28).")

    n = images.shape[0]

    # Linearly scale pixel values from [0, 255] to [-0.1, 1.175].
    scaled_images = (
        images.astype(np.float64) / 255.0
    ) * 1.275 - 0.1

    # Initialize the 32 x 32 input field with background values.
    output = np.full((n, 1, 32, 32), -0.1, dtype=np.float64)

    # Place each 28 x 28 image in the center, leaving a 2-pixel border.
    output[:, 0, 2:30, 2:30] = scaled_images

    return output

