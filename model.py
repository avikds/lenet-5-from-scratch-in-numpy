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

# Step 2 - squash
def squash(a):
    """
    LeNet-5 scaled hyperbolic tangent activation function.

    f(a) = 1.7159 * tanh(2a / 3)

    Supports scalar values and NumPy arrays of any shape.
    """

    # Convert input to float64 for consistent numerical precision.
    a = np.asarray(a, dtype=np.float64)

    # Apply the scaled hyperbolic tangent activation.
    return 1.7159 * np.tanh((2.0 / 3.0) * a)


def squash_prime(a):
    """
    Derivative of the LeNet-5 scaled hyperbolic tangent function.

    f'(a) = 1.7159 * (2/3) * (1 - tanh(2a/3)^2)

    Supports scalar values and NumPy arrays of any shape.
    """

    # Convert input to float64 for consistent numerical precision.
    a = np.asarray(a, dtype=np.float64)

    # Calculate tanh once and reuse it in the derivative.
    t = np.tanh((2.0 / 3.0) * a)

    # Apply the derivative using the chain rule.
    return 1.7159 * (2.0 / 3.0) * (1.0 - t**2)

# Step 3 - connection_table
def connection_table():
    """
    Return the LeNet-5 connection table between S2 and C3.

    Returns:
        T: Boolean NumPy array of shape (6, 16).
           T[i, j] is True when C3 map j receives input
           from S2 map i.
    """

    # Initialize the connection table with all connections disabled.
    T = np.zeros((6, 16), dtype=bool)

    # C3 maps 0-5: Each connects to three consecutive S2 maps,
    # with circular wraparound.
    for j in range(6):
        for offset in range(3):
            i = (j + offset) % 6
            T[i, j] = True

    # C3 maps 6-11: Each connects to four consecutive S2 maps,
    # with circular wraparound.
    for j in range(6, 12):
        start = j - 6
        for offset in range(4):
            i = (start + offset) % 6
            T[i, j] = True

    # C3 map 12: Connects to S2 maps {0, 1, 3, 4}.
    T[[0, 1, 3, 4], 12] = True

    # C3 map 13: Connects to S2 maps {1, 2, 4, 5}.
    T[[1, 2, 4, 5], 13] = True

    # C3 map 14: Connects to S2 maps {0, 2, 3, 5}.
    T[[0, 2, 3, 5], 14] = True

    # C3 map 15: Connects to all six S2 maps.
    T[:, 15] = True

    return T

