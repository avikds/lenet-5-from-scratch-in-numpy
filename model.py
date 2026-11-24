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

# Step 4 - conv_forward
def conv_forward(x, W, b, table=None):
    """
    Perform a valid, stride-1 convolution without flipping kernels.

    Parameters:
        x: Input feature maps of shape (n, c_in, H, W).
        W: Convolution kernels of shape (c_out, c_in, k, k).
        b: Bias vector of shape (c_out,).
        table: Optional Boolean connection table of shape
               (c_in, c_out). A connection is enabled when
               table[i, o] is True.

    Returns:
        Weighted sums of shape (n, c_out, H-k+1, W-k+1).
    """

    # Convert inputs to NumPy arrays.
    x = np.asarray(x)
    W = np.asarray(W)
    b = np.asarray(b)

    # Validate input dimensions.
    if x.ndim != 4:
        raise ValueError("x must have shape (n, c_in, H, W).")

    if W.ndim != 4:
        raise ValueError("W must have shape (c_out, c_in, k, k).")

    c_out, c_in, k_h, k_w = W.shape
    n, x_c_in, H, W_in = x.shape

    if x_c_in != c_in:
        raise ValueError("Input channels must match kernel channels.")

    if k_h != k_w:
        raise ValueError("Convolution kernels must be square.")

    if H < k_h or W_in < k_w:
        raise ValueError("Kernel dimensions cannot exceed input dimensions.")

    if b.shape != (c_out,):
        raise ValueError("b must have shape (c_out,).")

    # Apply the optional connection table.
    # table[i, o] determines whether output map o uses input map i.
    if table is not None:
        table = np.asarray(table, dtype=bool)

        if table.shape != (c_in, c_out):
            raise ValueError("table must have shape (c_in, c_out).")

        # Transpose the table to match the kernel layout (c_out, c_in).
        # Disabled connections receive zero-valued kernels.
        W = W * table.T[:, :, None, None]

    # Calculate the spatial dimensions of the valid convolution.
    H_out = H - k_h + 1
    W_out = W_in - k_w + 1

    # Extract all sliding kernel-sized windows.
    # The resulting shape is (n, c_in, H_out, W_out, k_h, k_w).
    windows = np.lib.stride_tricks.sliding_window_view(
        x,
        window_shape=(k_h, k_w),
        axis=(2, 3)
    )

    # Multiply each input window by its corresponding shared kernel
    # and sum over input channels and kernel dimensions.
    # No kernel flipping is performed.
    output = np.einsum(
        'niphwkl,oikl->nophw',
        windows,
        W,
        optimize=True
    ) if False else np.einsum(
        'n i p q u v, o i u v -> n o p q',
        windows,
        W,
        optimize=True
    )

    # Add one bias per output feature map.
    output += b[None, :, None, None]

    return output

# Step 5 - conv_backward
def conv_backward(dout, x, W, table=None):
    """
    Compute gradients of a valid, stride-1 convolution layer.

    Parameters:
        dout: Gradient of the loss with respect to the convolution
              output, shape (n, c_out, H_out, W_out).
        x: Original input feature maps, shape (n, c_in, H, W).
        W: Convolution kernels, shape (c_out, c_in, k, k).
        table: Optional Boolean connection table of shape
               (c_in, c_out).

    Returns:
        dx: Gradient with respect to x, same shape as x.
        dW: Gradient with respect to W, same shape as W.
        db: Gradient with respect to biases, shape (c_out,).
    """

    # Convert inputs to NumPy arrays with consistent precision.
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    dout = np.asarray(dout, dtype=np.float64)

    # Validate input dimensions.
    if x.ndim != 4:
        raise ValueError("x must have shape (n, c_in, H, W).")

    if W.ndim != 4:
        raise ValueError("W must have shape (c_out, c_in, k, k).")

    n, c_in, H, W_in = x.shape
    c_out, w_c_in, k_h, k_w = W.shape

    if w_c_in != c_in:
        raise ValueError("Input channels must match kernel channels.")

    if k_h != k_w:
        raise ValueError("Convolution kernels must be square.")

    if H < k_h or W_in < k_w:
        raise ValueError("Kernel dimensions cannot exceed input dimensions.")

    # Validate the optional connection table.
    if table is not None:
        table = np.asarray(table, dtype=bool)

        if table.shape != (c_in, c_out):
            raise ValueError("table must have shape (c_in, c_out).")

        # Convert the connection table to kernel layout.
        # np.where ensures disconnected kernels are completely ignored.
        connection_mask = table.T[:, :, None, None]
        W_effective = np.where(connection_mask, W, 0.0)
    else:
        connection_mask = None
        W_effective = W

    # Calculate the output spatial dimensions.
    H_out = H - k_h + 1
    W_out = W_in - k_w + 1

    # Validate the upstream gradient dimensions.
    expected_shape = (n, c_out, H_out, W_out)

    if dout.shape != expected_shape:
        raise ValueError(
            f"dout must have shape {expected_shape}, got {dout.shape}."
        )

    # ---------------------------------------------------------
    # 1. Gradient with respect to the input: dx
    # ---------------------------------------------------------

    # Initialize the input gradient.
    dx = np.zeros_like(x)

    # Each kernel position contributes to a corresponding region
    # of the input gradient.
    for u in range(k_h):
        for v in range(k_w):

            # Accumulate contributions from all output feature maps.
            dx[:, :, u:u + H_out, v:v + W_out] += np.einsum(
                'nohw,oi->nihw',
                dout,
                W_effective[:, :, u, v],
                optimize=True
            )

    # ---------------------------------------------------------
    # 2. Gradient with respect to the shared kernels: dW
    # ---------------------------------------------------------

    # Extract every sliding input window.
    windows = np.lib.stride_tricks.sliding_window_view(
        x,
        window_shape=(k_h, k_w),
        axis=(2, 3)
    )

    # Sum the gradient contributions over the batch and spatial
    # positions because each kernel is shared across all positions.
    dW = np.einsum(
        'nohw,nihwuv->oiuv',
        dout,
        windows,
        optimize=True
    )

    # Disconnected kernel pairs must have exactly zero gradients.
    if connection_mask is not None:
        dW = np.where(connection_mask, dW, 0.0)

    # ---------------------------------------------------------
    # 3. Gradient with respect to the output biases: db
    # ---------------------------------------------------------

    # Each output bias contributes to every spatial position
    # and every image in the batch.
    db = np.sum(dout, axis=(0, 2, 3))

    return dx, dW, db

