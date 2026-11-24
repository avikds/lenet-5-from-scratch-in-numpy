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

# Step 6 - subsample_forward
def subsample_forward(x, coef, bias):
    """
    Perform trainable 2 x 2 subsampling.

    Parameters:
        x: Input feature maps of shape (n, c, H, W).
        coef: Trainable coefficient for each feature map, shape (c,).
        bias: Trainable bias for each feature map, shape (c,).

    Returns:
        Output of shape (n, c, H/2, W/2).
    """

    # Convert inputs to float64 for numerical consistency.
    x = np.asarray(x, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    # Validate input dimensions.
    if x.ndim != 4:
        raise ValueError("x must have shape (n, c, H, W).")

    n, c, H, W = x.shape

    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("Input height and width must be even.")

    if coef.shape != (c,):
        raise ValueError("coef must have shape (c,).")

    if bias.shape != (c,):
        raise ValueError("bias must have shape (c,).")

    # Divide each feature map into non-overlapping 2 x 2 blocks.
    # Reshaped dimensions: (n, c, H/2, 2, W/2, 2).
    blocks = x.reshape(n, c, H // 2, 2, W // 2, 2)

    # Sum the four values in each 2 x 2 block.
    block_sums = np.sum(blocks, axis=(3, 5))

    # Apply the trainable coefficient and bias for each feature map.
    output = (
        coef[None, :, None, None] * block_sums
        + bias[None, :, None, None]
    )

    return output


def subsample_backward(dout, x, coef):
    """
    Compute gradients of the trainable subsampling layer.

    Parameters:
        dout: Gradient of the loss with respect to the output,
              shape (n, c, H/2, W/2).
        x: Original input feature maps, shape (n, c, H, W).
        coef: Trainable coefficient for each feature map,
              shape (c,).

    Returns:
        dx: Gradient with respect to x, same shape as x.
        dcoef: Gradient with respect to coef, shape (c,).
        dbias: Gradient with respect to bias, shape (c,).
    """

    # Convert inputs to float64.
    dout = np.asarray(dout, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)

    # Validate input dimensions.
    if x.ndim != 4:
        raise ValueError("x must have shape (n, c, H, W).")

    n, c, H, W = x.shape

    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("Input height and width must be even.")

    if coef.shape != (c,):
        raise ValueError("coef must have shape (c,).")

    expected_shape = (n, c, H // 2, W // 2)

    if dout.shape != expected_shape:
        raise ValueError(
            f"dout must have shape {expected_shape}, got {dout.shape}."
        )

    # Divide the original input into the same 2 x 2 blocks
    # used during the forward pass.
    blocks = x.reshape(n, c, H // 2, 2, W // 2, 2)

    # Sum each 2 x 2 block.
    block_sums = np.sum(blocks, axis=(3, 5))

    # ---------------------------------------------------------
    # 1. Gradient with respect to the input: dx
    # ---------------------------------------------------------

    # Every pixel in a 2 x 2 block receives the same gradient,
    # equal to dout multiplied by the feature map coefficient.
    scaled_dout = dout * coef[None, :, None, None]

    # Expand each gradient into its corresponding 2 x 2 block.
    dx = np.broadcast_to(
        scaled_dout.reshape(n, c, H // 2, 1, W // 2, 1),
        (n, c, H // 2, 2, W // 2, 2)
    ).reshape(n, c, H, W).copy()

    # ---------------------------------------------------------
    # 2. Gradient with respect to the coefficients: dcoef
    # ---------------------------------------------------------

    # Each coefficient is shared across all blocks and images.
    # Therefore, accumulate its gradient over the batch and
    # all spatial positions.
    dcoef = np.sum(
        dout * block_sums,
        axis=(0, 2, 3)
    )

    # ---------------------------------------------------------
    # 3. Gradient with respect to the biases: dbias
    # ---------------------------------------------------------

    # Each feature map has one shared bias.
    # Accumulate gradients over the batch and spatial positions.
    dbias = np.sum(dout, axis=(0, 2, 3))

    return dx, dcoef, dbias

# Step 7 - rbf_forward
def rbf_forward(x, centers):
    """
    Compute squared Euclidean distances for the LeNet-5 RBF output layer.

    Parameters:
        x: Input feature vectors of shape (n, d).
        centers: Fixed prototype vectors of shape (n_centers, d).

    Returns:
        y: Squared Euclidean distances of shape (n, n_centers).
    """

    # Convert inputs to float64 for numerical consistency.
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    # Validate input dimensions.
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n, d).")

    if centers.ndim != 2:
        raise ValueError("centers must be a 2D array of shape (n_centers, d).")

    if x.shape[1] != centers.shape[1]:
        raise ValueError("x and centers must have the same feature dimension.")

    # Calculate the difference between each input vector
    # and every prototype vector.
    diff = x[:, None, :] - centers[None, :, :]

    # Calculate squared Euclidean distances.
    # Sum over the feature dimension.
    y = np.sum(diff ** 2, axis=2)

    return y


def rbf_backward(dy, x, centers):
    """
    Compute the gradient of the RBF output layer with respect to x.

    Parameters:
        dy: Gradient with respect to RBF outputs,
            shape (n, n_centers).
        x: Original input feature vectors, shape (n, d).
        centers: Fixed prototype vectors, shape (n_centers, d).

    Returns:
        dx: Gradient with respect to x, shape (n, d).

    Note:
        The prototype vectors are fixed and receive no gradients.
    """

    # Convert inputs to float64 for numerical consistency.
    dy = np.asarray(dy, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    # Validate input dimensions.
    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")

    if centers.ndim != 2:
        raise ValueError("centers must be a 2D array.")

    if x.shape[1] != centers.shape[1]:
        raise ValueError("x and centers must have the same feature dimension.")

    n, d = x.shape
    n_centers = centers.shape[0]

    if dy.shape != (n, n_centers):
        raise ValueError(
            f"dy must have shape {(n, n_centers)}, got {dy.shape}."
        )

    # Calculate differences between inputs and prototype vectors.
    diff = x[:, None, :] - centers[None, :, :]

    # Apply the chain rule:
    # dE/dx = sum_i [dy_i * 2 * (x - centers_i)].
    dx = 2.0 * np.sum(
        dy[:, :, None] * diff,
        axis=1
    )

    return dx

# Step 8 - map_loss
def map_loss(y, labels, j):
    """
    Compute the LeNet-5 discriminative loss and its gradient.

    Parameters:
        y: RBF output penalties, shape (P, n_classes).
        labels: Correct class indices, shape (P,).
        j: Positive constant controlling the loss.

    Returns:
        E: Mean discriminative loss as a Python float.
        dy: Gradient of the loss with respect to y,
            shape (P, n_classes).
    """

    # Convert inputs to NumPy arrays for consistent computation.
    y = np.asarray(y, dtype=np.float64)
    labels = np.asarray(labels)

    # Validate input dimensions.
    if y.ndim != 2:
        raise ValueError("y must have shape (P, n_classes).")

    if labels.ndim != 1 or labels.shape[0] != y.shape[0]:
        raise ValueError("labels must have shape (P,).")

    P, n_classes = y.shape

    if P == 0 or n_classes == 0:
        raise ValueError("y must contain at least one sample and one class.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class indices.")

    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels contain an invalid class index.")

    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("y must contain finite, non-negative penalties.")

    if not np.isscalar(j) or not np.isfinite(j) or j <= 0:
        raise ValueError("j must be a positive finite scalar.")

    # ---------------------------------------------------------
    # 1. Compute the stable logarithmic denominator.
    # ---------------------------------------------------------

    # Construct the values whose exponentials appear in the loss:
    # exp(-j) + sum_i exp(-y[p, i]).
    logits = np.concatenate(
        (
            np.full((P, 1), -float(j)),
            -y
        ),
        axis=1
    )

    # Apply the log-sum-exp trick to prevent numerical underflow.
    row_max = np.max(logits, axis=1, keepdims=True)

    shifted_exp = np.exp(logits - row_max)

    log_denom = (
        row_max[:, 0]
        + np.log(np.sum(shifted_exp, axis=1))
    )

    # ---------------------------------------------------------
    # 2. Compute the mean discriminative loss.
    # ---------------------------------------------------------

    # Select the penalty corresponding to the correct class.
    correct_penalties = y[np.arange(P), labels]

    # Compute the loss for each image.
    sample_losses = correct_penalties + log_denom

    # Average across the batch and return a Python float.
    E = float(np.mean(sample_losses))

    # ---------------------------------------------------------
    # 3. Compute the gradient with respect to y.
    # ---------------------------------------------------------

    # Calculate the normalized exponential probabilities.
    probabilities = np.exp(-y - log_denom[:, None])

    # Start with the negative normalized exponential terms.
    dy = -probabilities

    # Add 1 to the gradient of each correct class penalty.
    dy[np.arange(P), labels] += 1.0

    # Account for averaging the loss across the batch.
    dy /= P

    return E, dy

# Step 9 - init_lenet5
def init_lenet5(rng):
    """
    Initialize the parameters of the LeNet-5 architecture.

    Parameters:
        rng: NumPy random number generator.

    Returns:
        params: Dictionary containing 12 trainable arrays
                and one fixed RBF prototype array.

    Initialization:
        Each trainable parameter is sampled uniformly from
        [-2.4 / F, 2.4 / F], where F is its unit's fan-in.
    """

    params = {}

    # ---------------------------------------------------------
    # 1. C1: First convolution layer
    # ---------------------------------------------------------
    # Shape: (6, 1, 5, 5)
    # Fan-in: 5 x 5 = 25

    params["C1_W"] = (
        rng.uniform(-1.0, 1.0, size=(6, 1, 5, 5))
        * (2.4 / 25)
    )

    # One bias per C1 feature map.
    params["C1_b"] = (
        rng.uniform(-1.0, 1.0, size=(6,))
        * (2.4 / 25)
    )

    # ---------------------------------------------------------
    # 2. S2: First trainable subsampling layer
    # ---------------------------------------------------------
    # Shape: (6,)
    # Fan-in: 2 x 2 = 4

    params["S2_c"] = (
        rng.uniform(-1.0, 1.0, size=(6,))
        * (2.4 / 4)
    )

    params["S2_b"] = (
        rng.uniform(-1.0, 1.0, size=(6,))
        * (2.4 / 4)
    )

    # ---------------------------------------------------------
    # 3. C3: Second convolution layer
    # ---------------------------------------------------------
    # Shape: (16, 6, 5, 5)
    # Fan-in depends on the number of connected S2 maps.

    T = connection_table()

    # Calculate the number of incoming S2 maps for each C3 map.
    # T has shape (6, 16), so summing over axis 0 gives (16,).
    # Fan-in for each C3 map = 25 * number of connected maps.
    F_c3 = 25 * np.sum(T, axis=0)

    # Draw the complete C3 kernel array using one random call.
    C3_W = rng.uniform(
        -1.0,
        1.0,
        size=(16, 6, 5, 5)
    )

    # Apply the individual fan-in scaling to each C3 output map.
    C3_W *= (2.4 / F_c3)[:, None, None, None]

    # Set all kernels corresponding to disconnected S2-C3 pairs to zero.
    # T.T has shape (16, 6), matching the first two kernel dimensions.
    C3_W = np.where(
        T.T[:, :, None, None],
        C3_W,
        0.0
    )

    params["C3_W"] = C3_W

    # Initialize C3 biases using the fan-in of each output map.
    params["C3_b"] = (
        rng.uniform(-1.0, 1.0, size=(16,))
        * (2.4 / F_c3)
    )

    # ---------------------------------------------------------
    # 4. S4: Second trainable subsampling layer
    # ---------------------------------------------------------
    # Shape: (16,)
    # Fan-in: 2 x 2 = 4

    params["S4_c"] = (
        rng.uniform(-1.0, 1.0, size=(16,))
        * (2.4 / 4)
    )

    params["S4_b"] = (
        rng.uniform(-1.0, 1.0, size=(16,))
        * (2.4 / 4)
    )

    # ---------------------------------------------------------
    # 5. C5: Third convolution layer
    # ---------------------------------------------------------
    # Shape: (120, 16, 5, 5)
    # Fan-in: 16 x 5 x 5 = 400

    params["C5_W"] = (
        rng.uniform(-1.0, 1.0, size=(120, 16, 5, 5))
        * (2.4 / 400)
    )

    params["C5_b"] = (
        rng.uniform(-1.0, 1.0, size=(120,))
        * (2.4 / 400)
    )

    # ---------------------------------------------------------
    # 6. F6: Fully connected layer
    # ---------------------------------------------------------
    # Weight shape: (120, 84)
    # Fan-in: 120

    params["F6_W"] = (
        rng.uniform(-1.0, 1.0, size=(120, 84))
        * (2.4 / 120)
    )

    params["F6_b"] = (
        rng.uniform(-1.0, 1.0, size=(84,))
        * (2.4 / 120)
    )

    # ---------------------------------------------------------
    # 7. RBF: Fixed output prototypes
    # ---------------------------------------------------------
    # Shape: (10, 84)
    # These are fixed digit codes, not trainable parameters.

    params["RBF"] = digit_codes()

    return params

# Step 10 - lenet5_forward
def lenet5_forward(params, x):
    """
    Run the complete LeNet-5 forward pass.

    Parameters:
        params: Dictionary containing the initialized LeNet-5 parameters.
        x: Input images of shape (n, 1, 32, 32).

    Returns:
        y: RBF output penalties, shape (n, 10).
        cache: Dictionary containing the input and intermediate
               activations required for backpropagation.
    """

    # Convert input to float64 for numerical consistency.
    x = np.asarray(x, dtype=np.float64)

    # Validate input dimensions.
    if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
        raise ValueError("x must have shape (n, 1, 32, 32).")

    n = x.shape[0]

    # ---------------------------------------------------------
    # Layer 1: C1 - First convolution layer
    # ---------------------------------------------------------
    # Input:  (n, 1, 32, 32)
    # Output: (n, 6, 28, 28)

    a1 = conv_forward(
        x,
        params["C1_W"],
        params["C1_b"]
    )

    h1 = squash(a1)

    # ---------------------------------------------------------
    # Layer 2: S2 - First trainable subsampling layer
    # ---------------------------------------------------------
    # Input:  (n, 6, 28, 28)
    # Output: (n, 6, 14, 14)

    a2 = subsample_forward(
        h1,
        params["S2_c"],
        params["S2_b"]
    )

    h2 = squash(a2)

    # ---------------------------------------------------------
    # Layer 3: C3 - Second convolution layer
    # ---------------------------------------------------------
    # Uses the partial connectivity table between S2 and C3.
    # Input:  (n, 6, 14, 14)
    # Output: (n, 16, 10, 10)

    a3 = conv_forward(
        h2,
        params["C3_W"],
        params["C3_b"],
        table=connection_table()
    )

    h3 = squash(a3)

    # ---------------------------------------------------------
    # Layer 4: S4 - Second trainable subsampling layer
    # ---------------------------------------------------------
    # Input:  (n, 16, 10, 10)
    # Output: (n, 16, 5, 5)

    a4 = subsample_forward(
        h3,
        params["S4_c"],
        params["S4_b"]
    )

    h4 = squash(a4)

    # ---------------------------------------------------------
    # Layer 5: C5 - Third convolution layer
    # ---------------------------------------------------------
    # Input:  (n, 16, 5, 5)
    # Output: (n, 120, 1, 1)
    #
    # Flatten the output into (n, 120) for the next layer.

    a5 = conv_forward(
        h4,
        params["C5_W"],
        params["C5_b"]
    ).reshape(n, 120)

    h5 = squash(a5)

    # ---------------------------------------------------------
    # Layer 6: F6 - Fully connected layer
    # ---------------------------------------------------------
    # Input:  (n, 120)
    # Output: (n, 84)

    a6 = h5 @ params["F6_W"] + params["F6_b"]

    h6 = squash(a6)

    # ---------------------------------------------------------
    # Layer 7: RBF - Euclidean output layer
    # ---------------------------------------------------------
    # Input:  (n, 84)
    # Output: (n, 10)
    #
    # The ten prototype vectors are fixed digit codes.

    y = rbf_forward(
        h6,
        params["RBF"]
    )

    # ---------------------------------------------------------
    # Store intermediate values for backpropagation.
    # ---------------------------------------------------------

    cache = {
        "x": x,
        "a1": a1,
        "h1": h1,
        "a2": a2,
        "h2": h2,
        "a3": a3,
        "h3": h3,
        "a4": a4,
        "h4": h4,
        "a5": a5,
        "h5": h5,
        "a6": a6,
        "h6": h6
    }

    return y, cache


def error_rate(params, x, labels, batch=200):
    """
    Calculate the classification error rate of LeNet-5.

    Parameters:
        params: Dictionary containing LeNet-5 parameters.
        x: Input images of shape (n, 1, 32, 32).
        labels: Correct digit labels of shape (n,).
        batch: Maximum number of images processed at once.

    Returns:
        Error rate as a Python float.

    Prediction:
        The predicted digit is the class with the smallest
        squared Euclidean RBF penalty.
    """

    # Convert inputs to NumPy arrays.
    x = np.asarray(x)
    labels = np.asarray(labels)

    # Validate the batch size.
    if not isinstance(batch, (int, np.integer)) or batch <= 0:
        raise ValueError("batch must be a positive integer.")

    # Validate image and label counts.
    if x.ndim != 4:
        raise ValueError("x must have shape (n, 1, 32, 32).")

    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError("labels must have shape (n,).")

    n = len(x)

    if n == 0:
        raise ValueError("Input dataset must not be empty.")

    # Count the number of incorrectly classified images.
    incorrect = 0

    # Process the dataset in batches to limit memory usage.
    for start in range(0, n, batch):

        # Determine the end of the current batch.
        end = min(start + batch, n)

        # Run the forward pass on the current batch.
        y, _ = lenet5_forward(
            params,
            x[start:end]
        )

        # Select the digit corresponding to the smallest penalty.
        predictions = np.argmin(y, axis=1)

        # Compare predictions against the correct labels.
        incorrect += int(
            np.sum(predictions != labels[start:end])
        )

    # Return the fraction of incorrectly classified images.
    return float(incorrect / n)

# Step 11 - lenet5_backward
def lenet5_backward(params, cache, dy):
    """
    Perform backpropagation through the complete LeNet-5 network.

    Parameters:
        params: Dictionary containing the LeNet-5 parameters.
        cache: Dictionary of intermediate activations from lenet5_forward().
        dy: Gradient of the loss with respect to the RBF penalties,
            shape (n, 10).

    Returns:
        grads: Dictionary containing gradients for all 12 trainable
               parameter arrays, with shapes matching their parameters.

    Note:
        The RBF prototype vectors are fixed and do not receive gradients.
        Gradients are accumulated over the batch.
    """

    # Convert the upstream gradient to float64.
    dy = np.asarray(dy, dtype=np.float64)

    # Retrieve the original input and intermediate activations.
    x = cache["x"]

    a1, h1 = cache["a1"], cache["h1"]
    a2, h2 = cache["a2"], cache["h2"]
    a3, h3 = cache["a3"], cache["h3"]
    a4, h4 = cache["a4"], cache["h4"]
    a5, h5 = cache["a5"], cache["h5"]
    a6, h6 = cache["a6"], cache["h6"]

    # Validate the upstream gradient.
    n = x.shape[0]

    if dy.shape != (n, 10):
        raise ValueError("dy must have shape (n, 10).")

    # Initialize the dictionary of trainable parameter gradients.
    grads = {}

    # ---------------------------------------------------------
    # Layer 7: RBF - Euclidean output layer
    # ---------------------------------------------------------
    # Propagate gradients from the ten output penalties
    # back to the 84 F6 activations.

    dh6 = rbf_backward(
        dy,
        h6,
        params["RBF"]
    )

    # ---------------------------------------------------------
    # Layer 6: F6 - Fully connected layer
    # ---------------------------------------------------------
    # h6 = squash(h5 @ F6_W + F6_b)

    da6 = dh6 * squash_prime(a6)

    # Gradient with respect to F6 weights.
    # Accumulate contributions across the batch.
    grads["F6_W"] = h5.T @ da6

    # Gradient with respect to F6 biases.
    grads["F6_b"] = np.sum(da6, axis=0)

    # Propagate gradients to the C5 activations.
    dh5 = da6 @ params["F6_W"].T

    # ---------------------------------------------------------
    # Layer 5: C5 - Third convolution layer
    # ---------------------------------------------------------
    # h5 = squash(a5)
    # a5 is the flattened output of C5.
    # Restore its original spatial dimensions for convolution
    # backpropagation: (n, 120, 1, 1).

    da5 = dh5 * squash_prime(a5)

    da5_conv = da5.reshape(n, 120, 1, 1)

    # Compute gradients with respect to C5 inputs and parameters.
    dh4_conv, grads["C5_W"], grads["C5_b"] = conv_backward(
        da5_conv,
        h4,
        params["C5_W"]
    )

    # Restore the subsampling layer activation shape.
    dh4 = dh4_conv

    # ---------------------------------------------------------
    # Layer 4: S4 - Second trainable subsampling layer
    # ---------------------------------------------------------
    # h4 = squash(a4)

    da4 = dh4 * squash_prime(a4)

    # Compute input, coefficient and bias gradients.
    dh3, grads["S4_c"], grads["S4_b"] = subsample_backward(
        da4,
        h3,
        params["S4_c"]
    )

    # ---------------------------------------------------------
    # Layer 3: C3 - Second convolution layer
    # ---------------------------------------------------------
    # h3 = squash(a3)
    # C3 uses the original partial connectivity table.

    da3 = dh3 * squash_prime(a3)

    dh2, grads["C3_W"], grads["C3_b"] = conv_backward(
        da3,
        h2,
        params["C3_W"],
        table=connection_table()
    )

    # ---------------------------------------------------------
    # Layer 2: S2 - First trainable subsampling layer
    # ---------------------------------------------------------
    # h2 = squash(a2)

    da2 = dh2 * squash_prime(a2)

    # Compute input, coefficient and bias gradients.
    dh1, grads["S2_c"], grads["S2_b"] = subsample_backward(
        da2,
        h1,
        params["S2_c"]
    )

    # ---------------------------------------------------------
    # Layer 1: C1 - First convolution layer
    # ---------------------------------------------------------
    # h1 = squash(a1)

    da1 = dh1 * squash_prime(a1)

    # Compute the final convolution gradients.
    # The gradient with respect to the original image is not
    # needed because the input pixels are not trainable.
    dx, grads["C1_W"], grads["C1_b"] = conv_backward(
        da1,
        x,
        params["C1_W"]
    )

    # Return gradients for all 12 trainable parameter arrays.
    # No gradient is returned for the fixed RBF prototypes.
    return grads

# Step 12 - gn_diag
def gn_diag(params, x, chunk=100):
    """
    Estimate the diagonal of the Gauss-Newton curvature matrix
    for every trainable parameter in LeNet-5.

    Parameters:
        params: Dictionary containing initialized LeNet-5 parameters.
        x: Scaled input images of shape (n, 1, 32, 32).
        chunk: Number of images processed per chunk.

    Returns:
        h: Dictionary containing the average Gauss-Newton
           diagonal estimates for all 12 trainable arrays.

    Notes:
        - Uses a second derivative of 2 at every F6 state.
        - Propagates squared derivatives through the network.
        - Uses squared weights and squared layer inputs.
        - Preserves C3 partial connectivity.
        - Excludes the fixed RBF prototypes.
        - Accumulates curvature across chunks before averaging.
    """

    # Convert inputs to float64.
    x = np.asarray(x, dtype=np.float64)

    # Validate input dimensions.
    if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
        raise ValueError("x must have shape (n, 1, 32, 32).")

    if not isinstance(chunk, (int, np.integer)) or chunk <= 0:
        raise ValueError("chunk must be a positive integer.")

    n = x.shape[0]

    if n == 0:
        raise ValueError("At least one input image is required.")

    # Define all twelve trainable parameter keys.
    trainable_keys = (
        "C1_W", "C1_b",
        "S2_c", "S2_b",
        "C3_W", "C3_b",
        "S4_c", "S4_b",
        "C5_W", "C5_b",
        "F6_W", "F6_b"
    )

    # Initialize accumulated curvature estimates.
    h = {
        key: np.zeros_like(params[key], dtype=np.float64)
        for key in trainable_keys
    }

    # Process the dataset in chunks to reduce memory usage.
    for start in range(0, n, chunk):

        end = min(start + chunk, n)
        x_batch = x[start:end]

        # -----------------------------------------------------
        # 1. Forward pass
        # -----------------------------------------------------

        _, cache = lenet5_forward(params, x_batch)

        batch_size = x_batch.shape[0]

        # -----------------------------------------------------
        # 2. Initialize curvature at the F6 output states
        # -----------------------------------------------------

        # The second derivative of the squared Euclidean
        # distance with respect to each F6 state is 2.
        curv_h6 = np.full(
            (batch_size, 84),
            2.0,
            dtype=np.float64
        )

        # -----------------------------------------------------
        # 3. F6: Fully connected layer
        # -----------------------------------------------------

        curv_a6 = (
            curv_h6 * squash_prime(cache["a6"]) ** 2
        )

        # Accumulate F6 weight curvature.
        h["F6_W"] += (
            cache["h5"] ** 2
        ).T @ curv_a6

        # Accumulate F6 bias curvature.
        h["F6_b"] += np.sum(curv_a6, axis=0)

        # Propagate curvature using squared F6 weights.
        curv_h5 = curv_a6 @ (
            params["F6_W"] ** 2
        ).T

        # -----------------------------------------------------
        # 4. C5: Third convolution layer
        # -----------------------------------------------------

        curv_a5 = (
            curv_h5 * squash_prime(cache["a5"]) ** 2
        )

        curv_a5 = curv_a5.reshape(
            batch_size, 120, 1, 1
        )

        # Calculate the current chunk's C5 curvature.
        curv_h4, dC5_W, dC5_b = conv_backward(
            curv_a5,
            cache["h4"] ** 2,
            params["C5_W"] ** 2
        )

        # Accumulate instead of overwriting previous chunks.
        h["C5_W"] += dC5_W
        h["C5_b"] += dC5_b

        # -----------------------------------------------------
        # 5. S4: Second trainable subsampling layer
        # -----------------------------------------------------

        curv_a4 = (
            curv_h4 * squash_prime(cache["a4"]) ** 2
        )

        # Calculate the current chunk's S4 curvature.
        curv_h3, dS4_c, dS4_b = subsample_backward(
            curv_a4,
            cache["h3"] ** 2,
            params["S4_c"] ** 2
        )

        # Accumulate coefficient and bias curvature.
        h["S4_c"] += dS4_c
        h["S4_b"] += dS4_b

        # -----------------------------------------------------
        # 6. C3: Second convolution layer
        # -----------------------------------------------------

        curv_a3 = (
            curv_h3 * squash_prime(cache["a3"]) ** 2
        )

        # Apply the original C3 partial connection table.
        curv_h2, dC3_W, dC3_b = conv_backward(
            curv_a3,
            cache["h2"] ** 2,
            params["C3_W"] ** 2,
            table=connection_table()
        )

        # Accumulate C3 curvature across all chunks.
        h["C3_W"] += dC3_W
        h["C3_b"] += dC3_b

        # -----------------------------------------------------
        # 7. S2: First trainable subsampling layer
        # -----------------------------------------------------

        curv_a2 = (
            curv_h2 * squash_prime(cache["a2"]) ** 2
        )

        # Calculate the current chunk's S2 curvature.
        curv_h1, dS2_c, dS2_b = subsample_backward(
            curv_a2,
            cache["h1"] ** 2,
            params["S2_c"] ** 2
        )

        # Accumulate S2 coefficient and bias curvature.
        h["S2_c"] += dS2_c
        h["S2_b"] += dS2_b

        # -----------------------------------------------------
        # 8. C1: First convolution layer
        # -----------------------------------------------------

        curv_a1 = (
            curv_h1 * squash_prime(cache["a1"]) ** 2
        )

        # Calculate the current chunk's C1 curvature.
        _, dC1_W, dC1_b = conv_backward(
            curv_a1,
            x_batch ** 2,
            params["C1_W"] ** 2
        )

        # Accumulate C1 curvature across all chunks.
        h["C1_W"] += dC1_W
        h["C1_b"] += dC1_b

    # ---------------------------------------------------------
    # 9. Average curvature over the entire dataset
    # ---------------------------------------------------------

    for key in trainable_keys:
        h[key] /= n

        # Ensure non-negative curvature estimates.
        h[key] = np.maximum(h[key], 0.0)

    return h

# Step 13 - train_sdlm
def train_sdlm(
    params,
    x,
    labels,
    x_test,
    labels_test,
    etas,
    rng,
    mu=0.02,
    j=10.0,
    n_hess=500
):
    """
    Train LeNet-5 using Stochastic Diagonal Levenberg-Marquardt.

    Parameters:
        params: Dictionary containing the initialized LeNet-5 parameters.
        x: Scaled training images, shape (n, 1, 32, 32).
        labels: Training labels, shape (n,).
        x_test: Scaled test images, shape (n_test, 1, 32, 32).
        labels_test: Test labels, shape (n_test,).
        etas: Global learning rate for each training pass.
        rng: NumPy random number generator.
        mu: Levenberg-Marquardt damping parameter.
        j: Positive constant used in the discriminative loss.
        n_hess: Number of randomly selected images used for
                Gauss-Newton curvature estimation.

    Returns:
        errors: List containing the test error after each pass.

    Training procedure:
        - Estimate Gauss-Newton curvature once per pass.
        - Calculate an individual learning rate for each parameter.
        - Shuffle the training images.
        - Update parameters after every individual image.
        - Keep the fixed RBF prototypes unchanged.
    """

    # Convert datasets and labels into NumPy arrays.
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)

    x_test = np.asarray(x_test, dtype=np.float64)
    labels_test = np.asarray(labels_test)

    # ---------------------------------------------------------
    # 1. Validate the training data
    # ---------------------------------------------------------

    if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
        raise ValueError("x must have shape (n, 1, 32, 32).")

    if x_test.ndim != 4 or x_test.shape[1:] != (1, 32, 32):
        raise ValueError("x_test must have shape (n_test, 1, 32, 32).")

    if labels.shape != (len(x),):
        raise ValueError("labels must have shape (n,).")

    if labels_test.shape != (len(x_test),):
        raise ValueError("labels_test must have shape (n_test,).")

    if len(x) == 0 or len(x_test) == 0:
        raise ValueError("Training and test datasets must not be empty.")

    if not isinstance(n_hess, (int, np.integer)) or n_hess <= 0:
        raise ValueError("n_hess must be a positive integer.")

    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be a positive finite value.")

    if not np.isfinite(j) or j <= 0:
        raise ValueError("j must be a positive finite value.")

    # Identify the twelve trainable parameter arrays.
    # The RBF prototypes are fixed and must never be updated.
    trainable_keys = [
        key for key in params if key != "RBF"
    ]

    if len(trainable_keys) != 12:
        raise ValueError("Expected exactly 12 trainable parameter arrays.")

    # Store test errors after each training pass.
    errors = []

    # ---------------------------------------------------------
    # 2. Training passes
    # ---------------------------------------------------------

    for eta in etas:

        # -----------------------------------------------------
        # Step 1: Estimate the Gauss-Newton curvature
        # -----------------------------------------------------

        # Randomly select up to n_hess training images.
        pick = rng.permutation(len(x))[:n_hess]

        # Estimate the average diagonal curvature for the
        # current network parameters.
        h = gn_diag(
            params,
            x[pick]
        )

        # -----------------------------------------------------
        # Step 2: Calculate individual learning rates
        # -----------------------------------------------------

        # Each trainable parameter receives its own elementwise
        # learning rate based on its Gauss-Newton curvature.
        #
        # These learning rates remain fixed throughout this pass.
        rates = {
            key: eta / (mu + h[key])
            for key in trainable_keys
        }

        # -----------------------------------------------------
        # Step 3: Randomly shuffle the training images
        # -----------------------------------------------------

        order = rng.permutation(len(x))

        # -----------------------------------------------------
        # Step 4: Online stochastic parameter updates
        # -----------------------------------------------------

        # Process exactly one training image at a time.
        for i in order:

            # Select the current training image.
            xi = x[i:i + 1]
            yi = labels[i:i + 1]

            # Forward propagation through LeNet-5.
            y_pred, cache = lenet5_forward(
                params,
                xi
            )

            # Calculate the paper's discriminative loss
            # and its gradient with respect to the RBF penalties.
            _, dy = map_loss(
                y_pred,
                yi,
                j
            )

            # Backpropagate the loss through all seven layers.
            grads = lenet5_backward(
                params,
                cache,
                dy
            )

            # Update every trainable parameter immediately.
            # The same learning rates are used throughout this pass.
            for key in trainable_keys:

                params[key] -= rates[key] * grads[key]

        # -----------------------------------------------------
        # Step 5: Evaluate the trained network
        # -----------------------------------------------------

        # Calculate test classification error after this pass.
        test_error = error_rate(
            params,
            x_test,
            labels_test
        )

        # Store the result as a Python float.
        errors.append(float(test_error))

    # Return test errors from all training passes.
    return errors

