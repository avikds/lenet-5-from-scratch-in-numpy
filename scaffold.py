"""
LeNet-5 from Scratch in NumPy scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""LeNet-5 from Scratch in NumPy (LeCun, Bottou, Bengio and Haffner, 1998).

Story: scale a few thousand MNIST digits the paper's way, build LeNet-5 with its
connection table, trainable subsampling and fixed RBF output codes, count its
60,000 parameters and 340,908 connections, and train it one image at a time
with the paper's MAP loss and stochastic diagonal Levenberg-Marquardt learning
rates. The test error is printed after every pass. Takes well under a minute on
the site's CPU.
"""
import numpy as np


# Stylized 7 x 12 bitmaps of the ten digits, one string per row ("#" = +1, "." = -1). The paper drew a bitmap
# like this for every character and used it, unchanged, as the parameter vector of that class's output unit.
# These drawings are in the spirit of the paper's bitmaps; the paper does not print its codes as numbers.
DIGIT_BITMAPS = [
    [".......", "..###..", ".##.##.", "##...##", "##...##", "##...##", "##...##", "##...##", "##...##", ".##.##.", "..###..", "......."],
    [".......", "...##..", "..###..", ".####..", "...##..", "...##..", "...##..", "...##..", "...##..", "...##..", ".######", "......."],
    [".......", ".#####.", "##...##", ".....##", ".....##", "....##.", "...##..", "..##...", ".##....", "##.....", "#######", "......."],
    [".......", "#######", ".....##", "....##.", "...##..", "..####.", ".....##", ".....##", ".....##", "##...##", ".#####.", "......."],
    [".......", "....##.", "...###.", "..####.", ".##.##.", "##..##.", "#######", "....##.", "....##.", "....##.", "....##.", "......."],
    [".......", "#######", "##.....", "##.....", "######.", ".....##", ".....##", ".....##", ".....##", "##...##", ".#####.", "......."],
    [".......", "..####.", ".##....", "##.....", "##.....", "######.", "##...##", "##...##", "##...##", "##...##", ".#####.", "......."],
    [".......", "#######", ".....##", ".....##", "....##.", "...##..", "..##...", "..##...", ".##....", ".##....", ".##....", "......."],
    [".......", ".#####.", "##...##", "##...##", "##...##", ".#####.", "##...##", "##...##", "##...##", "##...##", ".#####.", "......."],
    [".......", ".#####.", "##...##", "##...##", "##...##", "##...##", ".######", ".....##", ".....##", "....##.", ".####..", "......."],
]


def digit_codes():
    """The fixed RBF parameter vectors: a (10, 84) float array of +1 and -1, row d = the bitmap of digit d, read
    row by row."""
    return np.array([[1.0 if ch == "#" else -1.0 for row in rows for ch in row] for rows in DIGIT_BITMAPS])


def load_mnist(n_train, n_test, seed=0):
    """A random subset of MNIST: (x_train, y_train, x_test, y_test). Images are uint8 arrays of shape (n, 28, 28)
    with 0 for the white background and 255 for full ink; labels are int64 digits 0-9. The same seed always
    gives the same subset."""
    import dml_data
    ds = dml_data.load("mnist")
    rng = np.random.default_rng(seed)
    tr = rng.permutation(len(ds.y_train))[:n_train]
    te = rng.permutation(len(ds.y_test))[:n_test]
    return ds.x_train[tr], ds.y_train[tr].astype(np.int64), ds.x_test[te], ds.y_test[te].astype(np.int64)


def show_field(x):
    """Print one 32 x 32 input plane: '#' for ink above 0.6, '+' for lighter ink, '.' for background."""
    for row in x[0]:
        print("      " + "".join("#" if v > 0.6 else ("+" if v > 0.0 else ".") for v in row))


def main():
    N_TRAIN, N_TEST = 4000, 2000
    ETAS = [3e-5, 3e-5, 1.2e-5, 1.2e-5]

    # ---- 1. The input ----
    x_train, y_train, x_test, y_test = load_mnist(N_TRAIN, N_TEST, seed=0)
    x, xt = scale_input(x_train), scale_input(x_test)
    print(f"[1] MNIST subset: {N_TRAIN} training and {N_TEST} test digits, scaled to 32 x 32 fields")
    print(f"    the first training digit (label {int(y_train[0])}), background -0.1, full ink 1.175:")
    show_field(x[0])
    print(f"    over the training subset: min {x.min():.3f}, max {x.max():.3f}, mean {x.mean():.3f}, variance {x.var():.3f}")

    # ---- 2. The network ----
    rng = np.random.default_rng(0)
    params = init_lenet5(rng)
    T = connection_table()
    k = T.sum(axis=0)
    print("\n[2] LeNet-5, with the counts the paper gives")
    print("    layer   output          trainable parameters   connections")
    rows = [
        ("C1", "6 x 28 x 28", 6 * 26, 6 * 26 * 28 * 28),
        ("S2", "6 x 14 x 14", 6 * 2, 6 * 14 * 14 * 5),
        ("C3", "16 x 10 x 10", int((25 * k + 1).sum()), int((25 * k + 1).sum()) * 100),
        ("S4", "16 x 5 x 5", 16 * 2, 16 * 5 * 5 * 5),
        ("C5", "120", 120 * 401, 120 * 401),
        ("F6", "84", 84 * 121, 84 * 121),
        ("RBF", "10", 0, 10 * 84),
    ]
    for name, shape, n_par, n_con in rows:
        print(f"    {name:<7} {shape:<15} {n_par:>20,}   {n_con:>11,}")
    stored = sum(int(v.size) for key, v in params.items() if key not in ("RBF", "C3_W")) + int(T.sum()) * 25
    print(f"    total   {'':<15} {sum(r[2] for r in rows):>20,}   {sum(r[3] for r in rows):>11,}")
    print(f"    trainable values in the initialised network: {stored:,}")
    print("    C3 map j reads S2 maps:", " ".join("".join(str(i) for i in np.flatnonzero(T[:, j])) for j in range(16)))

    # ---- 3. The output codes ----
    print("\n[3] two of the fixed 7 x 12 output codes (the parameter vectors of the RBF units for 4 and 7)")
    for r in range(12):
        print("      " + DIGIT_BITMAPS[4][r] + "     " + DIGIT_BITMAPS[7][r])

    # ---- 4. Training ----
    print(f"\n[4] training one image at a time, MAP loss with j = 10, mu = 0.02, curvature re-estimated on 500 images per pass")
    print(f"    before training: test error {error_rate(params, xt, y_test):.3f}")
    for p_index, eta in enumerate(ETAS):
        h = gn_diag(params, x[:500])
        rates = np.concatenate([(eta / (0.02 + h[key])).ravel() for key in h])
        errors = train_sdlm(params, x, y_train, xt, y_test, [eta], rng)
        train_err = error_rate(params, x[:1000], y_train[:1000])
        print(f"    pass {p_index + 1}: eta {eta:.6f}, learning rates (estimated on 500 digits) {rates.min():.1e} to {rates.max():.1e}, "
              f"error on 1000 training digits {train_err:.3f}, test error {errors[0]:.3f}")

    # ---- 5. What the trained network says ----
    y, _ = lenet5_forward(params, xt[:4])
    print("\n[5] penalties y_i (smaller is better) for four test digits")
    for p_index in range(4):
        print(f"    label {int(y_test[p_index])}: answer {int(y[p_index].argmin())}, penalties " + " ".join(f"{v:5.1f}" for v in y[p_index]))
    preds = np.concatenate([lenet5_forward(params, xt[s:s + 200])[0].argmin(axis=1) for s in range(0, N_TEST, 200)])
    wrong = np.flatnonzero(preds != y_test)
    print(f"    {len(wrong)} of {N_TEST} test digits misread; per true digit:",
          " ".join(f"{d}:{int((y_test[wrong] == d).sum())}" for d in range(10)))
    if len(wrong):
        i = int(wrong[0])
        print(f"    a misread test digit (label {int(y_test[i])}, read as {int(preds[i])}):")
        show_field(xt[i])


if __name__ == "__main__":
    main()

