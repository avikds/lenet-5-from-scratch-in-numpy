# LeNet-5 from Scratch in NumPy

Gradient-Based Learning Applied to Document Recognition (LeCun, Bottou, Bengio and Haffner, 1998), rebuilt in NumPy with every gradient written by hand. Build the paper's LeNet-5 exactly as it describes it: a 32 x 32 input scaled so the background is -0.1 and the ink is 1.175, the squashing function 1.7159 tanh(2a/3), convolution layers C1, C3 and C5, subsampling layers S2 and S4 with one trainable coefficient and bias per map, the paper's connection table that wires each C3 map to only some of the S2 maps, the 84-unit layer F6, and an output of Euclidean radial basis function units whose parameter vectors are fixed 7 x 12 bitmaps of the digits. Write the forward and backward pass of every layer, check them against finite differences, and count the network's 60,000 trainable parameters. Then train it the way the paper did: one image at a time, with the paper's discriminative loss and the stochastic diagonal Levenberg-Marquardt method, which gives every parameter its own learning rate from a Gauss-Newton estimate of the second derivative. On a few thousand real MNIST digits the test error falls from about 90% to a few percent in a few passes on a CPU.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** scale_input

---

Built on Deep-ML.
