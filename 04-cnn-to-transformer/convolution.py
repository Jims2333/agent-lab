"""
Lab 1: Convolution: teaching a machine to "see edges" like a cat's visual cortex
================================================================================

In 1959, Hubel and Wiesel found that some neurons in a cat's visual cortex respond only to **edges
at a particular orientation**. The 1980 Neocognitron and LeCun's 1989 convolutional network both
imitated this kind of "small, local, edge-finding detector."

Here we write convolution by hand with numpy:
  1. Sweep a 3×3 "edge detector" across a small image and see where it responds
  2. Recreate Hubel and Wiesel's "orientation tuning curve": how strongly the same detector responds to lines at different angles
  3. See how pooling makes recognition "not mind a shift"
  4. Count how many parameters convolution saves compared with a fully connected layer

Run: python convolution.py
"""

import numpy as np


def conv2d(image, kernel):
    """The plainest 2D convolution (strictly speaking it's cross-correlation, but deep learning calls it convolution): slide a small window across the image and take a weighted sum at every position."""
    kh, kw = kernel.shape
    h, w = image.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


def max_pool(x, size=2):
    h, w = x.shape[0] // size, x.shape[1] // size
    return x[:h * size, :w * size].reshape(h, size, w, size).max(axis=(1, 3))


def show(x, title, chars=" .:-=+*#%@"):
    print(title)
    m = np.abs(x).max() or 1.0
    for row in x:
        print("   " + "".join(chars[int(abs(v) / m * (len(chars) - 1))] * 2 for v in row))
    print()


# ---------------------------------------------------------------- 1. a small image
img = np.zeros((14, 14))
img[3:11, 3:11] = 1.0              # a solid square
img[5:9, 5:9] = 0.0                # hollow out the middle, leaving a square frame

KERNELS = {
    "vertical edge": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),   # Sobel operator
    "horizontal edge": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
}

show(img, "[1] The original image: a hollow square (14×14 pixels)")
for name, k in KERNELS.items():
    out = np.maximum(conv2d(img, k), 0) + np.maximum(-conv2d(img, k), 0)   # count edges in both directions
    show(out, f"   After sweeping the 3×3 '{name}' detector across it (denser = stronger response):")
print("   The same small 3×3 detector (just 9 numbers) is reused at every position in the image.")
print("   The vertical detector lights up only on the left and right sides; the horizontal detector only on the top and bottom.\n")


# ---------------------------------------------------------------- 2. orientation tuning curve
def bar_image(angle_deg, size=21, width=0.8):
    """Draw a bright line through the center at the given angle (with soft edges, like a strip of light from a projector)."""
    c = size // 2
    t = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.abs((xx - c) * np.sin(t) - (c - yy) * np.cos(t))   # each pixel's distance from the line
    return np.exp(-dist ** 2 / (2 * width ** 2))


print("[2] Recreating Hubel and Wiesel's experiment: show the 'vertical edge' detector lines at different angles. How strong is its strongest response?")
responses = []
for angle in range(0, 181, 15):
    r = conv2d(bar_image(angle), KERNELS["vertical edge"])
    responses.append((angle, np.abs(r).max()))
top = max(v for _, v in responses)
for angle, v in responses:
    print(f"   line at {angle:3d}° {'█' * int(round(v / top * 30)):<30} {v / top:4.0%}")
print("   The closer the line is to vertical (90°), the stronger the response; it is weakest when the line is horizontal (0° and 180°).")
print("   This is an 'orientation-selective' neuron, just like the ones found in the cat's visual cortex.\n")

# ---------------------------------------------------------------- 3. pooling and translation invariance
print("[3] Pooling: if the image moves a little, is the result still stable?")
k = KERNELS["vertical edge"]
a = np.abs(conv2d(img, k))
shifted = np.roll(img, 1, axis=1)          # shift the whole image 1 pixel to the right
b = np.abs(conv2d(shifted, k))
pa, pb = max_pool(a, 4), max_pool(b, 4)
print(f"   After convolution: the two feature maps differ at {np.mean(np.abs(a - b) > 1e-9):.0%} of positions")
print(f"   After 4×4 max pooling as well: they differ at only {np.mean(np.abs(pa - pb) > 1e-9):.0%} of positions")
print("   Pooling only cares whether an edge shows up somewhere in a patch, not exactly where, so it barely notices small shifts.")
print("   Kunihiko Fukushima put this goal right in the title of his 1980 paper: 'pattern recognition unaffected by shift in position'.\n")

# ---------------------------------------------------------------- 4. parameter count
H = W = 224
full = (H * W) * (H * W)
conv = 3 * 3
print("[4] Why convolution: comparing parameter counts (for a 224×224 grayscale image)")
print(f"   Fully connected: every output pixel connects to every input pixel, which takes {full:,} weights")
print(f"   Convolution:     one 3×3 detector is shared across all positions and needs only {conv} weights")
print(f"   That's {full // conv:,} times fewer. This is the power of 'local connections + weight sharing'.")

print("""
Think about it:
  1. Design your own 3×3 "diagonal edge" detector. At what angle will its tuning curve peak?
  2. If you stack two convolution layers, how big a region of the original image can a neuron in the second layer "see"?
  3. In a CNN the detectors are learned, not designed by hand. What do you think the first layer looks like after training?""")
