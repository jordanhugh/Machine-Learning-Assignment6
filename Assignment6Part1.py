import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def conv2d(array, kernel):
    n, n0 = array.shape
    assert n == n0
    k, k0 = kernel.shape
    assert k == k0
    
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros((n - k + 1, n - k + 1))

    for y in range(output.shape[1]):
        for x in range(output.shape[0]):
            output[y, x] = (kernel * array[y:y+k, x:x+k]).sum()
    return output

img_grey = cv2.imread('tcd.jpg', 0)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Original Image', fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.imshow(img_grey, cmap='gray')
plt.show()

kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
img_conv1 = conv2d(img_grey, kernel=kernel1)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Kernel 1', fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.imshow(img_conv1, cmap='gray')
plt.show()
cv2.imwrite('img_conv1.jpg', img_conv1)

kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])
img_conv2 = conv2d(img_grey, kernel=kernel2)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Kernel 2', fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.imshow(img_conv2, cmap='gray')
plt.show()
cv2.imwrite('img_conv2.jpg', img_conv2)