from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import ndimage

#  image
image = plt.imread('/Users/jolie/Desktop/work/4-Algorithms/Image Segmentation/1.jpeg')

# Create a graphics window and define a layout of 1 row and 4 columns 创建一个图形窗口，定义 1 行 4 列的布局
plt.figure(figsize=(20, 5))  # Resize the figure to fit a 1 row, 4 column layout 调整 figure 的尺寸适应 1 行 4 列的布局

# 1. Display original image
plt.subplot(1, 4, 1)  # 1st of 4 rows 1行4列中的第1个
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')  # Hide Axes 隐藏坐标轴

# 2. Convert the image to grayscale 将图像转换为灰度图像
gray = rgb2gray(image)
plt.subplot(1, 4, 2)  # 2nd of 1 row and 4 columns 1行4列中的第2个
plt.imshow(gray, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

# 3. Threshold segmentation, binarize the pixel values ​​according to the average value 阈值分割，将像素值按平均值进行二值化
gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray_binary = gray_r.reshape(gray.shape[0], gray.shape[1])

plt.subplot(1, 4, 3)  # 3rd of 1 row and 4 columns 1行4列中的第3个
plt.imshow(gray_binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# 4. Set multiple thresholds to detect multiple objects 设置多个阈值，检测多个对象
gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray_multi = gray_r.reshape(gray.shape[0], gray.shape[1])

plt.subplot(1, 4, 4)  # 4th in row 1 and column 4 1行4列中的第4个
plt.imshow(gray_multi, cmap='gray')
plt.title('Multi-level Segmentation')
plt.axis('off')

# Show all subgraphs 显示所有子图
plt.tight_layout()  # Automatically adjust the layout of subplots to prevent overlapping 自动调整子图的布局以防止重叠
plt.show()
