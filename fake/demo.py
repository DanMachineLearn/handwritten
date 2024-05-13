import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度
image_path = 'handwritten_chinese.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 二值化图像
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 计算水平投影函数
horizontal_projection = np.sum(binary_image, axis=1)  # 在垂直方向上进行像素值求和

# 计算垂直投影函数
vertical_projection = np.sum(binary_image, axis=0)  # 在水平方向上进行像素值求和

# 绘制投影函数图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(horizontal_projection, range(binary_image.shape[0]))
plt.title('Horizontal Projection')
plt.xlabel('Sum of Pixel Values')
plt.ylabel('Row')

plt.subplot(1, 2, 2)
plt.plot(range(binary_image.shape[1]), vertical_projection)
plt.title('Vertical Projection')
plt.xlabel('Column')
plt.ylabel('Sum of Pixel Values')

plt.tight_layout()
plt.show()
