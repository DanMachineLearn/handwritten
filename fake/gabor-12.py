import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载jpg图像
image_path = 'handwritten_chinese.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 加载为灰度图像

# 检查图像是否成功加载
if image is None:
    raise ValueError("无法加载图像，请检查路径是否正确")

# 定义Gabor滤波器的参数
ksize = (31, 31)  # 滤波器的大小
sigma = 4.0  # 标准差
theta = 0  # 初始角度
lambd = 10.0  # 波长
gamma = 0.5  # 纵横比
psi = 0  # 相位偏移

# 创建Gabor滤波器
gabor_bank = []
for angle in np.linspace(0, np.pi, 12, endpoint=False):  # 分12个角度
    gabor_filter = cv2.getGaborKernel(ksize, sigma, angle, lambd, gamma, psi, ktype=cv2.CV_32F)
    gabor_bank.append(gabor_filter)

# 应用Gabor滤波器
gabor_images = []
for gabor_filter in gabor_bank:
    filtered_img = cv2.filter2D(image, cv2.CV_32F, gabor_filter)  # 应用滤波器
    gabor_images.append(filtered_img)

# 绘制结果
plt.figure(figsize=(15, 10))
plt.subplot(3, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

for i, gabor_img in enumerate(gabor_images):
    plt.subplot(3, 4, i + 1)
    plt.imshow(gabor_img, cmap='gray')
    plt.title(f"Gabor Angle {i * 30}°")  # 0到330度，每30度一个角度
    plt.axis('off')

plt.tight_layout()
plt.show()
