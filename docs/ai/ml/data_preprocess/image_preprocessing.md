
# 图像数据预处理技术深度解析


## 1. 基础预处理

### 1.1 尺寸调整与插值

#### 数学原理
- 双线性插值：
  对于目标点 $(x,y)$，其值由四个最近邻点加权计算：
  $$
  f(x,y) = \frac{1}{(x_2 - x_1)(y_2 - y_1)} \left[ \begin{matrix} x_2 - x & x - x_1 \end{matrix} \right]
  \left[ \begin{matrix} f(Q_{11}) & f(Q_{12}) \\ f(Q_{21}) & f(Q_{22}) \end{matrix} \right]
  \left[ \begin{matrix} y_2 - y \\ y - y_1 \end{matrix} \right]
  $$
  其中 $Q_{ij}$ 为四个相邻像素坐标

#### Python实现
```python
import cv2

# 双线性插值缩放
def resize_bilinear(img, new_size):
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

# 示例
img = cv2.imread('image.jpg')
resized = resize_bilinear(img, (224, 224))
```

### 1.2 颜色空间转换

#### RGB转灰度公式
$$
Y = 0.299R + 0.587G + 0.114B
$$
#### RGB转HSV公式
$$
\begin{aligned}
V &= \max(R,G,B) \\
S &= \begin{cases} 
\frac{V - \min(R,G,B)}{V} & V \neq 0 \\
0 & \text{otherwise}
\end{cases} \\
H &= \begin{cases}
60^\circ \times \frac{G - B}{V - \min(R,G,B)} & V = R \\
60^\circ \times \left( 2 + \frac{B - R}{V - \min(R,G,B)} \right) & V = G \\
60^\circ \times \left( 4 + \frac{R - G}{V - \min(R,G,B)} \right) & V = B
\end{cases}
\end{aligned}
$$

#### 代码实现
```python
def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
```



## 2. 数据增强

### 2.1 几何变换

#### 仿射变换矩阵
$$
M = \begin{bmatrix}
a_{11} & a_{12} & t_x \\
a_{21} & a_{22} & t_y \\
0 & 0 & 1
\end{bmatrix}
$$
包含旋转 ($θ$)、缩放 ($s$)、剪切 ($sh$) 和平移 ($t_x,t_y$) 操作

#### Python实现
```python
import albumentations as A

transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2)
])

augmented = transform(image=img)['image']
```

### 2.2 色彩扰动

#### 颜色抖动公式
$$
\begin{aligned}
R' &= R \times (1 + \delta_r) + \epsilon_r \\
G' &= G \times (1 + \delta_g) + \epsilon_g \\
B' &= B \times (1 + \delta_b) + \epsilon_b
\end{aligned}
$$
其中 $\delta \sim \mathcal{U}(-0.1,0.1)$, $\epsilon \sim \mathcal{N}(0,5)$

#### 代码实现
```python
transform = A.ColorJitter(
    brightness=0.2, 
    contrast=0.2, 
    saturation=0.2, 
    hue=0.1, 
    p=0.8
)
```



## 3. 噪声处理与滤波

### 3.1 高斯滤波

#### 卷积核构建
$$
G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}
$$
归一化核：$\sum_{i,j} G(i,j) = 1$

#### Python实现
```python
def gaussian_blur(img, kernel_size=(5,5), sigma=1.5):
    return cv2.GaussianBlur(img, kernel_size, sigmaX=sigma)
```

### 3.2 中值滤波

#### 数学定义
$$
\text{output}(x,y) = \text{median} \{ I(x+i,y+j) | (i,j) \in W \}
$$
其中 $W$ 为邻域窗口

#### 代码实现
```python
def median_filter(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)
```



## 4. 特征增强

### 4.1 边缘检测

#### Canny算法流程
1. 高斯滤波平滑
2. Sobel算子计算梯度：
   $$
   G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad 
   G_y = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix} * I
   $$
3. 非极大值抑制
4. 双阈值检测

#### Python实现
```python
edges = cv2.Canny(img, threshold1=50, threshold2=150)
```

### 4.2 角点检测

#### Harris响应函数
$$
R = \det(M) - k \cdot \text{trace}(M)^2
$$
其中结构张量：
$$
M = \sum_{W} \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}
$$

#### 代码实现
```python
corners = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
```



## 5. 高级预处理

### 5.1 直方图均衡化

#### 累积分布函数（CDF）
$$
\text{CDF}(k) = \sum_{i=0}^k p(i), \quad p(i) = \frac{n_i}{N}
$$
转换函数：
$$
T(k) = \text{round}\left( \text{CDF}(k) \times (L-1) \right)
$$

#### Python实现
```python
equ = cv2.equalizeHist(gray_img)
```

### 5.2 形态学操作

#### 膨胀与腐蚀
$$
\text{膨胀}: A \oplus B = \{ z | (\hat{B})_z \cap A \neq \emptyset \}
$$
$$
\text{腐蚀}: A \ominus B = \{ z | B_z \subseteq A \}
$$

#### 代码实现
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dilated = cv2.dilate(img, kernel)
eroded = cv2.erode(img, kernel)
```

### 5.3 频域处理

#### 傅里叶变换
$$
\mathcal{F}(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y)e^{-j2\pi(ux/M + vy/N)}
$$

#### 低通滤波实现
```python
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 创建低通掩模
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30
mask[crow-r:crow+r, ccol-r:ccol+r] = 1

# 应用滤波
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
```



## 预处理Pipeline设计

```python
from torchvision import transforms

transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 使用示例
from PIL import Image
img = Image.open('image.jpg')
processed = transform_pipeline(img)
```



## 技术选型矩阵

| 技术          | 计算复杂度      | 内存消耗 | 适用场景              |
|-------------------|-------------------|------------|-------------------------|
| 双线性插值          | $O(NM)$        | 低          | 通用尺寸调整              |
| 高斯滤波            | $O(k^2NM)$     | 中          | 噪声去除，平滑处理        |
| Canny边缘检测      | $O(NM)$        | 高          | 特征提取，物体检测        |
| 直方图均衡化        | $O(NM)$        | 低          | 低对比度图像增强          |
| 频域滤波            | $O(NM \log NM)$| 高          | 周期性噪声去除            |



## 评估指标

1. PSNR（峰值信噪比）：
   $$
   \text{PSNR} = 10 \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)
   $$
   其中 $\text{MSE} = \frac{1}{MN}\sum_{i,j}(I(i,j)-K(i,j))^2$

2. SSIM（结构相似性）：
   $$
   \text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
   $$
