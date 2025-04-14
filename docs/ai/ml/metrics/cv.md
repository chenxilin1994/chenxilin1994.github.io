# 计算机视觉相关评价指标



## 一、图像分类指标

### 1. **Top-1 Accuracy**
**原理**：预测概率最高的类别是否正确。  
**公式**：
$$
\text{Top-1 Accuracy} = \frac{\text{正确预测数}}{\text{总样本数}}
$$
**Python实现**：
```python
from sklearn.metrics import accuracy_score
y_true = [2, 0, 2, 1]
y_pred = [2, 1, 2, 0]
print("Top-1 Accuracy:", accuracy_score(y_true, y_pred))  # 输出 0.5
```

### 2. **Top-5 Accuracy**
**原理**：真实类别是否在预测概率前5的类别中。  
**适用场景**：类别数较多（如ImageNet）。  
**Python实现**：
```python
import numpy as np

def top_k_accuracy(y_true, y_pred_probs, k=5):
    top_k = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = [y_true[i] in top_k[i] for i in range(len(y_true))]
    return np.mean(correct)

y_pred_probs = np.array([
    [0.1, 0.2, 0.7],  # 类别2概率最高
    [0.6, 0.3, 0.1],  # 类别0概率最高
    [0.2, 0.5, 0.3],  # 类别1概率最高
    [0.4, 0.1, 0.5]   # 类别2概率最高
])
y_true = [2, 0, 1, 2]
print("Top-2 Accuracy:", top_k_accuracy(y_true, y_pred_probs, k=2))  # 输出 0.75
```



## 二、目标检测指标

### 1. **交并比（IoU, Intersection over Union）**
**原理**：预测框与真实框的交集面积与并集面积的比值。  
**公式**：
$$
IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}}
$$
**Python实现**：
```python
def calculate_iou(boxA, boxB):
    # box格式: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

boxA = [10, 10, 50, 50]
boxB = [30, 30, 80, 80]
print("IoU:", calculate_iou(boxA, boxB))  # 输出约0.255
```

### 2. **平均精度均值（mAP, mean Average Precision）**
**原理**：在不同IoU阈值（如0.5:0.95）下计算各类别AP的平均值。  
**计算步骤**：
1. 对每个类别计算Precision-Recall曲线。
2. 对PR曲线积分（或插值）得到AP。
3. 对所有类别的AP取平均得到mAP。

**Python实现**（使用COCO API）：
```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载标注和预测结果
coco_gt = COCO('annotations.json')
coco_dt = coco_gt.loadRes('predictions.json')

# 评估
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()  # 输出mAP@0.5:0.95
```



## 三、图像分割指标

### 1. **Dice系数（Dice Coefficient）**
**原理**：预测区域与真实区域重叠度的衡量，值越接近1越好。  
**公式**：
$$
\text{Dice} = \frac{2 \times |Y_{\text{pred}} \cap Y_{\text{true}}|}{|Y_{\text{pred}}| + |Y_{\text{true}}|}
$$
**Python实现**：
```python
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

y_true = np.array([[1, 0], [0, 1]])
y_pred = np.array([[1, 1], [0, 1]])
print("Dice:", dice_coefficient(y_true, y_pred))  # 输出 0.8
```

### 2. **像素精度（Pixel Accuracy）**
**原理**：正确分类的像素占总像素的比例。  
**公式**：
$$
\text{Pixel Accuracy} = \frac{\sum \text{正确像素数}}{\sum \text{总像素数}}
$$
**Python实现**：
```python
def pixel_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = y_true.size
    return correct / total

y_true = np.array([[0, 1], [2, 3]])
y_pred = np.array([[0, 1], [2, 2]])
print("Pixel Accuracy:", pixel_accuracy(y_true, y_pred))  # 输出 0.75
```



## 四、图像生成与重建指标

### 1. **峰值信噪比（PSNR, Peak Signal-to-Noise Ratio）**
**原理**：衡量图像重建质量，值越大表示失真越小。  
**公式**：
$$
\text{PSNR} = 10 \times \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right)
$$
- MAX为像素最大值（如255）。  

**Python实现**：
```python
import numpy as np

def psnr(y_true, y_pred, max_pixel=255.0):
    mse = np.mean((y_true - y_pred) ** 2)
    return 10 * np.log10(max_pixel**2 / mse)

y_true = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
y_pred = y_true + np.random.normal(0, 10, (100, 100))
print("PSNR:", psnr(y_true, y_pred))  # 值越大越好
```

### 2. **结构相似性指数（SSIM, Structural Similarity Index）**
**原理**：从亮度、对比度、结构三方面评估图像相似性，范围[-1, 1]。  
**Python实现**：
```python
from skimage.metrics import structural_similarity as ssim

ssim_score = ssim(y_true, y_pred, data_range=255, multichannel=True)
print("SSIM:", ssim_score)  # 越接近1越好
```



## 五、关键点检测指标

### **PCK（Percentage of Correct Keypoints）**
**原理**：关键点预测位置与真实位置的误差小于阈值（如头部尺寸的50%）的比例。  
**公式**：
$$
\text{PCK} = \frac{\sum I(\text{误差} \leq \alpha \times \text{参考长度})}{N}
$$
**Python实现**：
```python
def pck(y_true, y_pred, alpha=0.5, ref_lengths=None):
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    thresholds = alpha * ref_lengths  # ref_lengths如头部尺寸
    correct = np.sum(distances <= thresholds)
    return correct / len(y_true)

y_true = np.array([[10, 20], [30, 40]])
y_pred = np.array([[12, 22], [29, 38]])
ref_lengths = [20, 20]  # 每个关键点的参考长度
print("PCK@0.5:", pck(y_true, y_pred, alpha=0.5, ref_lengths=ref_lengths))
```



## 六、生成对抗网络（GAN）评估指标

### 1. **Inception Score (IS)**
**原理**：评估生成图像的多样性和质量，基于Inception模型的预测分布。  
**公式**：
$$
\text{IS} = \exp \left( \mathbb{E}_{x \sim p_g} [KL(p(y|x) \parallel p(y))] \right)
$$
**Python实现**（需预训练Inception-v3模型）：
```python
import torch
from torchvision.models import inception_v3
from torch.nn.functional import softmax

model = inception_v3(pretrained=True, transform_input=False).eval()
def inception_score(images, batch_size=32):
    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            pred = model(batch)
        preds.append(softmax(pred, dim=1))
    preds = torch.cat(preds, dim=0)
    p_yx = preds.mean(dim=0)
    kl = preds * (torch.log(preds) - preds * torch.log(p_yx)
    kl = kl.sum(dim=1).mean()
    return torch.exp(kl).item()
```

### 2. **Frèchet Inception Distance (FID)**
**原理**：计算真实图像与生成图像在特征空间的分布距离，值越小越好。  
**公式**：
$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$
**Python实现**：
```python
from scipy.linalg import sqrtm
from torchvision.models import inception_v3

def calculate_fid(real_features, gen_features):
    mu_r, sigma_r = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = gen_features.mean(0), np.cov(gen_features, rowvar=False)
    cov_mean = sqrtm(sigma_r.dot(sigma_g))
    fid = np.sum((mu_r - mu_g)**2) + np.trace(sigma_r + sigma_g - 2*cov_mean)
    return fid
```



## 七、实时性指标

### **FPS（Frames Per Second）**
**原理**：模型每秒处理的帧数，衡量实时性。  
**Python实现**：
```python
import time

def calculate_fps(model, input_tensor, warmup=10, runs=100):
    # 预热
    for _ in range(warmup):
        _ = model(input_tensor)
    # 计时
    start = time.time()
    for _ in range(runs):
        _ = model(input_tensor)
    elapsed = time.time() - start
    return runs / elapsed

# 示例：输入张量尺寸 (1, 3, 224, 224)
input_tensor = torch.randn(1, 3, 224, 224)
fps = calculate_fps(model, input_tensor)
print("FPS:", fps)
```



## 八、指标对比与选择建议
| 指标               | 适用任务               | 优点                          | 缺点                      |
|--------------------|-----------------------|-----------------------------|--------------------------|
| **mAP**            | 目标检测              | 综合IoU和分类准确性           | 计算复杂度高               |
| **Dice系数**       | 图像分割              | 对不平衡数据敏感               | 仅关注重叠区域             |
| **PSNR/SSIM**      | 图像重建              | 计算简单，物理意义明确         | 与人类感知不完全一致        |
| **FID**            | GAN评估               | 反映生成图像分布真实性         | 依赖预训练模型             |
| **FPS**            | 实时性评估            | 直观反映推理速度               | 受硬件环境影响大           |



## 九、总结
- **目标检测**：优先使用mAP和IoU，结合FPS评估实时性。  
- **图像分割**：Dice系数和像素精度互补，联合评估分割效果。  
- **图像生成**：FID和IS评估生成质量，SSIM/PSNR衡量重建效果。  
- **关键点检测**：PCK指标直观反映定位精度。  
- **实际应用**：根据任务需求选择核心指标，并结合可视化分析模型行为。