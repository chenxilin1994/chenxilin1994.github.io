
# 主成分分析
## 主成分分析（PCA）算法原理详解（含公式推导）

### 1. 核心思想与数学目标  
主成分分析（Principal Component Analysis, PCA）是一种线性降维方法，旨在通过正交变换将原始特征映射到**低维空间**，使得新特征：  
1. **最大方差**：各主成分方向保持数据最大方差（信息量最大）。  
2. **线性无关**：主成分之间彼此正交（协方差为零）。  

**数学目标**：找到投影矩阵 $W \in \mathbb{R}^{D \times K}$（$K < D$），使得投影后数据 $Y = XW$ 的方差最大化，且 $W$ 的列向量正交。



### 2. 算法步骤的数学推导  
**输入**：中心化数据矩阵 $X \in \mathbb{R}^{N \times D}$（每行一个样本，已中心化）  
**输出**：主成分矩阵 $W$，降维数据 $Y = XW$  

**步骤1：计算协方差矩阵**  
$$
C = \frac{1}{N-1} X^T X \quad \in \mathbb{R}^{D \times D}
$$  

**步骤2：特征值分解**  
求解协方差矩阵的特征方程：  
$$
C v = \lambda v
$$  
得到特征值 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D$ 及对应的特征向量 $v_1, v_2, ..., v_D$。  

**步骤3：选择主成分**  
按特征值从大到小排序，选择前 $K$ 个特征向量组成投影矩阵：  
$$
W = [v_1 \quad v_2 \quad \cdots \quad v_K] \quad \in \mathbb{R}^{D \times K}
$$  

**步骤4：数据投影**  
降维后的数据为：  
$$
Y = XW \quad \in \mathbb{R}^{N \times K}
$$  



### 3. 关键公式与理论  
**方差最大化证明**：  
投影后第 $k$ 个主成分的方差为：  
$$
\text{Var}(Y_k) = w_k^T C w_k
$$  
在约束 $w_k^T w_k = 1$ 和 $w_k \perp \{w_1, ..., w_{k-1}\}$ 下，通过拉格朗日乘数法可得：  
$$
C w_k = \lambda_k w_k
$$  
即 $w_k$ 为协方差矩阵 $C$ 的特征向量，方差 $\lambda_k$ 为对应特征值。  

**累积方差解释率**：  
前 $K$ 个主成分保留的方差比例为：  
$$
\text{Explained Variance Ratio} = \frac{\sum_{i=1}^K \lambda_i}{\sum_{j=1}^D \lambda_j}
$$  



## Python深度实践指南（扩展版）



### 1. 基础实现：手动计算PCA  
```python
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
X = np.dot(np.random.randn(100, 2), [[2, 2], [0.5, 0.5]]).T  # 线性相关数据
X = X - X.mean(axis=0)  # 中心化

# 手动PCA
cov_matrix = np.cov(X, rowvar=False)  # 计算协方差矩阵
eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)  # 特征值分解

# 排序特征值
sorted_idx = np.argsort(eigen_vals)[::-1]
eigen_vals = eigen_vals[sorted_idx]
eigen_vecs = eigen_vecs[:, sorted_idx]

# 选择前K个主成分
K = 1
W = eigen_vecs[:, :K]
X_pca = X @ W

# 可视化
plt.scatter(X[:,0], X[:,1], alpha=0.6, label='Original Data')
plt.quiver(0, 0, W[0], W[1], angles='xy', scale_units='xy', scale=1, color='r', label='PC1')
plt.axis('equal')
plt.legend()
plt.title("Manual PCA Projection")
plt.show()
```

### 2. 方差解释率分析  
```python
from sklearn.decomposition import PCA

# 计算累积方差解释率
pca = PCA().fit(X)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative = np.cumsum(explained_variance_ratio)

# 绘制碎石图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, alpha=0.5, align='center', label='Individual')
plt.step(range(len(cumulative)), cumulative, where='mid', label='Cumulative')
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.title("Scree Plot")

# 选择保留95%方差的成分数
K = np.argmax(cumulative >= 0.95) + 1
plt.subplot(1, 2, 2)
plt.plot(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), '|', markersize=100)
plt.title(f"1D Projection (K={K})")
plt.tight_layout()
plt.show()
```

### 3. 高维数据可视化（以手写数字为例）  
```python
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

# PCA降维到2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Digits Dataset")
plt.show()
```

### 4. 图像压缩与重建  
```python
from sklearn.datasets import fetch_olivetti_faces

# 加载人脸数据
faces = fetch_olivetti_faces()
X = faces.data

# 使用PCA重建图像
pca = PCA(n_components=100).fit(X)
components = pca.components_

# 随机选择样本重建
sample_idx = np.random.randint(0, X.shape[0])
sample = X[sample_idx]
sample_proj = pca.transform(sample.reshape(1, -1))
sample_recon = pca.inverse_transform(sample_proj)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(sample.reshape(64, 64), cmap='gray')
ax[0].set_title("Original")
ax[1].imshow(sample_recon.reshape(64, 64), cmap='gray')
ax[1].set_title(f"Reconstructed (K=100)")
plt.show()
```

### 5. 核PCA处理非线性数据  
```python
from sklearn.decomposition import KernelPCA

# 生成非线性数据（同心圆）
X, y = make_circles(n_samples=500, noise=0.05, factor=0.3, random_state=0)
X = StandardScaler().fit_transform(X)

# 对比线性PCA与核PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis')
ax[0].set_title("Linear PCA")
ax[1].scatter(X_kpca[:,0], X_kpca[:,1], c=y, cmap='viridis')
ax[1].set_title("Kernel PCA (RBF)")
plt.show()
```



## 数学补充证明

### 1. 协方差矩阵与最大方差等价性  
最大化投影方差等价于求解协方差矩阵特征值分解：  
$$
\max_w \text{Var}(Y) = \max_w w^T C w \quad \text{s.t.} \quad w^T w = 1
$$  
拉格朗日函数为：  
$$
\mathcal{L} = w^T C w - \lambda (w^T w - 1)
$$  
求导得 $C w = \lambda w$，即最优解为最大特征值对应的特征向量。

### 2. 数据重建误差最小化  
PCA等价于最小化重建误差：  
$$
\min_W \|X - X W W^T\|_F^2 \quad \text{s.t.} \quad W^T W = I
$$  
其解与协方差矩阵特征分解一致。



## 总结与扩展方向  
1. **理论扩展**：  
   - 概率PCA（Probabilistic PCA）的贝叶斯推导  
   - 稀疏PCA（Sparse PCA）的 $L_1$ 正则化优化  

2. **工程优化**：  
   - 增量PCA（Incremental PCA）处理流式数据  
   - 随机PCA（Randomized PCA）加速大规模计算  

3. **应用场景**：  
   - 金融数据去噪与因子分析  
   - 基因表达数据的维度约简  
