# 高斯混合模型
## 高斯混合模型（GMM）原理详解（含公式推导）

### 1. 核心思想与概率模型  
高斯混合模型（Gaussian Mixture Model, GMM）假设数据由 $K$ 个高斯分布混合生成，每个分布对应一个簇。其概率密度函数为：  
$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$  
其中：  
- $\pi_k$：第 $k$ 个高斯分布的混合权重（满足 $\sum_{k=1}^K \pi_k = 1$）  
- $\mu_k$：第 $k$ 个高斯分布的均值向量  
- $\Sigma_k$：第 $k$ 个高斯分布的协方差矩阵  

### 2. 隐变量与完全数据似然  
引入隐变量 $z = (z_1, ..., z_K)$，其中 $z_k \in \{0,1\}$ 表示样本属于第 $k$ 个高斯分布的概率：  
$$
p(z_k=1) = \pi_k \quad \text{且} \quad p(x \mid z_k=1) = \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$  
完全数据似然函数为：  
$$
p(x, z) = \prod_{k=1}^K \left[ \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k) \right]^{z_k}
$$  

### 3. EM算法推导  
通过期望最大化（Expectation-Maximization, EM）算法迭代优化参数：  

**E步（期望步）**：计算隐变量后验概率（责任值 $\gamma(z_{nk})$）：  
$$
\gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}
$$  

**M步（最大化步）**：更新参数以最大化对数似然：  
$$
\begin{aligned}
N_k &= \sum_{n=1}^N \gamma(z_{nk}) \\
\pi_k^{\text{new}} &= \frac{N_k}{N} \\
\mu_k^{\text{new}} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n \\
\Sigma_k^{\text{new}} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^T
\end{aligned}
$$  

### 4. 对数似然函数与收敛性  
对数似然函数为：  
$$
\ln p(X \mid \pi, \mu, \Sigma) = \sum_{n=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \right)
$$  
EM算法保证对数似然单调递增直至收敛到局部最优。

### 5. 协方差矩阵约束  
协方差矩阵可设定不同形式以控制模型复杂度：  
- **Full**：任意对称正定矩阵 $\Sigma_k \in \mathbb{R}^{D \times D}$  
- **Tied**：所有簇共享同一协方差矩阵 $\Sigma_k = \Sigma$  
- **Diag**：对角矩阵 $\Sigma_k = \text{diag}(\sigma_{k1}^2, ..., \sigma_{kD}^2)$  
- **Spherical**：标量方差 $\Sigma_k = \sigma_k^2 I$  

### 6. 模型选择与超参数调优  
- **分量数 $K$ 选择**：  
  - 信息准则：AIC（Akaike）或 BIC（Bayesian）  
  $$
  \text{AIC} = -2\ln p(X \mid \hat{\theta}) + 2P \\
  \text{BIC} = -2\ln p(X \mid \hat{\theta}) + P \ln N
  $$  
  （其中 $P$ 为参数总数，选择使准则最小的 $K$）  
- **初始化策略**：K-means初始化或随机多次重启。


## Python深度实践指南（扩展版）


### 1. 环境准备  
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import multivariate_normal
```

### 2. 基础实验：不同分布数据拟合  
```python
# 生成球形数据
X1, y1 = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)
# 生成非球形数据
X2, _ = make_blobs(n_samples=300, centers=1, cluster_std=2.0, random_state=0)
X2 = np.dot(X2, [[0.6, -0.6], [-0.4, 0.8]])  # 线性变换

# 合并数据并标准化
X = np.vstack([X1, X2])
X_scaled = StandardScaler().fit_transform(X)

# 训练GMM模型
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

# 可视化概率分布
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=probs.argmax(axis=1), cmap='viridis', alpha=0.6)
plt.title("GMM Clustering with Probability Assignment")
plt.colorbar()
plt.show()
```

### 3. 模型选择：BIC与AIC分析  
```python
# 测试不同K值的BIC和AIC
K_range = range(1, 11)
bic, aic = [], []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
    gmm.fit(X_scaled)
    bic.append(gmm.bic(X_scaled))
    aic.append(gmm.aic(X_scaled))

# 绘制曲线
plt.plot(K_range, bic, marker='o', label='BIC')
plt.plot(K_range, aic, marker='s', label='AIC')
plt.xticks(K_range)
plt.xlabel("Number of Components (K)")
plt.ylabel("Criterion Value")
plt.legend()
plt.title("Model Selection by Information Criteria")
plt.show()
```

### 4. 协方差类型对比实验  
```python
cov_types = ['spherical', 'diag', 'tied', 'full']
plt.figure(figsize=(15, 10))

for i, cov_type in enumerate(cov_types, 1):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=0)
    labels = gmm.fit_predict(X_scaled)
    
    # 绘制等高线
    x_min, x_max = X_scaled[:,0].min()-1, X_scaled[:,0].max()+1
    y_min, y_max = X_scaled[:,1].min()-1, X_scaled[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))
    Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.subplot(2, 2, i)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis', edgecolor='k')
    plt.title(f"Covariance: {cov_type}\nBIC: {gmm.bic(X_scaled):.1f}")
plt.tight_layout()
plt.show()
```

### 5. 高维数据与降维可视化  
```python
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

# 加载数据并标准化
wine = load_wine()
X = wine.data
X_scaled = StandardScaler().fit_transform(X)

# GMM聚类
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
labels = gmm.fit_predict(X_scaled)

# PCA降维
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# 可视化
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', edgecolor='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM Clustering on Wine Data (PCA Reduced)")
plt.show()
```

### 6. 概率软聚类与异常检测  
```python
# 计算样本属于各簇的概率
probs = gmm.predict_proba(X_scaled)

# 定义异常得分为最小簇概率
anomaly_scores = probs.min(axis=1)

# 标记前5%为异常
threshold = np.percentile(anomaly_scores, 95)
outliers = np.where(anomaly_scores >= threshold)[0]

# 可视化
plt.scatter(X_pca[:,0], X_pca[:,1], c='gray', alpha=0.3)
plt.scatter(X_pca[outliers,0], X_pca[outliers,1], c='red', label='Anomaly')
plt.title("Anomaly Detection by GMM Probability")
plt.legend()
plt.show()
```

### 7. 手写数字生成（生成式应用）  
```python
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data

# 训练GMM生成模型
gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=0)
gmm.fit(X)

# 从每个分量生成样本
samples = []
for _ in range(5):  # 每个簇生成5个样本
    for k in range(10):
        sample = gmm.sample()[0][0].reshape(8, 8)
        samples.append(sample)

# 可视化生成样本
plt.figure(figsize=(10, 5))
for i, img in enumerate(samples[:20]):
    plt.subplot(4, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle("Handwritten Digits Generated by GMM")
plt.show()
```


## 数学补充证明

### 1. EM算法收敛性证明  
- **E步**：计算 $Q(\theta, \theta^{\text{old}}) = \mathbb{E}_{Z \mid X, \theta^{\text{old}}}[\ln p(X,Z \mid \theta)]$  
- **M步**：更新 $\theta^{\text{new}} = \arg\max_\theta Q(\theta, \theta^{\text{old}})$  
- **定理**：每次迭代保证 $\ln p(X \mid \theta^{\text{new}}) \geq \ln p(X \mid \theta^{\text{old}})$  

### 2. 协方差矩阵更新公式推导  
最大化 $Q$ 函数对 $\Sigma_k$ 求导：  
$$
\frac{\partial Q}{\partial \Sigma_k} = -\frac{1}{2} \sum_{n=1}^N \gamma(z_{nk}) \left[ \Sigma_k^{-1} - \Sigma_k^{-1}(x_n-\mu_k)(x_n-\mu_k)^T \Sigma_k^{-1} \right] = 0
$$  
解得：  
$$
\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^T
$$


## 总结与扩展方向  
1. **理论扩展**：  
   - 贝叶斯GMM（Dirichlet过程混合模型）  
   - 非高斯混合模型（如Student's t混合）  

2. **计算优化**：  
   - 使用变分推断（Variational Inference）加速  
   - 小批量EM算法处理大规模数据  

3. **应用场景**：  
   - 语音识别中的音素分类  
   - 图像分割与背景建模  
