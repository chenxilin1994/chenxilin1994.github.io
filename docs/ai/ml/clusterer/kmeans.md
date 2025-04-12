# K均值聚类
## K均值聚类算法原理详解（含公式推导）

### 1. 核心数学目标
K均值的目标是最小化**簇内平方误差（SSE）**，定义如下：

$$
SSE = \sum_{i=1}^K \sum_{x \in C_i} \|x - \mu_i\|^2
$$

其中：
- $K$：簇的数量
- $C_i$：第 $i$ 个簇的数据点集合
- $\mu_i$：第 $i$ 个簇的中心（均值向量）
- $\|x - \mu_i\|$：数据点 $x$ 到簇中心 $\mu_i$ 的欧氏距离

### 2. 算法步骤的数学描述
**步骤1：初始化簇中心**  
随机选择 $K$ 个初始簇中心 $\{\mu_1^{(0)}, \mu_2^{(0)}, ..., \mu_K^{(0)}\}$，或使用K-means++优化选择。

**步骤2：分配数据点（Expectation Step）**  
对每个数据点 $x_p$（$p=1,2,...,N$），计算其到所有簇中心的距离，并分配到最近簇：  
$$
C_i^{(t)} = \left\{ x_p : \|x_p - \mu_i^{(t)}\|^2 \leq \|x_p - \mu_j^{(t)}\|^2 \ \forall j \right\}
$$

**步骤3：更新簇中心（Maximization Step）**  
重新计算每个簇的均值作为新簇中心：  
$$
\mu_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{x \in C_i^{(t)}} x
$$

**步骤4：收敛判定**  
重复步骤2-3，直到满足终止条件：  
$$
\max_{i} \|\mu_i^{(t+1)} - \mu_i^{(t)}\| < \epsilon \quad \text{或达到最大迭代次数}
$$

### 3. 关键公式推导
**欧氏距离计算**：  
$$
d(x, \mu_i) = \sqrt{\sum_{j=1}^D (x_j - \mu_{ij})^2}
$$

**簇中心更新公式证明**：  
SSE对 $\mu_i$ 求导并令导数为零：  
$$
\frac{\partial SSE}{\partial \mu_i} = -2 \sum_{x \in C_i} (x - \mu_i) = 0 \quad \Rightarrow \quad \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
$$

### 4. K-means++初始化原理
优化初始中心选择，降低局部最优风险：  
1. 随机选择第一个中心 $\mu_1$。  
2. 选择第 $m+1$ 个中心时，以概率 $\frac{D(x)^2}{\sum_{x} D(x)^2}$ 选择数据点 $x$，其中 $D(x)$ 是 $x$ 到已选中心的最近距离。  
3. 重复直至选出 $K$ 个中心。

---

## Python深度实践指南（扩展版）

### 1. 复杂数据分布场景实验
**生成非球形数据并观察K均值局限性**：
```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
X_scaled = StandardScaler().fit_transform(X)

# 尝试K均值聚类
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_scaled)
labels = kmeans.labels_

# 可视化（明显无法正确分割）
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.title("K-Means on Non-Convex Data")
plt.show()

# 对比DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(X_scaled)
plt.scatter(X[:,0], X[:,1], c=db.labels_, cmap='viridis')
plt.title("DBSCAN on Non-Convex Data")
plt.show()
```

### 2. 高维数据聚类与降维
**PCA降维后可视化**（以手写数字数据集为例）：
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = digits.data
y = digits.target

# 标准化与降维
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# K均值聚类
kmeans = KMeans(n_clusters=10, random_state=0).fit(X_scaled)
labels = kmeans.labels_

# 可视化降维后的聚类结果
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.6)
plt.title("K-Means Clustering on PCA-Reduced Digits Data")
plt.show()
```

### 3. 超参数调优实验
**分析 `n_init` 和 `max_iter` 对结果的影响**：
```python
n_inits = [1, 5, 10, 20]
max_iters = [10, 50, 100, 200]

results = []
for n_init in n_inits:
    for max_iter in max_iters:
        kmeans = KMeans(n_clusters=4, n_init=n_init, max_iter=max_iter, random_state=0)
        kmeans.fit(X_scaled)
        results.append({
            'n_init': n_init,
            'max_iter': max_iter,
            'SSE': kmeans.inertia_,
            'Silhouette': silhouette_score(X_scaled, kmeans.labels_)
        })

# 转换为DataFrame分析
import pandas as pd
df = pd.DataFrame(results)
print(df.pivot_table(index='n_init', columns='max_iter', values=['SSE', 'Silhouette']))
```

### 4. 完整项目实战：客户分群
**数据集**：Mall Customer Segmentation Data  
**步骤**：  
1. 数据加载与探索：
```python
import pandas as pd
df = pd.read_csv("Mall_Customers.csv")
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
```

2. 特征工程与标准化：
```python
from sklearn.preprocessing import PowerTransformer
X_scaled = PowerTransformer().fit_transform(X)  # 处理偏态分布
```

3. 确定最佳K值（轮廓系数与Gap Statistic结合）：
```python
from gap_statistic import OptimalK  # 需安装gap-stat包

optimalK = OptimalK(n_jobs=-1)
n_clusters = optimalK(X_scaled, cluster_array=range(1, 11))
print(f"Optimal K by Gap Statistic: {n_clusters}")
```

4. 训练最终模型与业务解释：
```python
kmeans = KMeans(n_clusters=5, n_init=20, random_state=42).fit(X_scaled)
df['Cluster'] = kmeans.labels_

# 分析聚类特征
cluster_profile = df.groupby('Cluster').mean()
print(cluster_profile)

# 3D可视化
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], 
           c=df['Cluster'], cmap='viridis', depthshade=False)
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Spending Score')
plt.title("3D Customer Segmentation")
plt.show()
```

### 5. 高级技巧：半监督聚类
```python
# 假设已知部分标签（10%的数据）
partial_labels = np.full(len(X), -1)  # -1表示未知
known_idx = np.random.choice(len(X), size=20, replace=False)
partial_labels[known_idx] = y[known_idx]

# 使用约束K均值
from sklearn.semi_supervised import KMeans
constrained_kmeans = KMeans(n_clusters=10).fit(X_scaled, partial_labels)
```

---

## 数学补充证明

### K均值收敛性证明
1. **单调性**：每次迭代SSE必定减小  
   - 分配步骤：固定 $\mu_i$，优化 $C_i$ → SSE非增  
   - 更新步骤：固定 $C_i$，优化 $\mu_i$ → SSE非增  
2. **有下界**：SSE ≥ 0  
3. **收敛定理**：单调有界序列必收敛  

### K-means++的理论保证
- 期望近似比：$O(\log K)$ 近似最优初始化  
- 显著降低坏初始化的概率  

---

## 总结与扩展方向
1. **理论深化**：  
   - 推导Mahalanobis距离下的K均值变种  
   - 研究核K均值（Kernel K-Means）的数学形式  

2. **工程优化**：  
   - 实现Mini-Batch K-Means处理超大规模数据  
   - 使用Elkan算法加速距离计算  

3. **领域应用**：  
   - 图像压缩：将颜色空间聚类为K种代表色  
   - 文档聚类：TF-IDF向量化后应用K均值  
