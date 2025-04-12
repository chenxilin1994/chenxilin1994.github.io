# 层次聚类

## 层次聚类算法原理详解（含公式推导）

---

### 1. 核心思想与分类  
层次聚类（Hierarchical Clustering）通过构建 **树状结构（树状图）** 表示数据的层次关系，分为两类：  
1. **凝聚法（Agglomerative）**：自底向上，初始每个样本为单独簇，逐步合并最近簇。  
2. **分裂法（Divisive）**：自顶向下，初始所有样本为一个簇，逐步分裂为更小簇。  
*（实践中凝聚法更常用，下文重点讨论）*

### 2. 算法步骤的数学描述  
**输入**：数据集 $X = \{x_1, x_2, ..., x_N\}$，距离度量 $d(\cdot, \cdot)$，连接准则（Linkage Criterion）  
**输出**：树状图（Dendrogram）与指定簇数目的聚类结果  

**步骤1：初始化**  
- 每个样本初始化为一个簇：$C = \{\{x_1\}, \{x_2\}, ..., \{x_N\}\}$  

**步骤2：迭代合并簇**  
重复以下步骤直至所有样本合并为一个簇：  
1. 计算所有簇间距离矩阵 $D$，其中 $D_{ij} = d(C_i, C_j)$  
2. 找到最小距离的簇对 $(C_p, C_q) = \arg\min_{i,j} D_{ij}$  
3. 合并 $C_p$ 和 $C_q$ 为新簇 $C_{new} = C_p \cup C_q$  
4. 更新簇集合 $C = (C \setminus \{C_p, C_q\}) \cup \{C_{new}\}$  

### 3. 关键公式与连接准则  
不同连接准则定义簇间距离 $d(C_i, C_j)$：  

| 准则名称       | 数学定义                                                                 | 特点                          |  
|----------------|--------------------------------------------------------------------------|-------------------------------|  
| **单连接**     | $d_{single}(C_i, C_j) = \min_{a \in C_i, b \in C_j} d(a, b)$         | 擅长发现细长簇，对噪声敏感    |  
| **全连接**     | $d_{complete}(C_i, C_j) = \max_{a \in C_i, b \in C_j} d(a, b)$       | 生成紧凑簇，对噪声鲁棒        |  
| **平均连接**   | $d_{average}(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{a \in C_i} \sum_{b \in C_j} d(a, b)$ | 平衡单连接与全连接            |  
| **Ward方法**   | $d_{ward}(C_i, C_j) = \frac{|C_i||C_j|}{|C_i| + |C_j|} \|\mu_i - \mu_j\|^2$ | 最小化合并后的簇内方差增加量  |  

**Ward方法推导**：  
合并簇 $C_i$ 和 $C_j$ 后的总方差增加量为：  
$$
\Delta V = \frac{|C_i||C_j|}{|C_i| + |C_j|} \|\mu_i - \mu_j\|^2
$$  
其中 $\mu_i, \mu_j$ 为原簇中心，证明如下：  
- 合并前总方差：$V_{before} = \sum_{x \in C_i} \|x - \mu_i\|^2 + \sum_{x \in C_j} \|x - \mu_j\|^2$  
- 合并后方差：$V_{after} = \sum_{x \in C_i \cup C_j} \|x - \mu_{new}\|^2$  
- 可证 $V_{after} = V_{before} + \Delta V$，故Ward方法选择最小化 $\Delta V$。

### 4. 树状图与距离阈值  
- **树状图（Dendrogram）**：可视化合并过程，纵轴为合并时的距离。  
- **确定簇数**：通过水平切割树状图选择簇数 $K$，切割高度 $h$ 应反映显著的距离跳跃。

### 5. 优缺点分析  
- **优点**：  
  - 无需预设簇数，通过树状图灵活选择。  
  - 能发现任意形状的簇（依赖连接准则）。  
- **缺点**：  
  - 计算复杂度 $O(N^3)$（朴素实现），适合小规模数据（$N < 10^4$）。  
  - 内存消耗高（需存储距离矩阵 $O(N^2)$）。  


## Python实践

---

### 1. 环境准备与数据生成  
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
```

### 2. 不同连接准则对比实验  
```python
# 生成模拟数据（非凸分布）
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
X_scaled = StandardScaler().fit_transform(X)

# 定义不同连接准则
linkage_methods = ['single', 'complete', 'average', 'ward']

plt.figure(figsize=(15, 10))
for i, method in enumerate(linkage_methods, 1):
    # 训练模型
    model = AgglomerativeClustering(n_clusters=2, linkage=method)
    labels = model.fit_predict(X_scaled)
    
    # 可视化
    plt.subplot(2, 2, i)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', edgecolor='k')
    plt.title(f"Linkage: {method.upper()}\nSilhouette: {silhouette_score(X_scaled, labels):.2f}")
plt.tight_layout()
plt.show()
```

### 3. 树状图分析与最佳簇数选择  
```python
# 计算全连接矩阵
Z = linkage(X_scaled, method='ward')

# 绘制树状图
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='level', p=5)
plt.axhline(y=3.5, color='r', linestyle='--')  # 假设选择切割高度3.5
plt.xlabel("Sample Index")
plt.ylabel("Merge Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# 根据树状图选择K=2
model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = model.fit_predict(X_scaled)
```

### 4. 高维数据处理与降维可视化  
```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 加载数据并标准化
iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

# 层次聚类
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X_scaled)

# PCA降维可视化
X_pca = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', edgecolor='k')
plt.title("Agglomerative Clustering on Iris Data (PCA Reduced)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

### 5. 超参数调优与性能分析  
```python
# 分析不同簇数和连接准则的轮廓系数
k_range = range(2, 6)
methods = ['ward', 'average', 'complete', 'single']

results = []
for k in k_range:
    for method in methods:
        model = AgglomerativeClustering(n_clusters=k, linkage=method)
        labels = model.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            score = -1  # 无效聚类
        else:
            score = silhouette_score(X_scaled, labels)
        results.append({'K': k, 'Method': method, 'Silhouette': score})

# 转换为DataFrame并展示热力图
import pandas as pd
import seaborn as sns

df = pd.DataFrame(results)
pivot_df = df.pivot_table(index='Method', columns='K', values='Silhouette')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu')
plt.title("Silhouette Score Analysis")
plt.show()
```

### 6. 真实案例：基因表达数据聚类  
```python
import pandas as pd
from sklearn.preprocessing import normalize

# 加载基因表达数据（行：样本，列：基因）
data = pd.read_csv("gene_expression.csv", index_col=0)
X = data.values
X_normalized = normalize(X, axis=0)  # 列归一化

# 层次聚类（样本聚类）
Z = linkage(X_normalized.T, method='average')  # 对基因聚类

# 绘制基因聚类树状图
plt.figure(figsize=(15, 8))
dendrogram(Z, labels=data.columns, leaf_rotation=90)
plt.title("Gene Expression Clustering")
plt.show()
```

### 7. 高级优化：加速层次聚类  
```python
# 使用FastCluster库加速（C++后端）
from fastcluster import linkage as fast_linkage
import time

# 生成大规模数据
X_large, _ = make_blobs(n_samples=5000, centers=5, random_state=0)

# 对比计算时间
start = time.time()
Z_fast = fast_linkage(X_large, method='ward')
print(f"FastCluster time: {time.time() - start:.2f}s")

start = time.time()
Z_sklearn = linkage(X_large, method='ward')
print(f"SciPy time: {time.time() - start:.2f}s")
```

---

## 数学补充证明

### 1. Ward方法的方差增量推导  
合并两个簇 $C_i$ 和 $C_j$，新簇中心为：  
$$
\mu_{new} = \frac{|C_i|\mu_i + |C_j|\mu_j}{|C_i| + |C_j|}
$$  
合并后的方差增量为：  
$$
\Delta V = \sum_{x \in C_i \cup C_j} \|x - \mu_{new}\|^2 - \left( \sum_{x \in C_i} \|x - \mu_i\|^2 + \sum_{x \in C_j} \|x - \mu_j\|^2 \right)
$$  
展开化简可得：  
$$
\Delta V = \frac{|C_i||C_j|}{|C_i| + |C_j|} \|\mu_i - \mu_j\|^2
$$  

### 2. 单连接与全连接的性质  
- **单连接**：满足**最短路径**特性，可能产生链式效应。  
- **全连接**：满足**直径**特性，对噪声更鲁棒。  

---

## 总结与扩展方向  
1. **理论扩展**：  
   - 研究不同距离度量（如DTW距离）对时间序列聚类的影响。  
   - 推导BIRCH算法（层次聚类的内存优化版本）的数学形式。  

2. **工程优化**：  
   - 结合KD-Tree加速最近邻搜索。  
   - 使用GPU加速大规模层次聚类（如CuML库）。  

3. **领域应用**：  
   - 社交网络分析：通过用户行为数据发现社区结构。  
   - 生物信息学：蛋白质相互作用网络聚类。  
