
# 聚类评价指标

## 一、核心概念
聚类评价指标分为三类：
- **内部指标（Internal）**：无需真实标签，基于数据分布和聚类结果评估。
- **外部指标（External）**：需要真实标签，衡量聚类结果与标签的一致性。
- **相对指标（Relative）**：比较不同聚类参数或算法的性能。



## 二、内部指标（无需真实标签）

### 1. 轮廓系数（Silhouette Coefficient）
**原理**：衡量单个样本聚类紧密度和分离度的综合指标，取值范围为 $[-1, 1]$，值越大越好。  
**公式**：
$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$
- $a(i)$：样本 $i$ 到同簇其他样本的平均距离（簇内不相似度）。  
- $b(i)$：样本 $i$ 到最近其他簇所有样本的平均距离（簇间不相似度）。  

**全局轮廓系数**：所有样本 $s(i)$ 的均值。

**Python实现**：
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
print("Silhouette Score:", score)
```



### 2. Calinski-Harabasz指数（方差比准则）
**原理**：簇间离散度与簇内离散度的比值，值越大表示聚类效果越好。  
**公式**：
$$
CH = \frac{\text{Tr}(B_k) / (k-1)}{\text{Tr}(W_k) / (n - k)}
$$
- $B_k$：簇间协方差矩阵。  
- $W_k$：簇内协方差矩阵。  
- $k$：簇数，$n$：样本数。

**Python实现**：
```python
from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(X, labels)
print("Calinski-Harabasz Score:", score)
```



### 3. Davies-Bouldin指数（DB指数）
**原理**：簇间距离与簇内直径的比值，值越小越好。  
**公式**：
$$
DB = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left( \frac{\text{diam}(C_i) + \text{diam}(C_j)}{d(C_i, C_j)} \right)
$$
- $\text{diam}(C_i)$：簇 $C_i$ 内样本的最大距离。  
- $d(C_i, C_j)$：簇 $C_i$ 和 $C_j$ 中心点的距离。

**Python实现**：
```python
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(X, labels)
print("Davies-Bouldin Score:", score)
```



### 4. Dunn指数
**原理**：最小簇间距离与最大簇内直径的比值，值越大越好。  
**公式**：
$$
Dunn = \frac{\min_{i<j} d(C_i, C_j)}{\max_{1 \leq l \leq k} \text{diam}(C_l)}
$$

**Python实现**（需手动计算）：
```python
from sklearn.metrics import pairwise_distances
import numpy as np

def dunn_index(X, labels):
    clusters = np.unique(labels)
    intra_dists = [np.max(pairwise_distances(X[labels == c])) for c in clusters]
    max_intra = max(intra_dists)
    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            centroid_i = np.mean(X[labels == clusters[i]], axis=0)
            centroid_j = np.mean(X[labels == clusters[j]], axis=0)
            inter_dists.append(np.linalg.norm(centroid_i - centroid_j))
    min_inter = min(inter_dists)
    return min_inter / max_intra

print("Dunn Index:", dunn_index(X, labels))
```



### 5. 同质性（Homogeneity）和完整性（Completeness）
**原理**（需真实标签）：
- **同质性**：每个簇只包含单一类别样本的程度。  
- **完整性**：同一类别的样本被分配到同一簇的程度。  
**公式**：
$$
\text{Homogeneity} = 1 - \frac{H(C|K)}{H(C)}, \quad \text{Completeness} = 1 - \frac{H(K|C)}{H(K)}
$$
- $H(C|K)$：给定聚类结果的类别条件熵。  
- $H(K|C)$：给定类别标签的聚类条件熵。  

**Python实现**：
```python
from sklearn.metrics import homogeneity_completeness_v_measure
h, c, v = homogeneity_completeness_v_measure(true_labels, pred_labels)
print("Homogeneity:", h, "\nCompleteness:", c)
```



## 三、外部指标（需真实标签）

### 1. 调整兰德指数（Adjusted Rand Index, ARI）
**原理**：衡量聚类结果与真实标签的相似度，取值范围为 $[-1, 1]$，值越大越好。  
**公式**：
$$
ARI = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}
$$
- RI（Rand Index）：正确决策的比例（同簇同类别或不同簇不同类别）。

**Python实现**：
```python
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(true_labels, pred_labels)
print("Adjusted Rand Index:", ari)
```



### 2. 归一化互信息（Normalized Mutual Information, NMI）
**原理**：衡量聚类结果与真实标签的信息共享程度，值越大越好。  
**公式**：
$$
NMI = \frac{2 \cdot I(true\_labels; pred\_labels)}{H(true\_labels) + H(pred\_labels)}
$$
- $I$：互信息，$H$：熵。

**Python实现**：
```python
from sklearn.metrics import normalized_mutual_info_score
nmi = normalized_mutual_info_score(true_labels, pred_labels)
print("NMI:", nmi)
```



### 3. Fowlkes-Mallows指数（FMI）
**原理**：基于成对样本的精确率和召回率的几何平均，值越大越好。  
**公式**：
$$
FMI = \sqrt{\frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}}
$$

**Python实现**：
```python
from sklearn.metrics import fowlkes_mallows_score
fmi = fowlkes_mallows_score(true_labels, pred_labels)
print("Fowlkes-Mallows Score:", fmi)
```



### 4. 纯度（Purity）
**原理**：每个簇中占比最高的真实类别的样本比例。  
**公式**：
$$
\text{Purity} = \frac{1}{n} \sum_{k=1}^K \max_j |C_k \cap L_j|
$$
- $C_k$：第 $k$ 个簇，$L_j$：第 $j$ 个真实类别。

**Python实现**：
```python
from sklearn.metrics import contingency_matrix

def purity_score(true_labels, pred_labels):
    contingency = contingency_matrix(true_labels, pred_labels)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

print("Purity:", purity_score(true_labels, pred_labels))
```



## 四、相对指标（比较不同聚类结果）

### 1. 轮廓分析（Silhouette Analysis）
**原理**：通过比较不同簇数对应的轮廓系数，选择最优簇数。  
**Python实现**：
```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(range(2, 10), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
```



### 2. 肘部法（Elbow Method）
**原理**：根据簇内误差平方和（SSE）随簇数变化的拐点选择最优簇数。  
**Python实现**：
```python
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k).fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 10), sse)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()
```



## 五、其他指标

### 1. 熵（Entropy）
**原理**：衡量每个簇中类别分布的混乱程度，值越小越好。  
**公式**：
$$
E = \sum_{k=1}^K \frac{n_k}{n} \left( -\sum_{j=1}^J \frac{n_{kj}}{n_k} \log \frac{n_{kj}}{n_k} \right)
$$

**Python实现**：
```python
from scipy.stats import entropy

def clustering_entropy(true_labels, pred_labels):
    contingency = contingency_matrix(true_labels, pred_labels)
    cluster_entropies = [entropy(cluster) for cluster in contingency.T]
    return np.average(cluster_entropies, weights=np.sum(contingency, axis=0))

print("Entropy:", clustering_entropy(true_labels, pred_labels))
```



### 2. Jaccard指数
**原理**：同二分类中的Jaccard相似系数，衡量聚类结果与真实标签的重叠度。  
**Python实现**：
```python
from sklearn.metrics import jaccard_score
# 需将标签转换为二进制形式（示例略）
```



## 六、指标对比与选择建议
| 指标                | 类型     | 是否需要标签 | 适用场景                         |
|---------------------|----------|--------------|----------------------------------|
| **轮廓系数**        | 内部     | 否           | 评估单样本聚类质量               |
| **Calinski-Harabasz** | 内部   | 否           | 高维数据，快速评估簇间分离度     |
| **Davies-Bouldin**  | 内部     | 否           | 强调簇内紧密度和簇间分离度       |
| **调整兰德指数**    | 外部     | 是           | 有标签时全面评估聚类一致性       |
| **NMI**             | 外部     | 是           | 信息论视角评估聚类与标签的关联   |
| **肘部法**          | 相对     | 否           | 选择最优簇数                     |



## 七、总结
- **无监督场景**：优先使用轮廓系数、Calinski-Harabasz指数和Davies-Bouldin指数。  
- **有监督场景**：选择调整兰德指数、NMI或FMI。  
- **选择簇数**：结合肘部法和轮廓分析。  
- **综合评估**：结合多个指标验证聚类结果的鲁棒性。