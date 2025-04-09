# KNN算法

## 一、算法概述
K最近邻（K-Nearest Neighbors, KNN） 是一种基于实例的监督学习算法，适用于分类和回归任务。其核心思想是：相似样本在特征空间中距离相近，通过测量待预测样本与训练集样本的距离，找到最近的K个邻居，根据邻居的标签进行预测。


## 二、算法原理

### 1. 距离度量的数学本质与选择依据
距离度量是KNN算法的核心，决定了特征空间中样本的“相似性”。以下详细推导常见距离公式及其特性：

- **欧氏距离（L2）：**
  $$
  d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{k=1}^n (x_i^{(k)} - x_j^{(k)})^2}
  $$
  - 几何意义：多维空间中两点间的直线距离。
  - 导数分析：对每个维度误差平方求和后开根号，对大误差敏感（平方放大差异）。
  - 适用场景：特征尺度统一且分布接近高斯分布时效果最佳。

- **曼哈顿距离（L1）：**
  $$
  d(\mathbf{x}_i, \mathbf{x}_j) = \sum_{k=1}^n |x_i^{(k)} - x_j^{(k)}|
  $$
  - 几何意义：沿坐标轴行走的路径总和（城市街区距离）。
  - 鲁棒性：对异常值不敏感（线性惩罚），适合高维稀疏数据（如文本分类）。

- **闵可夫斯基距离的通用形式：**
  $$
  d(\mathbf{x}_i, \mathbf{x}_j) = \left( \sum_{k=1}^n |x_i^{(k)} - x_j^{(k)}|^p \right)^{1/p}
  $$
  - 参数$p$的影响：
    -$p=1$: 曼哈顿距离。
    -$p=2$: 欧氏距离。
    -$p \to \infty$: 切比雪夫距离（取最大维度差）。
  - 数学性质：当$p \geq 1$时满足距离公理（非负性、对称性、三角不等式）。

**距离选择实验方法：**
1. 对同一数据集尝试不同距离公式。
2. 使用交叉验证比较准确率。
3. 结合业务背景（如基因数据常用余弦相似度）。



### 2. K值选择的数学解释与过拟合分析
K值的作用本质上是控制模型的偏差-方差权衡：
- K=1：
  - 模型复杂度最高，完全依赖单个最近邻。
  - 训练误差为0（过拟合风险极大）。
  - 决策边界崎岖不平（高方差）。
  
- K=N（N为样本总数）：
  - 模型退化为全局多数类（分类）或全局均值（回归）。
  - 忽略局部特征（高偏差）。
  - 决策边界为直线或平面（完全平滑）。

数学证明（分类任务）：
假设真实条件概率为$P(y=c|\mathbf{x})$，K个近邻中属于类别$c$的数量为$N_c$，则预测概率：
$$
\hat{P}(y=c|\mathbf{x}) = \frac{N_c}{K}
$$
根据大数定律，当$K \to \infty$且$K/N \to 0$时，$\hat{P}(y=c|\mathbf{x})$依概率收敛于$P(y=c|\mathbf{x})$。但实际中需通过交叉验证选择平衡点。

交叉验证选择K值的步骤：
1. 将训练集划分为$m$折（如5折）。
2. 对每个候选K值，计算m次验证的平均准确率。
3. 选择使验证准确率最高的K值。
4. 可用网格搜索（GridSearchCV）自动化实现。



### 3. 决策规则的严格数学推导
分类任务：多数表决法的概率解释
设测试样本$\mathbf{x}$的K个近邻标签为$\{y_1, y_2, ..., y_K\}$，其中$y_i \in \{c_1, c_2, ..., c_M\}$。

定义指示函数：
$$
I(y_i = c_m) = 
\begin{cases} 
1 & \text{if } y_i = c_m \\
0 & \text{otherwise}
\end{cases}
$$
则类别$c_m$的得票数为：
$$
V(c_m) = \sum_{i=1}^K I(y_i = c_m)
$$
预测结果为：
$$
\hat{y} = \arg\max_{c_m} V(c_m)
$$
数学性质：
- 等价于极大后验概率估计（MAP）当先验概率相等时。
- 若各类别先验不相等，可引入加权投票（如权重=类别先验的倒数）。

回归任务：均值预测的无偏性证明
设K个近邻的目标值为$\{y_1, y_2, ..., y_K\}$，预测值为：
$$
\hat{y} = \frac{1}{K} \sum_{i=1}^K y_i
$$
假设真实模型为$y = f(\mathbf{x}) + \epsilon$，其中$\epsilon$为噪声且$E(\epsilon)=0$，则：
$$
E(\hat{y}) = E\left( \frac{1}{K} \sum_{i=1}^K f(\mathbf{x}_i) + \epsilon_i \right) = \frac{1}{K} \sum_{i=1}^K f(\mathbf{x}_i)
$$
当$\mathbf{x}_i$接近$\mathbf{x}$时，$f(\mathbf{x}_i) \approx f(\mathbf{x})$，因此$\hat{y}$是$f(\mathbf{x})$的无偏估计。



### 4. 加权KNN的数学形式
为更近的邻居分配更高权重，常见权重函数：
- 反距离权重：
  $$
  w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon} \quad (\epsilon \text{防止除以零})
  $$
- 高斯核权重：
  $$
  w_i = \exp\left( -\frac{d(\mathbf{x}, \mathbf{x}_i)^2}{2\sigma^2} \right)
  $$
  
加权分类预测公式：
$$
\hat{y} = \arg\max_{c} \sum_{i=1}^K w_i \cdot I(y_i = c)
$$

加权回归预测公式：
$$
\hat{y} = \frac{\sum_{i=1}^K w_i y_i}{\sum_{i=1}^K w_i}
$$



## 三、Python实践
### 使用Scikit-learn实现

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测与评估
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### 手动实现KNN分类器
```python
import numpy as np
from collections import Counter

class MyKNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)
    
    def _predict_one(self, x):
        # 计算欧氏距离
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # 获取最近的k个样本的索引
        k_indices = np.argsort(distances)[:self.k]
        # 统计类别
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# 使用手动实现的KNN
my_knn = MyKNN(k=5)
my_knn.fit(X_train, y_train)
y_pred_custom = my_knn.predict(X_test)
print("Custom KNN Accuracy:", accuracy_score(y_test, y_pred_custom))
```

### 手动实现加权KNN
```python
class WeightedKNN:
    def __init__(self, k=5, weight='uniform'):
        self.k = k
        self.weight = weight  # 'uniform' 或 'distance'
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])
    
    def _predict_one(self, x):
        # 计算距离
        distances = np.sqrt(np.sum((self.X_train - x)2, axis=1))
        # 获取最近的k个样本的索引和距离
        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]
        
        # 计算权重
        if self.weight == 'uniform':
            weights = np.ones_like(k_distances)
        elif self.weight == 'distance':
            weights = 1 / (k_distances + 1e-8)  # 避免除以零
        
        # 统计加权投票
        unique_labels = np.unique(k_labels)
        weighted_votes = {}
        for label in unique_labels:
            weighted_votes[label] = np.sum(weights[k_labels == label])
        return max(weighted_votes, key=weighted_votes.get)

# 测试加权KNN
weighted_knn = WeightedKNN(k=5, weight='distance')
weighted_knn.fit(X_train, y_train)
y_pred_weighted = weighted_knn.predict(X_test)
print("Weighted KNN Accuracy:", accuracy_score(y_test, y_pred_weighted))
```



## 四、复杂度分析与优化算法
### 1. 时间复杂度
- 训练阶段：$O(1)$（仅存储数据）。
- 预测阶段：$O(nd + n\log k)$（n为训练样本数，d为特征维度）。
  
### 2. 空间复杂度
- $O(nd)$（需存储全部训练数据）。

### 3. 优化方法
- KD-Tree：
  - 构建复杂度：$O(n\log n)$
  - 查询复杂度：$O(\log n)$（最坏情况下仍为$O(n)$）。
- Ball Tree：
  - 对高维数据更高效，通过超球体划分空间。
- LSH（局部敏感哈希）：
  - 近似最近邻搜索，适用于海量数据。

```python
# Scikit-learn中使用KD-Tree
knn_kd = KNeighborsClassifier(
    n_neighbors=5, 
    algorithm='kd_tree', 
    leaf_size=30
)
knn_kd.fit(X_train, y_train)
```



## 五、算法优缺点
- 优点：
    - 简单直观，无需训练过程。
    - 对数据分布无假设，适用于非线性问题。

- 缺点：
    - 计算复杂度高（需存储全部训练数据）。
    - 对高维数据和噪声敏感。

需要合理选择K值和距离度量方式。

**维度灾难（Curse of Dimensionality）**
在高维空间中，欧氏距离失去区分能力：
- 现象：所有样本间的距离趋于相同。
- 数学证明：假设特征独立且服从均匀分布，数据维度为$d$，则样本间距离的方差为：
  $$
  \text{Var}(d(\mathbf{x}_i, \mathbf{x}_j)) = \frac{d}{3}
  $$
  相对方差（方差与期望平方之比）：
  $$
  \frac{\text{Var}(d)}{[E(d)]^2} \propto \frac{1}{d}
  $$
  随维度增加，相对方差趋近于0，距离分布高度集中。

解决方法：
- 特征选择（如卡方检验、互信息）。
- 降维技术（PCA、t-SNE）。
- 使用曼哈顿距离或余弦相似度。

