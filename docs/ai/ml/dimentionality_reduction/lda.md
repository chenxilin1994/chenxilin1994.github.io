# LDA降维

## 一、LDA的核心思想
线性判别分析（Linear Discriminant Analysis, LDA）是一种有监督的降维方法，目标是找到一组投影方向，使得**类内样本的方差最小化**，同时**类间样本的均值差异最大化**。其核心优化目标为：

$$
J(W) = \frac{W^T S_b W}{W^T S_w W}
$$

其中：
- $S_w$：类内散度矩阵（Within-class scatter matrix）
- $S_b$：类间散度矩阵（Between-class scatter matrix）



## 二、公式推导

**1. 符号定义**
- 样本集：$X = \{x_1, x_2, ..., x_n\}$，维度为 $d$
- 类别标签：$y = \{y_1, y_2, ..., y_n\}$，共 $C$ 个类别
- 第 $i$ 类的样本集合：$X_i$，样本数为 $n_i$
- 第 $i$ 类的均值向量：$\mu_i = \frac{1}{n_i} \sum_{x \in X_i} x$
- 总体均值向量：$\mu = \frac{1}{n} \sum_{x \in X} x$

**2. 散度矩阵定义**
- **类内散度矩阵** $S_w$：
  $$
  S_w = \sum_{i=1}^C \sum_{x \in X_i} (x - \mu_i)(x - \mu_i)^T
  $$

- **类间散度矩阵** $S_b$：
  $$
  S_b = \sum_{i=1}^C n_i (\mu_i - \mu)(\mu_i - \mu)^T
  $$

**3. 优化目标**
最大化广义瑞利商：
$$
J(W) = \frac{W^T S_b W}{W^T S_w W}
$$

**4. 求解投影矩阵**
- 对广义特征方程求解：
  $$
  S_b W = \lambda S_w W
  $$
- 等价于求解 $S_w^{-1} S_b$ 的特征向量，取前 $k$ 个最大特征值对应的特征向量组成投影矩阵 $W$，其中 $k \leq C-1$。



## 三、Python代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.W = None

    def fit(self, X, y):
        # 计算类内散度矩阵 Sw 和类间散度矩阵 Sb
        n_features = X.shape[1]
        classes = np.unique(y)
        n_classes = len(classes)

        # 总体均值
        mean_global = np.mean(X, axis=0)

        # 初始化 Sw 和 Sb
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_c = np.mean(X_c, axis=0)

            # 类内散度矩阵
            Sw += (X_c - mean_c).T @ (X_c - mean_c)

            # 类间散度矩阵
            delta = (mean_c - mean_global).reshape(-1, 1)
            Sb += n_c * (delta @ delta.T)

        # 解决 Sw 的奇异性问题：添加正则化项
        Sw += 1e-6 * np.eye(n_features)

        # 计算 Sw^{-1} Sb 的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

        # 取前 k 个最大特征值对应的特征向量
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx].real
        k = min(self.n_components, n_classes - 1) if self.n_components else n_classes - 1
        self.W = eigenvectors[:, :k]

    def transform(self, X):
        return X @ self.W

# 实验：使用鸢尾花数据集验证
# 加载数据
data = load_iris()
X, y = data.data, data.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 自定义LDA降维
lda = LDA(n_components=2)
lda.fit(X_train, y_train)
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# 使用sklearn LDA对比
sklearn_lda = SklearnLDA(n_components=2)
X_train_sk = sklearn_lda.fit_transform(X_train, y_train)
X_test_sk = sklearn_lda.transform(X_test)

# 可视化降维结果
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis')
plt.title('Custom LDA Projection')

plt.subplot(122)
plt.scatter(X_train_sk[:, 0], X_train_sk[:, 1], c=y_train, cmap='viridis')
plt.title('Sklearn LDA Projection')
plt.show()

# 分类性能评估
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_lda, y_train)
y_pred = knn.predict(X_test_lda)
print(f"Custom LDA Accuracy: {accuracy_score(y_test, y_pred):.4f}")

knn.fit(X_train_sk, y_train)
y_pred_sk = knn.predict(X_test_sk)
print(f"Sklearn LDA Accuracy: {accuracy_score(y_test, y_pred_sk):.4f}")
```



## 四、代码解析
1. **数据预处理**：标准化处理消除量纲影响。
2. **散度矩阵计算**：
   - 遍历每个类别计算类内散度 $S_w$ 和类间散度 $S_b$。
3. **特征分解**：求解 $S_w^{-1}S_b$ 的特征向量，选取前 $k$ 个作为投影矩阵。
4. **正则化处理**：对 $S_w$ 添加微小单位矩阵避免奇异。
5. **可视化与评估**：对比自定义LDA与sklearn的LDA效果，并通过KNN分类器评估性能。



## 五、关键点说明
- **降维维度限制**：LDA最多可将数据降至 $C-1$ 维（$C$ 为类别数）。
- **奇异性问题**：当样本维度大于样本数时，$S_w$ 可能不可逆，需正则化或使用伪逆。
- **监督性质**：LDA利用类别信息，适合分类任务的特征降维。