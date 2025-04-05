# 逻辑回归

逻辑回归是一种用于解决二分类问题的广泛应用的统计学习方法。它通过建立自变量与因变量（取 0 或 1）之间的概率关系来实现分类任务，具有模型简单、解释性强和求解高效等优点。本文将从理论基础出发，详细推导逻辑回归的模型建立、对数似然函数及梯度求解过程，最后结合 Python 示例展示如何应用逻辑回归进行分类任务。


## 一、理论基础

### 1.1 模型定义

逻辑回归的核心在于用**逻辑函数（sigmoid 函数）**将线性回归的输出映射到 [0,1] 之间，从而作为概率估计。给定输入向量$\mathbf{x} = (x_1, x_2, \dots, x_p)$和参数向量$\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_p)$，逻辑回归模型假设：

$$
P(y=1\mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + \exp(-z)}, \quad \text{其中 } z = \beta_0 + \sum_{j=1}^{p} \beta_j x_j.
$$

对应的概率$P(y=0\mid \mathbf{x})$则为$1 - \sigma(z)$。

### 1.2 对数几率与决策边界

利用逻辑函数的性质，可以将概率的比值（几率）取对数，得到**对数几率（log odds）**：

$$
\log\frac{P(y=1\mid \mathbf{x})}{P(y=0\mid \mathbf{x})} = z = \beta_0 + \sum_{j=1}^{p} \beta_j x_j.
$$

这一形式表明，尽管模型输出的是概率，但决策边界实际上是线性的，即当$z = 0$时（对应概率$\sigma(0)=0.5$），分类器在两类之间做出划分。



## 二、详细推导过程

下面我们将从最大似然估计出发，推导逻辑回归的损失函数及梯度计算过程。

### 2.1 建立似然函数

假设训练数据集为$\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$，其中$y_i \in \{0, 1\}$。在逻辑回归中，每个样本的概率模型为：

$$
P(y_i\mid \mathbf{x}_i; \boldsymbol{\beta}) = \sigma(z_i)^{y_i} \left[1 - \sigma(z_i)\right]^{1-y_i}, \quad \text{其中 } z_i = \beta_0 + \mathbf{x}_i^\top \boldsymbol{\beta}_{1:p}.
$$

整个数据集的似然函数为各样本概率的乘积：

$$
L(\boldsymbol{\beta}) = \prod_{i=1}^{n} \sigma(z_i)^{y_i} \left[1-\sigma(z_i)\right]^{1-y_i}.
$$

### 2.2 对数似然函数

为了简化求解，通常取对数，将乘积转化为求和，得到对数似然函数：

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \Big[ y_i \log \sigma(z_i) + (1-y_i) \log \big(1-\sigma(z_i)\big) \Big].
$$

### 2.3 损失函数（交叉熵）

逻辑回归的目标是最大化对数似然函数。通常将其转化为最小化负对数似然函数（即交叉熵损失）：

$$
J(\boldsymbol{\beta}) = -\ell(\boldsymbol{\beta}) = -\sum_{i=1}^{n} \Big[ y_i \log \sigma(z_i) + (1-y_i) \log \big(1-\sigma(z_i)\big) \Big].
$$

### 2.4 梯度推导

为了利用梯度下降等优化方法，我们需要求损失函数关于参数$\boldsymbol{\beta}$的梯度。注意到对于单个样本，有

$$
\sigma(z_i) = \frac{1}{1+\exp(-z_i)},
$$
以及其导数为
$$
\frac{d\sigma(z_i)}{dz_i} = \sigma(z_i) \big(1-\sigma(z_i)\big).
$$

对$\beta_j$求偏导（包括截距项$\beta_0$）可得：

$$
\frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j} = -\sum_{i=1}^{n} \left[ \frac{y_i}{\sigma(z_i)} - \frac{1-y_i}{1-\sigma(z_i)} \right] \cdot \sigma(z_i)(1-\sigma(z_i)) \cdot x_{ij},
$$
其中$x_{i0} = 1$（对应截距项）。

经过整理，可以得到更简洁的形式：

$$
\frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j} = \sum_{i=1}^{n} \big[\sigma(z_i) - y_i\big] \, x_{ij}.
$$

这一结果表明，梯度正是样本预测概率与真实标签之间的误差，加权累加各样本的特征值。

### 2.5 参数更新

利用梯度下降法更新参数：

$$
\beta_j^{(t+1)} = \beta_j^{(t)} - \eta \, \frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j},
$$
其中$\eta$ 为学习率。

此外，还可以采用更高级的优化方法（如牛顿-拉夫森方法），其中利用 Hessian 矩阵进行二阶信息的更新，但梯度下降已经足够直观且易于实现。



## 三、Python 实践

在 Python 中，`scikit-learn` 提供了便捷的 `LogisticRegression` 类用于逻辑回归。下面以一个示例代码说明如何构建、训练逻辑回归模型并评估其分类性能。

### 3.1 环境准备

确保安装了 `scikit-learn`、`numpy` 以及 `matplotlib`：
```bash
pip install scikit-learn matplotlib
```

### 3.2 示例代码

下面的代码利用生成的二分类数据集，训练逻辑回归模型，并对模型进行评估和决策边界可视化。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 生成二分类数据集
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 模型评估
print("准确率:", accuracy_score(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

# 可视化决策边界
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr', edgecolors='k')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('逻辑回归决策边界')
plt.show()
```

### 3.3 代码说明

- **数据生成**：利用 `make_classification` 生成一个二维特征的二分类数据集，便于后续可视化决策边界。  
- **模型训练**：使用 `LogisticRegression` 构建并拟合模型。  
- **模型评估**：通过准确率、混淆矩阵和分类报告评估模型性能。  
- **决策边界可视化**：利用网格数据计算预测概率，并绘制出决策边界，直观展示逻辑回归的分类效果。

