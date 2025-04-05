# 最小角回归（LARS）

最小角回归（Least Angle Regression, LARS）是一种用于高维数据变量选择的高效算法。它不仅能够高效地构造模型，而且在变量选择问题上与 Lasso 等方法密切相关。LARS 算法能够计算出整个模型路径，展示从无变量到全变量模型过程中系数的变化情况。下面我们将从理论出发，详细推导其核心思想和计算步骤，并给出 Python 实践示例。


## 一、理论背景

### 1.1 最小角回归简介

最小角回归是一种逐步回归算法，其基本思想与前向逐步回归类似，但其变量进入模型的方式更加平滑。其主要特点有：
- **高效性**：一次性计算整个模型的路径（即系数随正则化参数变化的轨迹）。
- **等角前进**：在每一步中，不是简单地将与残差相关性最高的变量一次性加入模型，而是沿着与当前残差相关性最高变量的“等角”方向前进，直到另一变量与当前残差的相关性达到相同水平，此时共同进入模型。
- **与 Lasso 的关系**：LARS 算法经过适当修改可以用于求解 Lasso 问题，具有类似的变量选择效果，同时保持路径算法的高效性。

### 1.2 基本思想

假设有$n$个样本和$p$个特征，记响应变量为$y$以及预测变量矩阵为$X$。LARS 的基本步骤可以概括为：
1. **初始化**：令所有系数$\beta_j = 0$。
2. **选取最相关变量**：计算各预测变量与响应变量的相关系数，选出与响应最相关的变量（绝对相关性最大的变量）。
3. **等角前进**：沿着当前选中变量方向等角前进，使得进入模型的变量系数按一定比例同时增加，直到另一个变量与残差的相关性达到同一水平。
4. **更新方向与活跃集**：将新变量加入活跃集后，重新计算当前方向，在新的等角方向上前进，重复步骤3与4直至满足停止条件（例如所有变量均进入模型或残差足够小）。

这种逐步前进的方式使得 LARS 能够给出整个系数路径，而非单一的解。



## 二、详细推导过程

下面给出 LARS 算法核心部分的推导，说明如何在每一步计算前进方向以及系数更新。

### 2.1 初始化

假设数据已经中心化（即$y$和每个$x_j$均减去均值），初始状态下模型为：
$$
\hat{y} = 0,\quad \beta = \mathbf{0}.
$$
计算各变量与响应的相关性：
$$
c_j = X_j^\top y \quad (j = 1,2,\dots,p).
$$
设$C = \max_j |c_j|$，此时选择与$y$相关性最大的变量（或变量集合，若多个变量具有相同绝对相关性）构成活跃集$A$。

### 2.2 计算等角前进方向

设活跃集为$A$，对应变量矩阵为$X_A$。定义符号向量：
$$
s_j = \text{sign}(c_j) \quad \text{对于 } j\in A.
$$
接下来，需要确定一个方向向量$u$使得在该方向上前进时，各活跃变量与残差之间的相关性变化相同。令：
$$
u = X_A w,
$$
其中权重向量$w$可通过下面的公式确定：
$$
w = \frac{(X_A^\top X_A)^{-1} s_A}{\sqrt{s_A^\top (X_A^\top X_A)^{-1} s_A}},
$$
这样构造的$u$满足等角前进的性质，即所有活跃变量与$u$的投影具有相同的绝对值。

### 2.3 计算前进步长

沿着方向$u$进行前进时，系数$\beta_j$ 会发生变化。记当前残差为：
$$
r = y - \hat{y},
$$
其与非活跃变量$X_j$的相关性为：
$$
c_j = X_j^\top r.
$$
设$\gamma$为前进步长。对于每个非活跃变量$j \notin A$，计算其进入模型所需满足的条件：
$$
|c_j - \gamma a_j| = C - \gamma,
$$
其中$a_j = X_j^\top u$，而$C$是当前最大相关性。解此方程得到各候选变量对应的步长，选择最小正步长$\gamma^*$作为当前前进步长。当步长达到$\gamma^*$时，会有新的变量的相关性与活跃变量相等，触发活跃集的更新。

### 2.4 更新系数

在前进步长$\gamma^*$下，活跃变量对应的系数更新为：
$$
\beta_A \leftarrow \beta_A + \gamma^* w.
$$
同时更新预测值：
$$
\hat{y} \leftarrow \hat{y} + \gamma^* u.
$$
残差也随之更新：
$$
r \leftarrow y - \hat{y}.
$$
然后重新计算各变量与残差的相关性，判断是否有新的变量满足进入活跃集的条件，或者某些变量的系数应收缩为 0（在 Lasso 版本中会出现这种情况）。

### 2.5 重复迭代

重复上述步骤（计算方向、前进步长、更新系数）直至满足停止条件。最终，LARS 算法可以给出从零系数到全模型的完整路径，帮助研究者了解变量逐步进入模型的过程以及各变量对响应的贡献变化。



## 三、Python 实践

在 Python 中，`scikit-learn` 提供了 `Lars` 和 `LassoLars` 两个类，用于分别实现最小角回归和其 Lasso 版本。下面以 `Lars` 为例展示如何使用该算法。

### 3.1 环境准备

确保已安装 `scikit-learn` 以及其他必要的库：
```bash
pip install scikit-learn matplotlib
```

### 3.2 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lars
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成模拟数据
X, y = make_regression(n_samples=200, n_features=20, noise=10, random_state=42)

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建最小角回归模型
lars_model = Lars(n_nonzero_coefs=10)  # 可设置最多非零系数数目
lars_model.fit(X_train, y_train)

# 模型预测
y_pred = lars_model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("均方误差 (MSE):", mse)
print("R²得分:", r2)

# 可视化系数路径
plt.figure(figsize=(8, 6))
plt.plot(lars_model.coef_path_.T)
plt.xlabel('步数')
plt.ylabel('系数值')
plt.title('最小角回归系数路径')
plt.show()
```

### 3.3 代码说明

- **数据生成**：利用 `make_regression` 函数生成含噪声的回归数据集。
- **模型训练**：使用 `Lars` 类构造最小角回归模型，可以通过 `n_nonzero_coefs` 控制最终模型中最多保留的非零系数数目。
- **模型评估**：采用均方误差（MSE）和 R² 得分对模型进行评估。
- **系数路径可视化**：通过绘制系数路径图，可以直观观察各变量系数随算法迭代的变化情况，反映变量进入模型的顺序与变化趋势。
