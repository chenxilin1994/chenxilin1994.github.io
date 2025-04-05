

# 正交匹配追踪

正交匹配追踪（Orthogonal Matching Pursuit, OMP）是一种贪心算法，常用于求解稀疏表示问题。给定一个冗余字典，通过迭代选择与当前残差最相关的原子（字典列）并正交投影更新残差，OMP 能够高效地从大量候选原子中选择出最优子集，从而实现对信号的稀疏表示。本文将从理论基础出发，详细推导 OMP 的求解过程，并结合 Python 示例展示如何在实际中应用该方法。



## 一、理论基础

### 1.1 问题描述

在稀疏表示问题中，假设给定观测向量$\mathbf{y} \in \mathbb{R}^n$以及一个字典矩阵$D \in \mathbb{R}^{n \times m}$（通常$m > n$），我们的目标是寻找一个稀疏系数向量$\mathbf{x} \in \mathbb{R}^m$，使得

$$
\mathbf{y} \approx D \mathbf{x},
$$

且$\mathbf{x}$ 的非零元素个数尽可能少。通常可以形式化为如下优化问题：

$$
\min_{\mathbf{x}} \|\mathbf{x}\|_0 \quad \text{subject to} \quad \|\mathbf{y} - D\mathbf{x}\|_2 \leq \varepsilon,
$$

其中$\|\mathbf{x}\|_0$ 表示$\mathbf{x}$ 中非零元素的个数，$\varepsilon$ 是允许的重构误差。由于该问题是 NP 难的，OMP 作为一种贪心算法在求解中得到了广泛应用。

### 1.2 算法思想

OMP 的基本思想为：  
1. 从初始残差$\mathbf{r}^{(0)} = \mathbf{y}$出发，逐步选择与当前残差最相关的字典原子（即与残差内积绝对值最大的列）。  
2. 将选中的原子加入到活跃集（支持集）中。  
3. 对于活跃集中的所有原子，通过最小二乘求解构造其系数，从而使得新的重构向量$\hat{\mathbf{y}}$更接近原始观测向量。  
4. 用重构向量更新残差，重复上述过程直至残差足够小或达到预设的迭代次数。

这种逐步正交投影的方式保证了每一步都在当前选定的原子子空间中达到最优逼近，从而提高了稀疏重构的效果。



## 二、详细推导过程

### 2.1 问题转化

给定观测向量$\mathbf{y}$和字典$D$，目标是寻找支持集$S \subseteq \{1, 2, \dots, m\}$和系数向量$\mathbf{x}$使得

$$
\min_{\mathbf{x}} \|\mathbf{y} - D_S \mathbf{x}_S\|_2^2,
$$

其中$D_S$表示仅包含支持集$S$中原子的字典子矩阵，而$\mathbf{x}_S$是对应的系数子向量。

### 2.2 算法迭代步骤

OMP 的核心迭代步骤如下：

1. **初始化：**
   - 设初始残差为：  
    $$
     \mathbf{r}^{(0)} = \mathbf{y},
    $$
   - 活跃集$S^{(0)} = \emptyset$；
   - 初始迭代计数$t = 0$。

2. **选择原子：**  
   在第$t$次迭代中，计算每个字典原子与当前残差的相关性：
  $$
   c_j = \langle \mathbf{d}_j, \mathbf{r}^{(t)} \rangle, \quad j=1,2,\dots,m.
  $$
   选取使得绝对值$|c_j|$ 最大的原子索引$j^*$：
  $$
   j^* = \arg\max_{j} |c_j|.
  $$
   更新支持集：
  $$
   S^{(t+1)} = S^{(t)} \cup \{ j^* \}.
  $$

3. **正交投影求解系数：**  
   在新的支持集$S^{(t+1)}$下，求解最小二乘问题：
  $$
   \mathbf{x}_{S^{(t+1)}} = \arg\min_{\mathbf{z}} \left\| \mathbf{y} - D_{S^{(t+1)}} \mathbf{z} \right\|_2^2.
  $$
   这一步可以通过正规方程或 QR 分解等方法高效求解。

4. **更新重构向量和残差：**  
   得到新的重构向量：
  $$
   \hat{\mathbf{y}} = D_{S^{(t+1)}} \mathbf{x}_{S^{(t+1)}},
  $$
   更新残差：
  $$
   \mathbf{r}^{(t+1)} = \mathbf{y} - \hat{\mathbf{y}}.
  $$

5. **停止条件：**  
   当满足以下任一条件时停止迭代：
   -$\|\mathbf{r}^{(t+1)}\|_2 \leq \varepsilon$；
   - 迭代次数达到预设上限；
   - 支持集大小达到预设稀疏度上限。

### 2.3 推导说明

在每一步中，通过选取与残差最相关的原子，可以最大限度地降低当前的重构误差。而通过对支持集内的原子进行正交投影求解最小二乘问题，则保证了当前解在已选原子子空间内的最优性。OMP 正是利用这种贪心策略实现了对稀疏表示问题的高效求解。



## 三、Python 实践

在 Python 中，我们可以利用 `numpy` 实现 OMP 算法。下面提供一个简单示例，演示如何生成数据、构造字典，并利用 OMP 算法对信号进行稀疏重构。

### 3.1 环境准备

确保安装了 `numpy` 和 `matplotlib`：
```bash
pip install numpy matplotlib
```

### 3.2 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

def omp(y, D, sparsity, tol=1e-6):
    """
    正交匹配追踪算法

    参数：
    y : 观测向量，形状 (n, )
    D : 字典矩阵，形状 (n, m)
    sparsity : 预设稀疏度上限（最大非零系数个数）
    tol : 残差阈值

    返回：
    x : 稀疏系数向量，形状 (m, )
    S : 支持集（非零系数的索引列表）
    """
    n, m = D.shape
    r = y.copy()  # 初始残差
    S = []      # 初始化支持集
    x = np.zeros(m)  # 初始化系数向量

    for t in range(sparsity):
        # 计算所有原子与残差的相关性
        correlations = D.T @ r
        # 选择绝对相关性最大的原子
        j = np.argmax(np.abs(correlations))
        if j in S:
            break
        S.append(j)
        # 从字典中提取当前支持集对应的子矩阵
        D_S = D[:, S]
        # 求解最小二乘问题： min || y - D_S * z ||_2^2
        z, _, _, _ = np.linalg.lstsq(D_S, y, rcond=None)
        # 更新稀疏系数向量
        x[S] = z
        # 更新残差
        r = y - D_S @ z
        # 若残差足够小则提前停止
        if np.linalg.norm(r) < tol:
            break

    return x, S

# 生成示例数据
np.random.seed(42)
n = 50   # 信号维度
m = 100  # 字典中原子的数量
sparsity_level = 5  # 真实信号的非零个数

# 构造字典矩阵 D（标准化每一列）
D = np.random.randn(n, m)
D = D / np.linalg.norm(D, axis=0)

# 构造稀疏系数向量 x_true
x_true = np.zeros(m)
nonzero_indices = np.random.choice(m, sparsity_level, replace=False)
x_true[nonzero_indices] = np.random.randn(sparsity_level)

# 生成观测向量 y
y = D @ x_true

# 使用 OMP 算法重构稀疏系数
x_est, S_est = omp(y, D, sparsity=10)

print("真实非零系数索引：", np.sort(nonzero_indices))
print("OMP估计的支持集：", np.sort(S_est))
print("真实系数：", x_true[nonzero_indices])
print("估计系数：", x_est[S_est])

# 可视化真实系数与估计系数对比
plt.figure(figsize=(8, 6))
plt.stem(x_true, markerfmt='bo', label='真实系数', use_line_collection=True)
plt.stem(x_est, markerfmt='rx', label='估计系数', use_line_collection=True)
plt.xlabel('索引')
plt.ylabel('系数值')
plt.title('OMP 重构：真实系数与估计系数对比')
plt.legend()
plt.show()
```

### 3.3 代码说明

- **数据生成**：随机构造一个字典$D$（并对每一列进行标准化），随机生成一个稀疏系数向量$\mathbf{x}_{true}$，并利用$\mathbf{y} = D \mathbf{x}_{true}$得到观测向量。
- **OMP 实现**：函数 `omp` 实现了正交匹配追踪算法，包括残差更新、支持集的扩展以及最小二乘求解。  
- **结果展示**：代码输出真实支持集与估计支持集，并通过图形对比真实系数和 OMP 重构的系数，直观展示算法效果。



## 四、总结

本文详细介绍了正交匹配追踪（OMP）的理论基础、数学推导过程及 Python 实践：
- **理论部分**阐述了稀疏表示问题的背景以及 OMP 通过逐步选择与残差最相关的字典原子并进行正交投影，实现稀疏重构的基本思想。  
- **推导过程**中，我们详细描述了每一步骤的数学原理，从选择原子、最小二乘求解到残差更新，说明了算法的核心机制。  
- **Python 实践**展示了如何利用 `numpy` 实现 OMP 算法，并通过示例代码验证了算法在稀疏信号重构中的有效性。

正交匹配追踪作为一种简单高效的贪心算法，在信号处理、压缩感知和特征选择等领域具有广泛应用。希望本文能帮助你全面理解 OMP 的原理和实现方法。