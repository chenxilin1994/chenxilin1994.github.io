

# 特征缩放与变换技术

## 1. 标准化（Standardization）

### 理论深化
- 核心原理：将数据转换为均值为0、标准差为1的正态分布。
- 数学公式：
  $$
  z = \frac{x - \mu}{\sigma}
  $$
  其中 $\mu$ 为均值，$\sigma$ 为标准差。
- 适用场景：大多数机器学习算法（如SVM、线性回归、神经网络），尤其是假设输入服从正态分布的模型。
- 优势：消除量纲差异，加速梯度下降收敛。
- 局限：对异常值敏感（因使用均值和标准差）。

### Python实现扩展
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 生成数据
np.random.seed(42)
data = np.random.normal(50, 10, 100).reshape(-1, 1)
data[5] = 200  # 添加异常值
df = pd.DataFrame(data, columns=['feature'])

# Scikit-learn实现
scaler = StandardScaler()
df['standardized'] = scaler.fit_transform(df[['feature']])

# 手动实现
mean = df['feature'].mean()
std = df['feature'].std()
df['standardized_manual'] = (df['feature'] - mean) / std

# 可视化对比
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['feature'], bins=30)
plt.title("原始数据分布")
plt.subplot(1, 2, 2)
plt.hist(df['standardized'], bins=30)
plt.title("标准化后分布")
plt.show()
```


## 2. 归一化（Min-Max Scaling）

### 理论深化
- 核心原理：将数据线性映射到[0, 1]区间。
- 数学公式：
  $$
  x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
  $$
- 变体：自定义范围[a, b]：
  $$
  x_{\text{scaled}} = a + \frac{(x - x_{\min})(b - a)}{x_{\max} - x_{\min}}
  $$
- 适用场景：图像处理（像素值归一化）、神经网络输入层、KNN等距离敏感算法。
- 缺点：受异常值影响大（如最大/最小值由异常点决定）。

### Python实现扩展
```python
from sklearn.preprocessing import MinMaxScaler

# Scikit-learn实现
scaler = MinMaxScaler(feature_range=(0, 1))
df['minmax'] = scaler.fit_transform(df[['feature']])

# 手动实现
min_val = df['feature'].min()
max_val = df['feature'].max()
df['minmax_manual'] = (df['feature'] - min_val) / (max_val - min_val)

# 带异常值的敏感性演示
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(df['feature'])
plt.title("原始数据箱线图")
plt.subplot(1, 2, 2)
plt.boxplot(df['minmax'])
plt.title("归一化后箱线图")
plt.show()
```


## 3. 鲁棒缩放（Robust Scaling）

### 理论深化
- 核心原理：使用中位数和四分位距（IQR）缩放，抵抗异常值。
- 数学公式：
  $$
  x_{\text{robust}} = \frac{x - \text{median}(X)}{IQR}
  $$
  其中 $IQR = Q3 - Q1$（Q3为75%分位数，Q1为25%分位数）。
- 适用场景：含异常值的数据集，非正态分布数据。
- 优势：对异常值不敏感，保留数据分布形态。

### Python实现扩展
```python
from sklearn.preprocessing import RobustScaler

# Scikit-learn实现
scaler = RobustScaler()
df['robust'] = scaler.fit_transform(df[['feature']])

# 手动实现
median = df['feature'].median()
q1 = df['feature'].quantile(0.25)
q3 = df['feature'].quantile(0.75)
iqr = q3 - q1
df['robust_manual'] = (df['feature'] - median) / iqr

# 对比标准化与鲁棒缩放
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['standardized'], bins=30)
plt.title("标准化分布")
plt.subplot(1, 2, 2)
plt.hist(df['robust'], bins=30)
plt.title("鲁棒缩放分布")
plt.show()
```


## 4. 最大绝对值缩放（MaxAbs Scaling）

### 理论深化
- 核心原理：将数据缩放到[-1, 1]区间，保持数据稀疏性。
- 数学公式：
  $$
  x_{\text{scaled}} = \frac{x}{|x_{\max}|}
  $$
- 适用场景：稀疏数据（如词频矩阵）、已中心化数据。
- 优点：保留零值，不移动数据中心。

### Python实现扩展
```python
from sklearn.preprocessing import MaxAbsScaler

# 生成稀疏数据
sparse_data = np.array([[1., -2., 0.], [0., 0., -0.5], [-3., 1., 0.]])
df_sparse = pd.DataFrame(sparse_data, columns=['A', 'B', 'C'])

# Scikit-learn实现
scaler = MaxAbsScaler()
df_sparse_scaled = pd.DataFrame(scaler.fit_transform(df_sparse), columns=df_sparse.columns)
print("最大绝对值缩放结果:\n", df_sparse_scaled)
```


## 5. 非线性变换

### 5.1 对数变换
- 数学公式：
  $$
  x_{\text{log}} = \log(x + \epsilon) \quad (\epsilon \text{防止零值})
  $$
- 适用场景：右偏分布（如收入、房价），异方差数据。

### 5.2 Box-Cox变换
- 数学公式：
  $$
  x^{(\lambda)} = 
  \begin{cases} 
  \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\
  \ln(x) & \lambda = 0 
  \end{cases}
  $$
  （要求 $x > 0$）
- 目标：寻找最优$\lambda$使数据接近正态分布。

### 5.3 Yeo-Johnson变换
- 数学公式：
  $$
  x^{(\lambda)} = 
  \begin{cases}
  \frac{(x+1)^\lambda - 1}{\lambda} & x \geq 0, \lambda \neq 0 \\
  \ln(x + 1) & x \geq 0, \lambda = 0 \\
  \frac{1 - (1 - x)^{2 - \lambda}}{2 - \lambda} & x < 0, \lambda \neq 2 \\
  -\ln(1 - x) & x < 0, \lambda = 2
  \end{cases}
  $$
  （允许 $x$ 为负值）

### Python实现扩展
```python
from scipy.stats import boxcox, yeojohnson

# 对数变换
df['log'] = np.log1p(df['feature'])  # log(x+1)

# Box-Cox变换（需数据>0）
df_pos = df[df['feature'] > 0].copy()
df_pos['boxcox'], lambda_bc = boxcox(df_pos['feature'])

# Yeo-Johnson变换（允许负值）
df['yeojohnson'], lambda_yj = yeojohnson(df['feature'])

# 可视化变换效果
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.hist(df['feature'], bins=30)
plt.title("原始数据")
plt.subplot(1, 3, 2)
plt.hist(df['log'], bins=30)
plt.title("对数变换后")
plt.subplot(1, 3, 3)
plt.hist(df['yeojohnson'], bins=30)
plt.title("Yeo-Johnson变换后")
plt.show()
```


## 6. 分位数变换（Quantile Transformation）

### 理论深化
- 核心原理：将数据映射到指定分布（如正态分布、均匀分布）的分位数。
- 数学步骤：
  1. 对每个特征计算分位数函数（累积分布函数的逆函数）。
  2. 将原始数据的分位数映射到目标分布的分位数。
- 优点：消除非线性关系，解决异方差问题。
- 缺点：易过拟合，需在训练集上拟合。

### Python实现扩展
```python
from sklearn.preprocessing import QuantileTransformer

# 映射到正态分布
transformer = QuantileTransformer(output_distribution='normal', random_state=42)
df['quantile_normal'] = transformer.fit_transform(df[['feature']])

# 映射到均匀分布
transformer_uniform = QuantileTransformer(output_distribution='uniform')
df['quantile_uniform'] = transformer_uniform.fit_transform(df[['feature']])

# 可视化对比
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['quantile_normal'], bins=30)
plt.title("分位数变换（正态）")
plt.subplot(1, 2, 2)
plt.hist(df['quantile_uniform'], bins=30)
plt.title("分位数变换（均匀）")
plt.show()
```


## 7. 幂变换（Power Transformation）

### 理论深化
- 核心原理：通过幂函数调整数据分布形态。
- 常见方法：
  - 平方/立方：放大差异，处理左偏分布。
  - 平方根：缓和右偏分布。
- 数学公式：
  $$
  x_{\text{sqrt}} = \sqrt{x}, \quad x_{\text{square}} = x^2
  $$

### Python实现扩展
```python
# 平方根变换（处理右偏）
df['sqrt'] = np.sqrt(df['feature'] - df['feature'].min() + 1e-6)  # 防止负数

# 平方变换（处理左偏）
df['square'] = df['feature']  2

# 对比分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['sqrt'], bins=30)
plt.title("平方根变换")
plt.subplot(1, 2, 2)
plt.hist(df['square'], bins=30)
plt.title("平方变换")
plt.show()
```


## 8. 单位向量归一化（L2 Normalization）

### 理论深化
- 核心原理：将样本向量缩放为单位范数（模长为1）。
- 数学公式：
  $$
  x_{\text{unit}} = \frac{x}{\|x\|_2} \quad \text{其中 } \|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}
  $$
- 适用场景：文本分类（TF-IDF向量）、余弦相似度计算。

### Python实现扩展
```python
from sklearn.preprocessing import Normalizer

# 样本数据
samples = np.array([[1., 2., 3.], [4., 5., 6.]])

# Scikit-learn实现（按行归一化）
normalizer = Normalizer(norm='l2')
samples_normalized = normalizer.fit_transform(samples)
print("L2归一化结果:\n", samples_normalized)

# 手动实现
norms = np.linalg.norm(samples, axis=1, keepdims=True)
samples_manual = samples / norms
print("手动计算结果:\n", samples_manual)
```


## 9. 自适应缩放（Adaptive Scaling）

### 理论深化
- 核心思想：基于模型反馈动态调整缩放策略。
- 实现方法：
  - 分位数分箱缩放：将特征划分为多个分位数区间，分别缩放。
  - 基于KL散度的缩放：最小化缩放前后分布差异。

### Python实现扩展
```python
from sklearn.preprocessing import PowerTransformer

# 基于Yeo-Johnson的自适应变换
transformer = PowerTransformer(method='yeo-johnson', standardize=True)
df['adaptive'] = transformer.fit_transform(df[['feature']])

# 分位数分箱缩放
from sklearn.preprocessing import QuantileTransformer
transformer = QuantileTransformer(n_quantiles=100, output_distribution='normal')
df['quantile_adaptive'] = transformer.fit_transform(df[['feature']])
```


## 10. 深度学习方法

### 理论深化
- 自编码器（Autoencoder）：通过编码-解码结构学习数据的低维表示。
  $$
  \text{Enc}(x) = h, \quad \text{Dec}(h) = \hat{x}, \quad \text{Loss} = \|x - \hat{x}\|^2
  $$
- 变分自编码器（VAE）：引入隐变量概率分布，生成更平滑的表示。
  $$
  \mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))
  $$

### Python实现扩展
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 构建自编码器
input_dim = df.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='linear')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# 训练自编码器进行特征学习
autoencoder.fit(df, df, epochs=50, batch_size=32, validation_split=0.2)

# 提取编码后的特征
encoder_model = Model(inputs=input_layer, outputs=encoder)
encoded_features = encoder_model.predict(df)
```



## 总结与选择策略

| 方法            | 适用场景                           | 优点                          | 缺点                          |
|---------------------|---------------------------------------|----------------------------------|----------------------------------|
| 标准化              | 正态分布数据、线性模型                | 消除量纲，保留异常值信息         | 对异常值敏感                     |
| 归一化              | 非正态分布、限定范围输入（如图像）    | 固定范围，适合梯度下降           | 受异常值影响大                   |
| 鲁棒缩放            | 含异常值数据、非参数模型              | 抗异常值，保留分布形态           | 不适用于假设正态分布的模型       |
| 非线性变换          | 非正态分布、异方差数据                | 改善分布形态，稳定方差           | 需选择合适变换，可能引入偏差     |
| 分位数变换          | 强非线性关系，需严格服从目标分布      | 消除非线性，解决异方差           | 计算成本高，可能过拟合           |
| 单位向量归一化      | 文本数据、余弦相似度计算              | 保留方向信息，适合稀疏数据       | 丢失尺度信息                     |
| 自适应缩放          | 复杂分布、动态调整需求                | 灵活适应数据特性                 | 实现复杂，需领域知识             |
| 深度学习            | 高维非线性数据、特征学习              | 自动提取高层次特征               | 计算资源消耗大，需大量数据       |

实践建议：
1. 数据探索先行：通过直方图、Q-Q图分析分布形态。
2. 模型适配选择：
   - 线性模型（如回归、SVM）：优先标准化/归一化。
   - 树模型（如随机森林）：通常无需缩放。
   - 神经网络：必须标准化/归一化输入。
3. 流水线集成：在`sklearn.pipeline`中集成缩放步骤，避免数据泄露。
4. 交叉验证调参：对比不同缩放方法对模型性能的影响。

```python
# 完整Pipeline示例（缩放 + 模型）
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# 构建Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 可替换为其他缩放器
    ('model', Ridge(alpha=1.0))
])

# 交叉验证评估
scores = cross_val_score(pipeline, df[['feature']], df['target'], cv=5)
print(f"平均R²分数: {scores.mean():.3f}")
```