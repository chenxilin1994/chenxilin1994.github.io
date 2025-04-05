# 异常值处理

## 异常值检测方法详解
### 1. 基于统计的检测方法

#### 理论深化
- 假设前提：数据服从正态分布或已知分布形态。
- Z-Score法：
  - 数学原理：计算观测值与均值的标准差倍数，适用于单变量分析。
 $$
  Z = \frac{x - \mu}{\sigma}
 $$
  - 阈值设定：通常$|Z| > 3$ 为异常（99.7%数据在±3σ内）。
  
- 修正Z-Score（MAD）：
  - 鲁棒性改进：用中位数（Median）代替均值，中位数绝对偏差（MAD）代替标准差。
 $$
  \text{MAD} = \text{median}(|x_i - \text{median}(X)|)
 $$
 $$
  \text{修正Z} = \frac{0.6745(x - \text{median}(X))}{\text{MAD}}
 $$
  - 适用场景：数据含异常值或非正态分布。

- IQR（四分位距）法：
  - 数学定义：
   $$
    Q1 = 25\% \text{分位数}, \quad Q3 = 75\% \text{分位数}, \quad IQR = Q3 - Q1
   $$
    正常值范围：$[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$
  - 优点：对非正态分布鲁棒，广泛用于箱线图。

#### Python实现扩展
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 生成含异常值的数据
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 100), np.array([10, -8, 12])])
df = pd.DataFrame({'value': data})

# Z-Score检测
def detect_zscore(data, threshold=3):
    z = np.abs((data - data.mean()) / data.std())
    return np.where(z > threshold)[0]

outliers_z = detect_zscore(df['value'])
print(f"Z-Score检测异常值索引: {outliers_z}")

# MAD修正Z-Score检测
def detect_mad(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z) > threshold)[0]

outliers_mad = detect_mad(df['value'])
print(f"MAD检测异常值索引: {outliers_mad}")

# IQR检测
def detect_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data < lower) | (data > upper)].index.tolist()

outliers_iqr = detect_iqr(df['value'])
print(f"IQR检测异常值索引: {outliers_iqr}")

# 可视化箱线图
plt.figure(figsize=(10, 4))
plt.boxplot(df['value'], vert=False)
plt.title("Boxplot with Outliers")
plt.show()
```



### 2. 基于距离的检测方法

#### 理论深化
- K近邻（KNN）距离：
  - 定义：计算每个点到其k个最近邻的距离，如平均距离或最大距离。
 $$
  \text{Outlier Score} = \frac{1}{k} \sum_{i=1}^k d(x, x_i^{(k)})
 $$
  - 适用性：全局异常检测，但对高维数据敏感。

- 局部异常因子（LOF）：
  - 核心思想：比较局部密度，密度显著低于邻居的点为异常。
  - 数学推导：
    1. k-距离：点$x$到第k近邻的距离$d_k(x)$。
    2. 可达距离：$ \text{reach\_dist}_k(x, y) = \max(d_k(y), d(x, y))$
    3. 局部可达密度（LRD）：
   $$
    \text{LRD}_k(x) = \frac{1}{\left(\frac{1}{k} \sum_{y \in N_k(x)} \text{reach\_dist}_k(x, y)\right)}
   $$
    4. LOF值：
   $$
    \text{LOF}_k(x) = \frac{\frac{1}{k} \sum_{y \in N_k(x)} \text{LRD}_k(y)}{\text{LRD}_k(x)}
   $$
  - 阈值：LOF > 1 表示密度低于邻居，可能为异常。

#### Python实现扩展
```python
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from scipy.spatial.distance import euclidean

# KNN距离检测
def knn_outlier_detection(data, k=5, threshold=2.0):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data.values.reshape(-1, 1))
    distances, _ = neigh.kneighbors(data.values.reshape(-1, 1))
    avg_dist = distances[:, 1:].mean(axis=1)  # 排除自身
    outliers = np.where(avg_dist > threshold)[0]
    return outliers

outliers_knn = knn_outlier_detection(df['value'], k=5, threshold=2.0)
print(f"KNN检测异常值索引: {outliers_knn}")

# LOF检测
def detect_lof(data, k=5, contamination=0.1):
    lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    preds = lof.fit_predict(data.values.reshape(-1, 1))
    return np.where(preds == -1)[0]

outliers_lof = detect_lof(df['value'], k=5)
print(f"LOF检测异常值索引: {outliers_lof}")

# 可视化LOF评分
lof_scores = -LocalOutlierFactor(n_neighbors=5).fit(df[['value']]).negative_outlier_factor_
plt.scatter(df.index, df['value'], c=lof_scores, cmap='viridis')
plt.colorbar(label="LOF Score")
plt.title("LOF Outlier Scores")
plt.show()
```



### 3. 基于密度的检测方法

#### 理论深化
- 孤立森林（Isolation Forest）：
  - 核心思想：异常点容易被随机树快速隔离（路径长度较短）。
  - 数学公式：  
    异常得分$s(x)$：
   $$
    s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
   $$
    其中$c(n) = 2H(n-1) - 2(n-1)/n$ 为路径长度均值，$H$ 为调和数。

- DBSCAN聚类：
  - 原理：基于密度可达性划分簇，无法归入任何簇的点为异常。
  - 参数：邻域半径$\epsilon$，最小点数$min\_samples$。

##### Python实现扩展
```python
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# 孤立森林检测
def detect_isolation_forest(data, contamination=0.1):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(data.values.reshape(-1, 1))
    return np.where(preds == -1)[0]

outliers_iso = detect_isolation_forest(df['value'])
print(f"孤立森林检测异常值索引: {outliers_iso}")

# DBSCAN检测
def detect_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data.values.reshape(-1, 1))
    return np.where(clusters == -1)[0]

outliers_dbscan = detect_dbscan(df['value'], eps=2.0, min_samples=3)
print(f"DBSCAN检测异常值索引: {outliers_dbscan}")

# 可视化DBSCAN聚类结果
plt.scatter(df.index, df['value'], c=dbscan.fit_predict(df[['value']]))
plt.title("DBSCAN Clustering (Outliers in Black)")
plt.show()
```



### 4. 基于机器学习的检测方法

#### 理论深化
- One-Class SVM：
  - 原理：在无标签数据中学习一个决策边界，将正常数据包含在内。
  - 核函数：RBF核通过参数$\nu$ 控制异常点比例。
  
- 自编码器（Autoencoder）：
  - 重建误差：异常点在潜在空间难以准确重建，误差较高。
 $$
  \text{Error} = \|x - \text{Decoder}(\text{Encoder}(x))\|^2
 $$

#### Python实现扩展
```python
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# One-Class SVM检测
def detect_ocsvm(data, nu=0.05):
    ocsvm = OneClassSVM(kernel='rbf', nu=nu)
    preds = ocsvm.fit_predict(data.values.reshape(-1, 1))
    return np.where(preds == -1)[0]

outliers_ocsvm = detect_ocsvm(df['value'], nu=0.05)
print(f"One-Class SVM检测异常值索引: {outliers_ocsvm}")

# 自编码器检测
def autoencoder_detection(data, threshold=3):
    model = Sequential([
        Dense(8, activation='relu', input_shape=(1,)),
        Dense(4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(data.values, data.values, epochs=50, batch_size=32, verbose=0)
    reconstructions = model.predict(data.values)
    mse = np.mean(np.power(data.values - reconstructions.flatten(), 2), axis=1)
    return np.where(mse > threshold)[0]

outliers_ae = autoencoder_detection(df['value'], threshold=2.0)
print(f"自编码器检测异常值索引: {outliers_ae}")
```



### 5. 可视化与时间序列方法

#### 理论深化
- 时间序列分解（STL）：
  - 原理：将序列分解为趋势、季节性和残差项，残差异常即为异常点。
 $$
  Y_t = T_t + S_t + R_t
 $$
  - 检测方法：对残差应用Z-Score或IQR。

- 滑动窗口统计：
  - 定义：计算窗口内均值/标准差，超出历史范围的点为异常。

#### Python实现扩展
```python
from statsmodels.tsa.seasonal import STL

# 时间序列分解检测
def detect_stl_outliers(series, period=12, threshold=3):
    stl = STL(series, period=period)
    res = stl.fit()
    residuals = res.resid
    z_scores = (residuals - residuals.mean()) / residuals.std()
    return np.where(np.abs(z_scores) > threshold)[0]

# 生成时间序列数据
dates = pd.date_range('2023-01-01', periods=100)
ts_data = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.2, 100)
ts_data[90] = 5  # 加入异常点
ts = pd.Series(ts_data, index=dates)

# 检测并可视化
outliers_stl = detect_stl_outliers(ts, period=12)
plt.figure(figsize=(12, 6))
ts.plot(label='Original')
ts.iloc[outliers_stl].plot(style='ro', label='Outliers')
plt.legend()
plt.title("STL分解异常检测")
plt.show()
```



## 异常值处理方法详解



### 1. 删除法
- 适用场景：明确噪声数据且占比极低时。
- 风险：可能删除重要信息，尤其在样本量小时。

```python
# 删除异常值
df_cleaned = df.drop(outliers_z)
```



### 2. 转换法
- 对数变换：压缩大值范围，适用于右偏分布。
- Box-Cox变换：寻找最优λ使数据接近正态分布：
 $$
  y(\lambda) = \begin{cases}
  \frac{y^\lambda - 1}{\lambda} & \lambda \neq 0 \\
  \ln(y) & \lambda = 0
  \end{cases}
 $$

```python
from scipy.stats import boxcox

# Box-Cox变换
df['value_transformed'], lambda_ = boxcox(df['value'] + 1)  # +1避免负值
plt.hist(df['value_transformed'], bins=30)
plt.title("Box-Cox变换后分布")
plt.show()
```



### 3. 缩尾处理（Winsorization）
- 定义：将超出分位点的值截断至指定百分位。
 $$
  x_{\text{winsorized}} = \begin{cases}
  Q1 - k \times IQR & \text{if } x < Q1 - k \times IQR \\
  Q3 + k \times IQR & \text{if } x > Q3 + k \times IQR \\
  x & \text{otherwise}
  \end{cases}
 $$

```python
from scipy.stats.mstats import winsorize

# 缩尾处理（上下1%）
df['value_winsorized'] = winsorize(df['value'], limits=[0.01, 0.01])
plt.boxplot(df['value_winsorized'])
plt.title("缩尾处理后箱线图")
plt.show()
```



### 4. 分箱（Binning）
- 方法：将连续变量离散化，用箱内均值或中位数代替异常值。
- 数学公式：  
 $$
  x_{\text{binned}} = \text{median}(Bin_j) \quad \text{若 } x \in Bin_j
 $$

```python
# 等宽分箱
df['value_binned'] = pd.cut(df['value'], bins=5, labels=False)
df['value_binned_median'] = df.groupby('value_binned')['value'].transform('median')
```



### 5. 模型鲁棒化
- 使用鲁棒模型：如RANSAC回归、Huber回归、分位数回归。
- 数学原理（Huber损失）：
 $$
  L_\delta(a) = \begin{cases}
  \frac{1}{2}a^2 & \text{for } |a| \leq \delta \\
  \delta(|a| - \frac{1}{2}\delta) & \text{otherwise}
  \end{cases}
 $$

```python
from sklearn.linear_model import HuberRegressor

# Huber回归处理异常值
X = df.index.values.reshape(-1, 1)
y = df['value']
model = HuberRegressor().fit(X, y)
df['value_huber'] = model.predict(X)
```



## 总结与选择策略

| 方法         | 适用场景                     | 优点                  | 缺点                     |
|------------------|---------------------------------|--------------------------|-----------------------------|
| 删除法           | 明确噪声且样本充足              | 简单直接                 | 信息损失，可能引入偏差       |
| 转换法           | 右偏分布或异方差数据            | 改善分布形态             | 不保留原始尺度               |
| 缩尾处理         | 保留数据分布，避免极端值影响    | 保持数据范围             | 丢失尾部信息                 |
| 分箱             | 非线性关系，粗粒度分析          | 降低噪声敏感性           | 引入信息损失                 |
| 鲁棒模型         | 建模阶段需抵抗异常值干扰        | 直接处理，无需预处理      | 模型复杂度增加               |

实践建议：
1. 多方法交叉验证：结合统计检验与领域知识确认异常值。
2. 可视化辅助决策：通过散点图、箱线图、残差图观察异常影响。
3. 处理前后对比：评估处理对模型性能（如MAE、R²）的影响。

```python
# 处理前后回归效果对比
from sklearn.metrics import mean_absolute_error

# 原始数据
model_orig = HuberRegressor().fit(X, y)
y_pred_orig = model_orig.predict(X)
mae_orig = mean_absolute_error(y, y_pred_orig)

# 缩尾处理后数据
model_winsor = HuberRegressor().fit(X, df['value_winsorized'])
y_pred_winsor = model_winsor.predict(X)
mae_winsor = mean_absolute_error(df['value_winsorized'], y_pred_winsor)

print(f"原始数据MAE: {mae_orig:.2f}, 缩尾处理后MAE: {mae_winsor:.2f}")
```

