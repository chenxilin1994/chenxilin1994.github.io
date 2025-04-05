
# 特征选择技术理论详解



## 1. 过滤式方法（Filter Methods）

### 1.1 单变量统计检验

#### 数学原理与推导
1. Pearson相关系数（线性关系检验）：
   - 定义：衡量两个连续变量间的线性相关程度。
   - 公式：
     $$
     r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}
     $$
   - 假设检验：
     $$
     t = r \sqrt{\frac{n-2}{1-r^2}} \sim t(n-2)
     $$
     当 $|t| > t_{\alpha/2}(n-2)$ 时拒绝原假设（无相关性）。

2. 卡方检验（类别独立性检验）：
   - 定义：检验分类特征与目标变量的独立性。
   - 统计量计算：
     $$
     \chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
     $$
     其中 $E_{ij} = \frac{\text{row}_i \times \text{col}_j}{n}$，自由度 $df = (r-1)(c-1)$。

3. ANOVA F检验（组间差异检验）：
   - 定义：比较连续特征在不同类别目标中的均值差异。
   - 公式分解：
     $$
     \text{总变异} (SST) = \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x})^2
     $$
     $$
     \text{组间变异} (SSB) = \sum_{i=1}^k n_i (\bar{x}_i - \bar{x})^2
     $$
     $$
     F = \frac{SSB/(k-1)}{SSE/(n-k)} \sim F(k-1, n-k)
     $$

4. 互信息（Mutual Information）：
   - 信息论基础：衡量特征与目标变量间的信息共享量。
   - 离散变量公式：
     $$
     I(X;Y) = \sum_{x \in X} \sum_{y \in Y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}
     $$
   - 连续变量估计：基于k近邻的KSG估计器。

#### 方法论特性
- 优点：计算高效，适合高维数据初筛。
- 缺点：忽略特征交互，无法识别组合效应。
- 适用场景：数据预处理阶段快速剔除无关特征。

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import matplotlib.pyplot as plt

# 生成模拟数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)
df = pd.DataFrame(X, columns=[f"F{i}" for i in range(1,21)])
df['target'] = y

# 卡方检验（适用于分类特征）
selector_chi = SelectKBest(chi2, k=5)
X_chi = selector_chi.fit_transform(df.iloc[:,:10], y)  # 假设前10个是分类特征
chi_scores = selector_chi.scores_
print("卡方检验特征得分:", chi_scores)

# ANOVA F检验（连续特征）
selector_f = SelectKBest(f_classif, k=5)
X_f = selector_f.fit_transform(df.iloc[:,10:20], y)
f_scores = selector_f.scores_
print("\nANOVA F值:", f_scores)

# 互信息（混合类型）
mi_scores = mutual_info_classif(df.iloc[:,:20], y, discrete_features=[0,1,2,3,4])
print("\n互信息得分:", mi_scores)

# 可视化特征得分
plt.figure(figsize=(15,5))
plt.bar(range(20), np.concatenate([chi_scores, f_scores]))
plt.title("单变量特征得分")
plt.xlabel("Feature Index")
plt.ylabel("Score")
plt.show()
```

### 1.2 方差阈值法

#### 数学原理
- 方差定义：
  $$
  \text{Var}(X_j) = \frac{1}{n} \sum_{i=1}^n (x_{ij} - \mu_j)^2
  $$
- 筛选逻辑：若 $\text{Var}(X_j) < \theta$，认为特征 $X_j$ 信息量不足。
- 标准化需求：需先进行归一化（如Z-Score），避免量纲影响。

#### 理论局限
- 假设缺陷：低方差≠低信息量（如二值特征方差可能小但重要）。
- 改进方案：结合目标变量相关性的加权方差阈值。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:,:20])

# 设置方差阈值（保留方差>0.8的特征）
selector = VarianceThreshold(threshold=0.8)
X_selected = selector.fit_transform(X_scaled)
print("原始特征数:", X_scaled.shape[1])
print("筛选后特征数:", X_selected.shape[1])

# 获取保留特征的索引
retained_features = np.where(selector.variances_ > 0.8)[0]
print("保留特征索引:", retained_features)
```

## 2. 包裹式方法（Wrapper Methods）

### 2.1 递归特征消除（RFE）

#### 算法流程
1. 初始化：特征全集 $S = \{1, 2, ..., p\}$。
2. 模型训练：在 $S$ 上训练模型，获取权重向量 $w$。
3. 特征排序：按 $|w_j|$ 升序排列特征。
4. 特征剔除：移除排名后 $k$ 个特征，更新 $S = S \setminus \{j_1, ..., j_k\}$。
5. 迭代终止：当 $|S| = m$（预设特征数）时停止。

#### 数学解释
- 权重敏感性：假设模型权重反映特征重要性（线性模型适用，非线性模型需置换重要性）。
- 收敛性证明：在凸损失函数下，RFE可收敛至局部最优特征子集。

#### 复杂度分析
- 时间复杂度：$O(p \times T_{\text{model}})$，其中 $T_{\text{model}}$ 为单次模型训练时间。
- 空间复杂度：$O(p)$ 存储特征权重。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# 初始化逻辑回归模型
model = LogisticRegression(max_iter=1000, random_state=42)

# 递归特征消除选择10个特征
selector = RFE(model, n_features_to_select=10, step=2)
selector.fit(df.iloc[:,:20], y)

# 输出选择结果
selected_features = df.columns[selector.support_]
print("RFE选择特征:", selected_features.tolist())

# 可视化特征排名
plt.figure(figsize=(10,6))
plt.barh(range(20), selector.ranking_)
plt.yticks(range(20), df.columns[:20])
plt.title("RFE特征排名（1表示被选择）")
plt.show()
```

### 2.2 顺序特征选择（SFS）

#### 前向选择数学描述
- 目标函数：最大化模型性能指标 $J(S)$。
- 贪心策略：
  $$
  S_{i+1} = S_i \cup \left\{ \arg \max_{j \notin S_i} J(S_i \cup \{j\}) \right\}
  $$
- 停止条件：达到预设特征数或性能提升不显著（早停）。

#### 后向消除数学描述
- 剔除策略：
  $$
  S_{i+1} = S_i \setminus \left\{ \arg \min_{j \in S_i} J(S_i \setminus \{j\}) \right\}
  $$

#### 理论局限
- 组合爆炸：特征数为 $p$ 时，全搜索空间为 $2^p$，贪心策略可能陷入局部最优。
- 改进方向：引入Beam Search或遗传算法扩大搜索范围。

```python
from mlxtend.feature_selection import SequentialFeatureSelector

# 前向选择（耗时操作，建议设置合理参数）
sfs = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000),
    k_features=10,
    forward=True,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
sfs.fit(df.iloc[:,:20], y)

print("前向选择特征:", sfs.k_feature_names_)
```

## 3. 嵌入式方法（Embedded Methods）

### 3.1 L1正则化（LASSO）

#### 优化问题形式化
$$
\min_{w \in \mathbb{R}^p} \left( \frac{1}{2n} \| y - Xw \|_2^2 + \alpha \| w \|_1 \right)
$$
- KKT条件：
  $$
  \frac{1}{n} X_j^T (y - Xw) = \alpha \cdot \text{sign}(w_j) \quad \text{if } w_j \neq 0
  $$
  $$
  \left| \frac{1}{n} X_j^T (y - Xw) \right| \leq \alpha \quad \text{if } w_j = 0
  $$

#### 稀疏性证明
- 几何解释：L1正则化的等高线与平方误差的等高线在角点相交，导致解稀疏。
- 概率视角：等价于给权重施加拉普拉斯先验 $p(w_j) = \frac{\alpha}{2} e^{-\alpha |w_j|}$。

#### 模型选择
- 正则化路径：通过调节 $\alpha$ 控制稀疏度，交叉验证选择最优 $\alpha$。

```python
from sklearn.linear_model import LassoCV

# LASSO路径分析
alphas = np.logspace(-4, 0, 50)
lasso = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso.fit(df.iloc[:,:20], y)

# 可视化正则化路径
plt.figure(figsize=(10,6))
plt.plot(alphas, lasso.mse_path_.mean(axis=-1))
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title("LASSO正则化路径")

# 显示非零系数
nonzero_coef = np.where(lasso.coef_ != 0)[0]
print("\nLASSO选择特征:", df.columns[nonzero_coef].tolist())
```

### 3.2 树模型特征重要性

#### 基尼重要性计算
- 节点分裂增益：
  $$
  \Delta G = G_{\text{parent}} - \left( \frac{N_{\text{left}}}{N} G_{\text{left}} + \frac{N_{\text{right}}}{N} G_{\text{right}} \right)
  $$
  其中基尼系数 $G = 1 - \sum_{k=1}^K p_k^2$。
- 特征重要性：在所有树中，特征 $j$ 被选为分裂节点的平均增益。

#### 置换重要性
- 定义：
  $$
  \text{Importance}_j = \frac{1}{B} \sum_{b=1}^B (L(D_b^{(j)}) - L(D))
  $$
  其中 $D_b^{(j)}$ 为第 $j$ 个特征被置换后的数据集，$L$ 为损失函数。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df.iloc[:,:20], y)

# 基尼重要性
plt.figure(figsize=(12,6))
plt.barh(df.columns[:20], rf.feature_importances_)
plt.title("基尼重要性")
plt.show()

# 置换重要性
result = permutation_importance(rf, df.iloc[:,:20], y, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(12,6))
plt.boxplot(result.importances[sorted_idx].T, vert=False)
plt.yticks(range(20), df.columns[sorted_idx])
plt.title("置换重要性")
plt.show()
```

## 4. 混合方法（Hybrid Methods）

### 4.1 稳定性选择

#### 理论框架
- 子采样机制：从原始数据中抽取 $B$ 个子集 $D_1, ..., D_B$。
- 选择频率：
  $$
  \hat{\pi}_j = \frac{1}{B} \sum_{b=1}^B I(j \in S_b)
  $$
  其中 $S_b$ 为第 $b$ 次选择的特征子集。
- 阈值决策：保留 $\hat{\pi}_j \geq \pi_{\text{thr}}$ 的特征。

#### 概率解释
- 假设检验：若特征 $j$ 无关，则 $\hat{\pi}_j$ 服从二项分布 $Bin(B, p_0)$，其中 $p_0$ 为随机选择概率。

```python
from sklearn.linear_model import Lasso
from sklearn.utils import resample

# 稳定性选择参数
n_iterations = 50
alpha = 0.01
selection_counts = np.zeros(20)

for i in range(n_iterations):
    # 子采样
    X_sample, y_sample = resample(df.iloc[:,:20], y, random_state=i)
    
    # LASSO拟合
    lasso = Lasso(alpha=alpha, random_state=i)
    lasso.fit(X_sample, y_sample)
    
    # 记录非零系数
    selection_counts += (lasso.coef_ != 0).astype(int)

# 计算稳定性分数
stability_scores = selection_counts / n_iterations
stable_features = np.where(stability_scores > 0.8)[0]
print("稳定性选择特征:", df.columns[stable_features].tolist())
```

### 4.2 特征聚类

#### 相似性度量
- 相关系数：
  $$
  \rho_{jk} = \frac{\text{Cov}(X_j, X_k)}{\sigma_{X_j} \sigma_{X_k}}
  $$
- 互信息：
  $$
  I(X_j; X_k) = \sum_{x_j, x_k} P(x_j, x_k) \log \frac{P(x_j, x_k)}{P(x_j)P(x_k)}
  $$

#### 聚类算法
- 层次聚类：基于距离矩阵构建树状结构，通过切割高度确定簇数。
- 图论方法：将特征视为节点，相似性为边权重，寻找最大团或最小割集。

```python
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

# 计算相关系数矩阵
corr_matrix = df.iloc[:,:20].corr().abs()

# 层次聚类
plt.figure(figsize=(15,10))
dendrogram = hierarchy.dendrogram(
    hierarchy.linkage(corr_matrix, method='ward'),
    labels=df.columns[:20],
    orientation='right'
)
plt.title("特征聚类树状图")

# 自动选择簇数（此处以3簇为例）
cluster = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
cluster_labels = cluster.fit_predict(1 - corr_matrix)

# 从每个簇选择代表特征
selected_features = []
for cluster_id in np.unique(cluster_labels):
    cluster_features = df.columns[:20][cluster_labels == cluster_id]
    # 选择与目标变量相关性最强的特征
    correlations = df[cluster_features].corrwith(df['target']).abs()
    selected = correlations.idxmax()
    selected_features.append(selected)
    
print("\n聚类选择特征:", selected_features)

```

## 5. 高级方法

### 5.1 自动机器学习（AutoML）

#### 贝叶斯优化
- 代理模型：使用高斯过程建模特征子集与模型性能的关系。
- 采集函数：通过期望改进（EI）或上置信界（UCB）指导搜索。

$$
\alpha_{\text{EI}}(S) = \mathbb{E} [\max(f(S) - f(S^+), 0)]
$$

#### 进化算法
- 染色体编码：每个个体表示一个特征子集（二进制编码）。
- 适应度函数：交叉验证得分与特征数惩罚项的加权和：
  $$
  \text{Fitness}(S) = \text{AUC}(S) - \lambda |S|
  $$

```python
# 需要安装遗传算法库：pip install genetic-algorithm
from genetic_algorithm import GeneticAlgorithm

# 定义适应度函数
def fitness_function(feature_subset):
    X_subset = df.iloc[:, list(feature_subset)]
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X_subset, y, cv=3, scoring='accuracy')
    return np.mean(scores) - 0.01*len(feature_subset)

# 运行遗传算法
ga = GeneticAlgorithm(
    gene_length=20,
    population_size=50,
    generations=30,
    fitness_function=fitness_function,
    crossover_prob=0.8,
    mutation_prob=0.1
)
best_solution = ga.run()

print("遗传算法选择特征:", df.columns[best_solution['genes']].tolist())
```

### 5.2 深度学习特征选择

#### 注意力机制
- 自注意力权重：
  $$
  \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^p \exp(e_{ik})}
  $$
  其中 $e_{ij} = \text{MLP}(x_i, x_j)$。
- 特征重要性：通过 $\sum_j \alpha_{ij}$ 聚合权重。

#### 可解释性方法
- SHAP值：基于合作博弈论的边际贡献计算：
  $$
  \phi_j = \sum_{S \subseteq P \setminus \{j\}} \frac{|S|!(p - |S| - 1)!}{p!} [f(S \cup \{j\}) - f(S)]
  $$
  其中 $P$ 为全体特征集合。

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Multiply
from tensorflow.keras.models import Model
import shap

# 构建注意力机制模型
inputs = Input(shape=(20,))
attention = Dense(20, activation='softmax')(inputs)
weighted_inputs = Multiply()([inputs, attention])
x = Dense(32, activation='relu')(weighted_inputs)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')

# 训练模型
model.fit(df.iloc[:,:20], y, epochs=50, batch_size=32, verbose=0)

# 提取注意力权重
attention_weights = model.layers[1].get_weights()[0]
plt.figure(figsize=(12,6))
plt.barh(range(20), attention_weights.mean(axis=1))
plt.yticks(range(20), df.columns[:20])
plt.title("注意力权重")

# SHAP分析
explainer = shap.DeepExplainer(model, df.iloc[:100,:20].values)
shap_values = explainer.shap_values(df.iloc[:100,:20].values)
shap.summary_plot(shap_values, df.columns[:20])
```

## 方法比较与选择策略

| 维度          | 过滤式                  | 包裹式                  | 嵌入式                  | 混合式                  |
|-------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| 计算效率       | 高                         | 低                         | 中                         | 中                         |
| 模型依赖性     | 无                         | 强                         | 强                         | 中                         |
| 特征交互处理   | 无                         | 有                         | 部分                       | 有                         |
| 过拟合风险     | 低                         | 高                         | 中                         | 中                         |
| 适用数据规模   | 大规模                     | 小规模                     | 中大规模                   | 中大规模                   |

理论指导原则：
1. 高维小样本：过滤式（方差阈值+互信息）→ 嵌入式（LASSO）。
2. 低维强交互：包裹式（RFE/SFS）→ 树模型（GBDT+SHAP）。
3. 非结构化数据：深度学习（注意力机制）→ 稳定性选择。
4. 在线学习系统：嵌入式（在线LASSO）→ 增量式特征聚类。



## 总结
特征选择是机器学习流程中连接数据预处理与模型构建的关键桥梁。从统计假设检验到深度学习注意力机制，不同方法在不同数据场景下展现独特优势。理论选择需结合数据分布、模型结构及计算资源，实践应用中常采用多层次筛选策略（过滤初筛→嵌入式精炼→包裹式优化）。未来趋势将更注重自动化（AutoML）与可解释性（SHAP）的结合，以实现高效透明的特征工程。