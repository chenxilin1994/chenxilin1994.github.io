
# 数据重采样



## 1. 核心概念与问题背景

### 1.1 类别不平衡问题
- 定义：目标变量类别分布显著不均，如欺诈检测中正常交易占99%，欺诈仅1%。
- 危害：
  - 模型偏向多数类，少数类识别率低
  - 评估指标失真（如准确率陷阱）

### 1.2 重采样目的
- 平衡类别分布：通过过采样（增加少数类）或欠采样（减少多数类）调整数据分布
- 改善模型训练：提升对少数类的关注度，优化决策边界



## 2. 过采样技术

### 2.1 随机过采样

#### 数学原理
- 基本方法：随机复制少数类样本直至类别平衡
- 公式表达：
  $$
  X_{\text{new}} = X_{\text{minority}} \cup \{x_i | x_i \sim \text{Uniform}(X_{\text{minority}}), i=1,...,N\}
  $$
  其中 $N = |X_{\text{majority}}| - |X_{\text{minority}}|$

#### Python实现
```python
from imblearn.over_sampling import RandomOverSampler

X, y = load_imbalanced_data()  # 假设已加载数据
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print(f"过采样后类别分布: {pd.Series(y_resampled).value_counts()}")
```

### 2.2 SMOTE（合成少数类过采样）

#### 理论推导
1. 选择样本：对每个少数类样本$x_i$，找到k近邻（通常k=5）
2. 线性插值：随机选择邻域样本$x_{zi}$，生成新样本：
   $$
   x_{\text{new}} = x_i + \lambda (x_{zi} - x_i)
   $$
   其中 $\lambda \sim U(0,1)$

#### 算法复杂度
- 时间复杂度：$O(n_{\text{minority}} \times k)$
- 空间复杂度：$O(n_{\text{minority}} \times d)$ （d为特征维度）

#### Python实现
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5, random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# 可视化生成样本
plt.scatter(X_smote[:,0], X_smote[:,1], c=y_smote, alpha=0.5)
plt.title("SMOTE生成样本分布")
plt.show()
```

### 2.3 ADASYN（自适应合成采样）

#### 数学原理
1. 计算需要生成的样本数：
   $$
   G = |X_{\text{majority}}| - |X_{\text{minority}}|
   $$
2. 计算每个少数类样本的生成权重：
   $$
   \Gamma_i = \frac{\text{邻近多数类样本数}}{\sum \text{邻近多数类样本数}}
   $$
3. 按比例生成样本：每个$x_i$生成$g_i = \Gamma_i \times G$个样本

#### Python实现
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(n_neighbors=5, random_state=42)
X_ada, y_ada = adasyn.fit_resample(X, y)
```



## 3. 欠采样技术

### 3.1 随机欠采样

#### 数学原理
- 简单随机抽样：从多数类中随机删除样本直至类别平衡
  $$
  X_{\text{majority}}' = \text{Subset}(X_{\text{majority}}, m) \quad \text{其中 } m = |X_{\text{minority}}|
  $$

#### Python实现
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
```

### 3.2 Tomek Links

#### 理论推导
- 定义：若两个不同类样本互为最近邻，则构成Tomek Link
- 删除策略：移除多数类样本以扩大类别间隔

#### 算法流程
1. 找到所有Tomek Links对 $(x_i, x_j)$
2. 删除多数类样本 $x_j$

#### Python实现
```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X, y)
```

### 3.3 NearMiss

#### 版本区别
- NearMiss-1：选择与少数类样本平均距离最小的多数类样本
- NearMiss-2：选择与少数类样本平均距离最大的多数类样本
- NearMiss-3：为每个少数类样本保留最近邻的多数类样本

#### 数学公式（NearMiss-1）
$$
\text{Score}(x_j) = \frac{1}{|X_{\text{minority}}|} \sum_{x_i \in X_{\text{minority}}} \|x_j - x_i\|_2
$$
选择得分最低的$m$个多数类样本

#### Python实现
```python
from imblearn.under_sampling import NearMiss

nm = NearMiss(version=1, n_neighbors=3)
X_nm, y_nm = nm.fit_resample(X, y)
```



## 4. 组合采样技术

### 4.1 SMOTEENN

#### 算法流程
1. SMOTE过采样：生成合成少数类样本
2. ENN欠采样：使用编辑最近邻（Edited Nearest Neighbors）删除噪声样本
   - 对每个样本，若其3近邻中多数类占优，则删除该样本

#### Python实现
```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_se, y_se = smote_enn.fit_resample(X, y)
```

### 4.2 SMOTETomek

#### 算法流程
1. SMOTE过采样：生成合成样本
2. Tomek Links清理：移除边界噪声

#### Python实现
```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_smt, y_smt = smt.fit_resample(X, y)
```



## 5. 高级重采样方法

### 5.1 集成欠采样（EasyEnsemble）

#### 理论原理
- Bootstrap集成：多次对多数类随机欠采样，训练多个基分类器
- 最终预测：通过投票或平均集成各基分类器

#### 数学表达
$$
\hat{y} = \text{mode}\{h_1(x), ..., h_T(x)\} \quad \text{其中 } h_t \text{ 在第t个子集训练}
$$

#### Python实现
```python
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(n_estimators=10, random_state=42)
eec.fit(X_train, y_train)
```

### 5.2 Balanced Random Forest

#### 算法改进
- 每棵树训练时：对多数类进行欠采样，保持类别平衡
- 特征选择：基于基尼重要性或置换重要性

#### Python实现
```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
```



## 6. 时间序列重采样

### 6.1 上采样（插值）
```python
# 创建时间序列数据
date_rng = pd.date_range(start='2023-01-01', end='2023-01-07', freq='D')
ts_data = pd.DataFrame(date_rng, columns=['date'])
ts_data['value'] = np.random.randint(1,100, size=(len(date_rng)))

# 上采样到小时粒度
ts_data.set_index('date', inplace=True)
ts_upsampled = ts_data.resample('H').asfreq()

# 线性插值填充
ts_interpolated = ts_upsampled.interpolate(method='linear')
```

### 6.2 下采样（聚合）
```python
# 下采样到周粒度
ts_downsampled = ts_data.resample('W').mean()
```

## 方法对比与选择策略

| 方法          | 优点                          | 缺点                          | 适用场景                     |
|-------------------|----------------------------------|----------------------------------|---------------------------------|
| 随机过采样        | 实现简单，快速                   | 导致过拟合，引入重复样本          | 小规模数据，初步实验            |
| SMOTE             | 生成多样化样本，缓解过拟合       | 对高维数据效果差，可能产生噪声    | 数值型特征，中等维度数据        |
| ADASYN            | 关注难分类样本                   | 可能放大噪声                      | 边界样本重要的场景              |
| Tomek Links       | 清理边界噪声                     | 无法显著改变类别平衡              | 后处理阶段，配合其他方法使用    |
| EasyEnsemble      | 保持数据多样性，集成鲁棒性       | 计算成本高                        | 大规模数据，资源充足场景        |
| 时间序列插值      | 保持时间连续性                   | 可能引入虚假模式                  | 传感器数据填充等                |

实践建议：
1. 轻度不平衡（1:4）：使用加权损失函数或阈值移动
2. 中度不平衡（1:100）：组合SMOTE与欠采样
3. 极度不平衡（1:1000+）：使用集成方法（如Balanced RF）
4. 时序数据：优先考虑时间感知的过采样（如TimeSeriesSynthesizer）



## 完整Pipeline示例
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

# 定义组合采样策略
resampling_pipeline = make_pipeline(
    SMOTE(sampling_strategy=0.5, random_state=42),
    RandomUnderSampler(sampling_strategy=0.8, random_state=42),
    GradientBoostingClassifier()
)

# 交叉验证评估
from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    resampling_pipeline, 
    X, y, 
    scoring='roc_auc', 
    cv=5
)
print(f"平均AUC: {np.mean(scores):.3f} (±{np.std(scores):.3f})")
```
