# 正则化判别分析
## 正则化判别分析（RDA）算法原理详解

### 一、核心概念
正则化判别分析（Regularized Discriminant Analysis, RDA）是LDA与QDA的混合模型，**通过两个正则化参数动态调节模型复杂度**，核心特点：
- **弹性协方差**：$$\lambda$$参数控制类协方差与全局协方差的混合比例
- **收缩稳定**：$$\gamma$$参数添加对角约束防止矩阵病态
- **自适应学习**：自动适应线性/非线性分类边界
- **双重防御**：同时防御过拟合（高方差）和欠拟合（高偏差）

### 二、算法结构
1. **协方差重建**：
   - 类协方差正则化：
     $$
     \boldsymbol{\Sigma}_c(\lambda) = (1-\lambda)\boldsymbol{\Sigma}_c + \lambda\boldsymbol{\Sigma}_W
     $$
   - 全局特征收缩：
     $$
     \boldsymbol{\Sigma}_c^{reg} = (1-\gamma)\boldsymbol{\Sigma}_c(\lambda) + \gamma\cdot\frac{\text{tr}(\boldsymbol{\Sigma}_c(\lambda))}{p}\mathbf{I}
     $$

2. **判别函数**：
   $$
   \delta_c(\mathbf{x}) = \underbrace{-\frac{1}{2}\ln|\boldsymbol{\Sigma}_c^{reg}|}_{\text{协方差体积项}} \underbrace{-\frac{1}{2}D_c^2(\mathbf{x})}_{\text{马氏距离项}} + \underbrace{\ln\pi_c}_{\text{先验项}}
   $$
   其中 $$D_c^2(\mathbf{x}) = (\mathbf{x}-\boldsymbol{\mu}_c)^T(\boldsymbol{\Sigma}_c^{reg})^{-1}(\mathbf{x}-\boldsymbol{\mu}_c)$$

### 三、关键技术细节
1. **参数动力学**：
   - 当$$\lambda=0, \gamma=0$$时退化为QDA
   - 当$$\lambda=1, \gamma=0$$时退化为LDA
   - 当$$\gamma \to 1$$时协方差矩阵趋于各向同性

2. **优化策略**：
   - 双重网格搜索：$$\lambda \in [0,1]$$与$$\gamma \in [0,1]$$组合遍历
   - 协方差预计算：复用全局协方差$$\boldsymbol{\Sigma}_W$$提升效率

3. **特征缩放**：
   - 对收缩参数$$\gamma$$敏感，需先进行标准化：
     $$
     \tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
     $$

### 四、数学推导
**正则化协方差分解**：
$$
\boldsymbol{\Sigma}_c^{reg} = \underbrace{(1-\gamma)(1-\lambda)}_{\text{类协方差权重}}\boldsymbol{\Sigma}_c + \underbrace{(1-\gamma)\lambda}_{\text{全局协方差权重}}\boldsymbol{\Sigma}_W + \underbrace{\gamma\cdot\text{tr}(\cdot)/p}_{\text{收缩项}}\mathbf{I}
$$

**损失函数视角**：
最小化带正则的负对数似然：
$$
\mathcal{L} = \sum_{c=1}^C \left[ \frac{N_c}{2}\ln|\boldsymbol{\Sigma}_c^{reg}| + \frac{1}{2}\text{tr}((\boldsymbol{\Sigma}_c^{reg})^{-1}\boldsymbol{S}_c) \right] + \text{Reg}(\lambda,\gamma)
$$

---

## Python实践指南（以信用卡欺诈检测为例）

### 一、环境准备
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from scipy.linalg import pinvh
```

### 二、数据准备
```python
# 加载信用卡欺诈数据
data = pd.read_csv('creditcard.csv')
X = data.drop(['Class','Time'], axis=1).values
y = data['Class'].values

# 标准化处理（RobustScaler抗异常值）
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集（分层抽样保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
```

### 三、高效RDA实现
```python
class RegularizedDA:
    def __init__(self, lambda_=0.5, gamma_=0.01):
        self.lambda_ = lambda_  # 混合参数
        self.gamma_ = gamma_    # 收缩参数
        
    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        self.n_classes_ = len(self.classes_)
        
        # 计算全局协方差
        self.means_ = np.array([X[y==c].mean(axis=0) for c in range(self.n_classes_)])
        self.priors_ = np.bincount(y) / n_samples
        
        # 类协方差与全局协方差
        covs, counts = [], []
        for c in range(self.n_classes_):
            Xc = X[y == c]
            counts.append(Xc.shape[0])
            covs.append(np.cov(Xc, rowvar=False, bias=True))
        self.global_cov_ = np.sum([(counts[c]-1)*covs[c] for c in range(self.n_classes_)], axis=0)
        self.global_cov_ /= (n_samples - self.n_classes_)
        
        # 正则化协方差计算
        self.reg_covs_ = []
        for c in range(self.n_classes_):
            # 混合协方差
            mixed_cov = (1-self.lambda_)*covs[c] + self.lambda_*self.global_cov_
            # 收缩正则
            trace = np.trace(mixed_cov)
            shrunk_cov = (1-self.gamma_)*mixed_cov + (self.gamma_*trace/n_features)*np.eye(n_features)
            self.reg_covs_.append(pinvh(shrunk_cov))  # 伪逆避免奇异
            
        return self
    
    def predict_proba(self, X):
        probas = []
        for c in range(self.n_classes_):
            diff = X - self.means_[c]
            mahalanobis = np.einsum('ij,ij->i', diff @ self.reg_covs_[c], diff)
            log_prob = -0.5 * mahalanobis + np.log(self.priors_[c])
            probas.append(log_prob)
        probas = np.exp(probas - np.max(probas, axis=0))
        return (probas / np.sum(probas, axis=0)).T
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
```

### 四、参数调优
```python
# 定义参数网格
param_grid = {
    'lambda_': np.linspace(0, 1, 5),  # 混合参数
    'gamma_': [0, 0.01, 0.1, 0.5]     # 收缩参数
}

# 分层交叉验证（考虑类别不平衡）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 使用AUC作为评估指标
grid = GridSearchCV(RegularizedDA(),
                   param_grid,
                   cv=cv,
                   scoring='roc_auc',
                   n_jobs=-1,
                   verbose=1)
grid.fit(X_train, y_train)

# 输出最佳参数
print(f"最优参数：{grid.best_params_}")
print(f"测试集AUC：{roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]):.3f}")
```

### 五、欺诈检测分析
```python
# 绘制PR曲线
probas = grid.predict_proba(X_test)[:,1]
precision, recall, _ = precision_recall_curve(y_test, probas)

plt.figure(figsize=(10,6))
plt.plot(recall, precision, lw=2, color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (AP={:.3f})'.format(average_precision_score(y_test, probas)))
plt.grid(True)
plt.show()

# 输出高风险交易
fraud_probas = probas[y_test == 1]
print(f"欺诈交易预测概率分布：")
print(pd.Series(fraud_probas).describe(percentiles=[0.5, 0.9, 0.99]))
```

### 六、正则化效应可视化
```python
# 提取协方差矩阵条件数
cond_numbers = []
for lambda_ in param_grid['lambda_']:
    for gamma_ in param_grid['gamma_']:
        model = RegularizedDA(lambda_=lambda_, gamma_=gamma_).fit(X_train, y_train)
        cond_num = np.mean([np.linalg.cond(cov) for cov in model.reg_covs_])
        cond_numbers.append(cond_num)

# 绘制热力图
cond_df = pd.DataFrame({
    'lambda': np.repeat(param_grid['lambda_'], len(param_grid['gamma_'])),
    'gamma': np.tile(param_grid['gamma_'], len(param_grid['lambda_'])),
    'cond': cond_numbers
})
cond_matrix = cond_df.pivot('lambda', 'gamma', 'cond')

plt.figure(figsize=(10,6))
sns.heatmap(cond_matrix, annot=True, fmt=".1e", cmap="viridis_r")
plt.title("协方差矩阵条件数变化")
plt.xlabel("Gamma收缩强度")
plt.ylabel("Lambda混合参数")
plt.show()
```

---

## 数学补充
**收缩参数优化**：
最优收缩系数$$\gamma^*$$可解析求解：
$$
\gamma^* = \frac{\sum_{i=1}^p \text{Var}(\hat{\sigma}_{ii})}{\sum_{i=1}^p \hat{\sigma}_{ii}^2 + \sum_{i \neq j} \hat{\sigma}_{ij}^2}
$$

**马氏距离近似**：
当$$\gamma \to 1$$时：
$$
D_c^2(\mathbf{x}) \approx \frac{p}{\text{tr}(\boldsymbol{\Sigma}_c(\lambda))}\|\mathbf{x}-\boldsymbol{\mu}_c\|^2
$$

---

## 注意事项
1. **类别不平衡处理**：
   - 调整先验概率：$$\pi_c = N_c^{\beta}/\sum_k N_k^{\beta}$$（$$\beta < 1$$降低多数类影响）
   - 过采样少数类：SMOTE或ADASYN

2. **计算优化**：
   - 协方差矩阵存储优化：对称性压缩存储
   - 矩阵求逆并行化：分块矩阵并行计算

3. **高维应对**：
   - 特征预筛选：ANOVA F值或互信息选择Top-K特征
   - 增量特征学习：逐步添加特征监控AUC变化

---

## 扩展应用
1. **在线欺诈检测**：
   ```python
   # 滑动窗口更新
   window_size = 1000
   for i in range(0, len(X), window_size):
       batch_X = X[i:i+window_size]
       batch_y = y[i:i+window_size]
       model.partial_fit(batch_X, batch_y)
       # 实时风险评估
       probas = model.predict_proba(current_transactions)
   ```

2. **多模态数据融合**：
   ```python
   # 混合协方差融合
   def fusion_cov(cov_structured, cov_unstructured, alpha):
       return alpha*cov_structured + (1-alpha)*cov_unstructured
   ```

3. **不确定性量化**：
   ```python
   # 计算预测置信度
   probas = model.predict_proba(X)
   confidence = np.max(probas, axis=1) - np.min(probas, axis=1)
   ```

---

## 与深度学习方法对比
| 特性                | RDA               | 深度神经网络（DNN）       |
|---------------------|-------------------|--------------------------|
| 数据需求            | 小样本高效        | 需要大数据量             |
| 解释性              | 高（概率分解）    | 低（黑箱模型）           |
| 计算资源            | CPU高效           | 需GPU加速               |
| 特征工程            | 需标准化          | 自动特征提取             |
| 实时预测            | 微秒级响应        | 毫秒级响应               |
| 对抗攻击鲁棒性      | 较高              | 较低                     |

---

正则化判别分析通过弹性协方差建模，在金融风控、医疗诊断等**高风险决策场景**展现独特优势。其概率输出特性支持风险评估量化，双重正则化机制保障了小样本下的稳健性。与深度学习相比，RDA在模型透明度和计算效率方面具有优势，但学习复杂非线性模式的能力有限。建议在实际应用中与集成学习（如LightGBM）组成混合模型，兼顾可解释性与预测性能。