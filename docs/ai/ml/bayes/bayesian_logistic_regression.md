# 贝叶斯逻辑回归

## 贝叶斯逻辑回归算法原理详解

### 一、核心概念
贝叶斯逻辑回归（Bayesian Logistic Regression）是传统逻辑回归的贝叶斯扩展，通过引入参数的先验分布，将二分类问题转化为概率推断问题。核心特征：
- **概率建模**：参数 $\boldsymbol{w}$ 服从概率分布，输出为类别概率
- **不确定性量化**：提供预测概率的置信区间
- **正则化内生化**：先验分布自然实现L2正则化
- **非线性决策边界**：通过基函数扩展处理复杂模式

### 二、算法结构
1. **先验层**：
   $$ p(\boldsymbol{w} \mid \alpha) = \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{0}, \alpha^{-1}\boldsymbol{I}) $$
   - $$\alpha$$ 为精度超参数（可设置Gamma超先验）

2. **似然层**：
   $$ p(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}) = \text{Bernoulli}(y_n \mid \sigma(\boldsymbol{w}^\top \phi(\boldsymbol{x}_n))) $$
   - $$\sigma(a) = 1/(1+e^{-a})$$ 为sigmoid函数
   - $$\phi(\boldsymbol{x})$$ 为基函数扩展

3. **后验层**：
   $$ p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) \propto p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{w}) p(\boldsymbol{w}) $$
   - 无解析解，需采用近似推断

### 三、关键技术细节
1. **推断方法**：
   - MCMC采样（NUTS/HMC）：
     $$ \boldsymbol{w}^{(t+1)} \sim p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) $$
   - 变分推断（平均场近似）：
     $$ q(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) $$

2. **预测分布**：
   $$ p(y^*=1 \mid \boldsymbol{x}^*, \boldsymbol{X}, \boldsymbol{y}) = \int \sigma(\boldsymbol{w}^\top \phi(\boldsymbol{x}^*)) p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) d\boldsymbol{w} $$
   - 通过后验采样近似计算

3. **超参数优化**：
   - 经验贝叶斯：最大化边际似然
   $$p(\boldsymbol{y} \mid \boldsymbol{X}, \alpha}) = \int p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{w}) p(\boldsymbol{w} \mid \alpha}) d\boldsymbol{w}$$

### 四、数学表达
**后验分布**：
$$
p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) \propto \exp\left( -\frac{\alpha}{2} \boldsymbol{w}^\top \boldsymbol{w} \right) \prod_{n=1}^N \sigma(\boldsymbol{w}^\top \phi(\boldsymbol{x}_n)))^{y_n} (1-\sigma(\boldsymbol{w}^\top \phi(\boldsymbol{x}_n)))^{1-y_n}
$$

**拉普拉斯近似**：
1. 找到后验众数 $$\boldsymbol{w}_{\text{MAP}}$$
2. 计算Hessian矩阵：
   $$ \boldsymbol{H} = -\nabla^2 \log p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) \big|_{\boldsymbol{w}=\boldsymbol{w}_{\text{MAP}}} $$
3. 近似后验：
   $$ q(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{w}_{\text{MAP}}, \boldsymbol{H}^{-1}) $$

---

## Python实践指南（使用PyMC3）

### 一、环境准备
```python
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

# 生成模拟数据
X, y = make_classification(
    n_samples=200, 
    n_features=2, 
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)
```

### 二、模型定义
```python
with pm.Model() as bayesian_logreg:
    # 超先验
    alpha = pm.Gamma('alpha', alpha=1, beta=0.1)
    
    # 权重先验
    w = pm.Normal('w', mu=0, sigma=1/np.sqrt(alpha), shape=X.shape[1]+1)
    
    # 线性组合（含偏置）
    linear = pm.math.dot(pm.math.concatenate([np.ones((X.shape[0],1)), X]), w)
    
    # Sigmoid变换
    p = pm.Deterministic('p', 1 / (1 + pm.math.exp(-linear)))
    
    # 似然
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
    
    # NUTS采样
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9)
```

### 三、后验分析
```python
# 参数轨迹可视化
pm.plot_trace(trace, var_names=['w', 'alpha'], compact=True)
plt.show()

# 后验统计摘要
print(pm.summary(trace, var_names=['w', 'alpha'], credible_interval=0.95))
```

### 四、预测评估
```python
# 计算预测概率
with bayesian_logreg:
    post_pred = pm.sample_posterior_predictive(trace, var_names=['p'])

# AUC评估
y_prob = post_pred['p'].mean(axis=0)
print(f"AUC: {roc_auc_score(y, y_prob):.3f}")

# 决策边界可视化
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
X_grid = np.c_[xx.ravel(), yy.ravel()]

with bayesian_logreg:
    # 禁用观测值
    model = pm.modelcontext(bayesian_logreg)
    model.y_obs.missing_values = True  
    grid_pred = pm.sample_posterior_predictive(
        trace, 
        var_names=['p'],
        samples=500,
        predictions=True,
        data={'X': X_grid}
    )

# 绘制置信区域
plt.figure(figsize=(10, 6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k')
contour = plt.contourf(xx, yy, grid_pred['p'].mean(0).reshape(xx.shape), 
                      levels=20, cmap='coolwarm', alpha=0.6)
plt.colorbar(contour)
plt.title("Bayesian Logistic Regression Decision Boundary")
plt.show()
```

### 五、稀疏化实现
```python
# 使用Horseshoe先验进行特征选择
with pm.Model() as sparse_logreg:
    tau = pm.HalfCauchy('tau', beta=1)
    lam = pm.HalfCauchy('lam', beta=1, shape=X.shape[1])
    w = pm.Normal('w', mu=0, sigma=tau*lam, shape=X.shape[1])
    
    p = pm.Deterministic('p', 1/(1+pm.math.exp(-pm.math.dot(X, w)))
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
    
    trace_sparse = pm.sample(2000, tune=1000)
```


## 数学补充
**Probit替代模型**：
$$ p(y=1 \mid \boldsymbol{x}) = \Phi(\boldsymbol{w}^\top \phi(\boldsymbol{x})) $$
其中 $$\Phi$$ 为标准正态CDF，可提高采样效率

**多项式逻辑回归扩展**：
$$
p(y=k \mid \boldsymbol{x}) = \frac{\exp(\boldsymbol{w}_k^\top \phi(\boldsymbol{x}))}{\sum_{j=1}^K \exp(\boldsymbol{w}_j^\top \phi(\boldsymbol{x}))}
$$

**贝叶斯模型平均**：
$$ p(y^* \mid \boldsymbol{x}^*) = \sum_{m=1}^M p(y^* \mid \boldsymbol{x}^*, \boldsymbol{w}^{(m)}) p(m \mid \boldsymbol{X}, \boldsymbol{y}) $$


## 性能优化技巧
1. **数据标准化**：
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **ADVI加速**：
   ```python
   with model:
       approx = pm.fit(n=50000, method='advi')
       trace = approx.sample(1000)
   ```

3. **稀疏矩阵优化**：
   ```python
   import scipy.sparse
   X_sparse = scipy.sparse.csr_matrix(X)
   ```


## 注意事项
1. **共线性问题**：
   - 导致后验分布扁平化
   - 解决方案：正则化先验或PCA降维

2. **类别不平衡**：
   - 调整先验偏置项：
     ```python
     w0 = pm.Normal('w0', mu=np.log(pos_class_ratio), sigma=10)
     ```

3. **收敛诊断**：
   ```python
   pm.energyplot(trace)
   pm.gelman_rubin(trace)
   ```


## 扩展应用
1. **层次逻辑回归**：
   ```python
   with pm.Model() as hierarchical_logreg:
       # 组间共享超先验
       mu_w = pm.Normal('mu_w', 0, 10)
       sigma_w = pm.HalfNormal('sigma_w', 10)
       
       # 各组独立参数
       w = pm.Normal('w', mu=mu_w, sigma=sigma_w, shape=(n_groups, n_features))
       
       # 计算各组概率
       p = 1/(1 + pm.math.exp(-pm.math.dot(X, w[group_ids].T)))
   ```

2. **贝叶斯神经网络**：
   ```python
   with pm.Model() as bayesian_nn:
       # 隐含层权重
       w1 = pm.Normal('w1', 0, 1, shape=(n_input, n_hidden))
       b1 = pm.Normal('b1', 0, 1, shape=n_hidden)
       h = pm.math.tanh(pm.math.dot(X, w1) + b1)
       
       # 输出层
       w2 = pm.Normal('w2', 0, 1, shape=(n_hidden, 1))
       p = 1/(1 + pm.math.exp(-pm.math.dot(h, w2)))
   ```

3. **在线学习**：
   ```python
   # 序贯更新后验
   for batch in dataloader:
       with model:
           pm.set_data({'X': batch_X, 'y': batch_y})
           step = pm.NUTS()
           trace = pm.sample(500, step=step, start=last_params)
           last_params = trace[-1]
   ```


## 与传统逻辑回归对比
| 特性                | 传统逻辑回归         | 贝叶斯逻辑回归           |
|---------------------|---------------------|--------------------------|
| 参数估计            | MLE点估计           | 后验分布                 |
| 正则化              | 需显式添加L2惩罚    | 通过先验自然实现         |
| 不确定性估计        | 仅参数标准误        | 完整预测分布             |
| 计算复杂度          | O(N³)               | 更高（依赖采样次数）     |
| 小样本表现          | 易过拟合            | 通过先验控制过拟合       |
| 特征选择            | L1正则化            | 稀疏先验（如Horseshoe）  |

---

贝叶斯逻辑回归通过概率框架为分类问题提供了更丰富的推断能力，特别适用于需要概率解释、不确定性量化和稳健预测的场景。结合现代概率编程工具，可扩展处理高维数据、层次结构和复杂模式，在医疗诊断、信用评分等领域展现独特价值。其计算复杂度可通过变分推断和GPU加速有效缓解，是贝叶斯机器学习的重要基础模型。