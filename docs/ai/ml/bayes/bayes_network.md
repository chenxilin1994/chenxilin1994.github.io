# 贝叶斯网络

## 贝叶斯网络算法原理详解

### 一、核心概念
贝叶斯网络（Bayesian Network）是一种概率图模型，通过有向无环图（DAG）表示变量间的条件依赖关系。核心特征：
- **结构化表示**：节点表示随机变量，边表示条件依赖
- **局部马尔可夫性**：节点在给定父节点时条件独立于非后代节点
- **概率传播机制**：支持因果推理和证据传播
- **联合概率分解**：将高维联合分布分解为条件概率乘积

### 二、算法结构
1. **图结构层**：
   - 节点：随机变量（离散/连续）
   - 边：因果关系（父节点→子节点）
   - 网络结构需满足DAG约束

2. **参数层**：
   - 条件概率表（CPT）：$$ P(X_i \mid \text{Pa}(X_i)) $$
   - 先验概率：无父节点的节点概率分布
   - 联合概率分解：$$ P(X_1,...,X_n) = \prod_{i=1}^n P(X_i \mid \text{Pa}(X_i)) $$

3. **推理层**：
   - 精确推理：变量消除、联结树算法
   - 近似推理：MCMC采样、变分推断

### 三、关键技术细节
1. **结构学习**：
   - 基于约束的方法（PC算法）：
     - 使用统计检验发现条件独立性
     - 时间复杂度：$$ O(n^k) $$（n为节点数，k为最大父节点数）
   - 基于评分的方法（BIC评分）：
     $$ \text{BIC}(G) = \log P(D \mid \theta_G) - \frac{d}{2} \log N $$
     其中d为参数数量，N为样本量

2. **参数学习**：
   - 最大似然估计：
     $$ \hat{\theta}_{ijk} = \frac{N_{ijk}}{N_{ij}} $$
     （$N_{ijk}$为满足配置的样本数）
   - 贝叶斯估计（狄利克雷先验）：
     $$ \hat{\theta}_{ijk} = \frac{N_{ijk} + \alpha_{ijk}}{N_{ij} + \alpha_{ij}} $$

3. **推理算法**：
   - 变量消除法：
     $$ P(Q \mid E=e) = \frac{\sum_{H} \prod_{i} P(X_i \mid \text{Pa}(X_i))}{\sum_{Q,H} \prod_{i} P(X_i \mid \text{Pa}(X_i))} $$
   - MCMC采样（Gibbs Sampling）：
     $$ X_i^{(t+1)} \sim P(X_i \mid \text{MB}(X_i)) $$
     其中MB为马尔可夫毯

### 四、数学表达
设网络包含变量 $$ \{X_1,...,X_n\} $$，其联合概率分布为：
$$
P(X_1,...,X_n) = \prod_{i=1}^n P(X_i \mid \text{Pa}(X_i))
$$

**d-分离准则**：
- 路径 $A \rightarrow B \leftarrow C$ 在给定B时开放
- 路径 $A \rightarrow B \rightarrow C$ 在未观测B时开放

**条件独立性**：
$$
X \perp\!\!\!\perp Y \mid Z \iff P(X,Y \mid Z) = P(X \mid Z)P(Y \mid Z)
$$

## Python实践指南（以医疗诊断为例）

### 一、环境准备
```python
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
```

### 二、网络构建
```python
# 定义网络结构
model = BayesianNetwork([
    ('Smoking', 'LungCancer'),
    ('AirPollution', 'LungCancer'),
    ('LungCancer', 'Cough'),
    ('LungCancer', 'XRay'),
    ('Fatigue', 'LungCancer')
])

# 可视化网络
nx.draw(model, with_labels=True, node_size=2000)
plt.show()
```

### 三、参数学习
```python
# 生成模拟数据
data = pd.DataFrame({
    'Smoking': np.random.choice(['Yes','No'], 1000, p=[0.3,0.7]),
    'AirPollution': np.random.choice(['High','Low'], 1000, p=[0.4,0.6]),
    'LungCancer': np.random.choice(['Present','Absent'], 1000),
    'Cough': np.random.choice(['Yes','No'], 1000),
    'XRay': np.random.choice(['Abnormal','Normal'], 1000),
    'Fatigue': np.random.choice(['Yes','No'], 1000)
})

# 参数估计
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 查看CPT
print(model.get_cpds('LungCancer'))
```

### 四、概率推理
```python
# 创建推理引擎
infer = VariableElimination(model)

# 诊断推理（给定症状求病因）
query = infer.query(
    variables=['LungCancer'],
    evidence={'Cough':'Yes', 'XRay':'Abnormal'}
)
print(query)

# 预测推理（给定病因预测症状）
query = infer.query(
    variables=['Cough'],
    evidence={'Smoking':'Yes', 'AirPollution':'High'}
)
print(query)
```

### 五、结构学习
```python
from pgmpy.estimators import PC

# 使用PC算法学习结构
est = PC(data)
learned_model = est.estimate(
    variant='stable', 
    max_cond_vars=4,
    significance_level=0.01
)

# 比较学习结构与真实结构
print("Learned edges:", learned_model.edges())
```

### 六、连续变量处理
```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD

# 构建线性高斯模型
model = BayesianNetwork([('X', 'Y'), ('Z', 'Y')])
model.add_cpds(
    LinearGaussianCPD('X', [0], 1),
    LinearGaussianCPD('Z', [0], 0.5),
    LinearGaussianCPD('Y', ['X','Z'], [0.5, -0.3], 0.2, [0,0])
)

# 连续变量推理
infer = VariableElimination(model)
result = infer.query(variables=['Y'], evidence={'X':1.2, 'Z':0.8})
```

### 七、动态贝叶斯网络
```python
from pgmpy.models import DynamicBayesianNetwork as DBN

# 构建时间切片网络
dbn = DBN([
    (('D', 0), ('D', 1)),
    (('S', 0), ('S', 1))
])

# 定义转移概率
dbn.add_cpds(
    TabularCPD(('D',0), 2, [[0.8], [0.2]]),
    TabularCPD(('D',1), 2, [[0.7, 0.3], [0.3, 0.7]], 
               evidence=[('D',0)], evidence_card=[2])
)

# 时序推理
from pgmpy.inference import DBNInference
infer = DBNInference(dbn)
```

### 八、性能优化
1. **近似推理加速**：
   ```python
   from pgmpy.sampling import GibbsSampling
   
   gibbs = GibbsSampling(model)
   samples = gibbs.sample(size=1000, evidence={'Cough':'Yes'})
   posterior = samples['LungCancer'].value_counts(normalize=True)
   ```

2. **并行计算**：
   ```python
   from pgmpy.parallel import Parallel
   
   with Parallel(n_jobs=4) as parallel:
       results = parallel(
           delayed(infer.query)(variables=['X'], evidence={'Y':y})
           for y in range(5)
       )
   ```

3. **记忆化技术**：
   ```python
   from pgmpy.inference import CachedInference
   
   cached_infer = CachedInference(infer)
   cached_infer.query(variables=['A'], evidence={'B':1})
   ```

---

## 数学补充
**联合分布分解**：
$$
P(X_1,...,X_n) = \prod_{i=1}^n P(X_i \mid \text{Pa}(X_i))
$$

**马尔可夫毯定理**：
$$
\text{MB}(X_i) = \text{Pa}(X_i) \cup \text{Ch}(X_i) \cup \text{Pa}(\text{Ch}(X_i))
$$

**变量消除复杂度**：
$$
O(n \exp(w))
$$
其中w为消去序的树宽


## 典型应用场景
| 领域           | 应用案例                     | 网络规模       |
|----------------|----------------------------|---------------|
| 医疗诊断       | 疾病-症状关系建模          | 50-100节点    |
| 金融风控       | 欺诈检测系统               | 100-500节点   |
| 工业控制       | 故障诊断系统               | 20-50节点     |
| 自然语言处理   | 语义角色标注               | 1000+节点     |

---

贝叶斯网络通过显式建模变量间的因果关系，在不确定性推理和可解释性方面具有独特优势。相比朴素贝叶斯，虽然计算复杂度更高，但能更准确地刻画现实世界的复杂依赖关系。在医疗诊断、风险预测等领域已成为重要工具，最新进展结合深度学习发展出贝叶斯深度学习等混合方法。