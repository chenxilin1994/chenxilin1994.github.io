# 支持向量机（SVM）算法原理

支持向量机（Support Vector Machine, SVM）是一种基于统计学习理论和结构风险最小化的监督学习算法，其核心思想是通过最大化分类间隔（Margin）来构建最优分类超平面。



## 一、数学基础与优化目标

### 1. 超平面与间隔的数学定义

- 超平面方程：在$n$维空间中，超平面定义为$w^T x + b = 0$，其中：
  -$w \in \mathbb{R}^n$是法向量，决定超平面的方向。
  -$b \in \mathbb{R}$是偏置项，控制超平面与原点的距离。
- 函数间隔：对样本点$(x_i, y_i)$，函数间隔为：
$$
\hat{\gamma}_i = y_i(w^T x_i + b)
$$
表示分类的正确性和置信度（值越大分类越可靠）。
- 几何间隔：归一化后的间隔，定义为：
$$
\gamma_i = \frac{\hat{\gamma}_i}{\|w\|} = \frac{y_i(w^T x_i + b)}{\|w\|}
$$
几何间隔是样本到超平面的真实欧氏距离。

### 2. 最大化间隔的优化问题

- 目标函数：寻找使最小几何间隔最大的超平面，等价于：
$$
\max_{w,b} \min_i \gamma_i \quad \Rightarrow \quad \max_{w,b} \frac{1}{\|w\|} \min_i y_i(w^T x_i + b)
$$
- 约束简化：通过缩放$w$和$b$，可固定最小函数间隔为1，优化问题转化为：
$$
\min_{w,b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \quad \forall i
$$
这里使用$\frac{1}{2} \|w\|^2$是为了后续求导方便，且与原问题等价。

### 3. 拉格朗日对偶问题

- 拉格朗日函数：引入拉格朗日乘子$\alpha_i \geq 0$，构造拉格朗日函数：
$$
\mathcal{L}(w,b,\alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^n \alpha_i \left[ y_i(w^T x_i + b) - 1 \right]
$$
- 对偶问题推导：
  1. 对$w$和$b$求偏导并令其为零：
$$
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i=1}^n \alpha_i y_i x_i = 0 \quad \Rightarrow \quad w = \sum_{i=1}^n \alpha_i y_i x_i
$$
$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^n \alpha_i y_i = 0
$$
  2. 将$w$代入拉格朗日函数，消去$w$和$b$，得到对偶问题：
$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j \quad \text{s.t.} \quad \alpha_i \geq 0, \quad \sum_{i=1}^n \alpha_i y_i = 0
$$
- 支持向量：最终解中$\alpha_i > 0$对应的样本即为支持向量，位于间隔边界上（满足$y_i(w^T x_i + b) = 1$）。



## 二、软间隔与松弛变量

### 1. 线性不可分问题的处理

- 引入松弛变量$\xi_i$：允许部分样本违反间隔约束，优化问题变为：
$$
\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$
  -$C$是正则化参数，平衡间隔最大化和误分类惩罚。
  -$\xi_i$表示第$i$个样本的违反程度：$ \xi_i = 0$表示无违反，$ \xi_i > 0$表示样本位于间隔内或错分侧。

### 2. 软间隔的拉格朗日对偶

- 拉格朗日函数：
$$
\mathcal{L}(w,b,\xi,\alpha,\mu) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i \left[ y_i(w^T x_i + b) - 1 + \xi_i \right] - \sum_{i=1}^n \mu_i \xi_i
$$
- 对偶问题推导：
  1. 对$w, b, \xi_i$求偏导并令其为零：
$$
\frac{\partial \mathcal{L}}{\partial w} = 0 \Rightarrow w = \sum_{i=1}^n \alpha_i y_i x_i
$$
$$
\frac{\partial \mathcal{L}}{\partial b} = 0 \Rightarrow \sum_{i=1}^n \alpha_i y_i = 0
$$
$$
\frac{\partial \mathcal{L}}{\partial \xi_i} = 0 \Rightarrow C - \alpha_i - \mu_i = 0 \Rightarrow \mu_i = C - \alpha_i
$$
  2. 代入后得到对偶问题：
$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0
$$
- KKT条件：
  -$\alpha_i = 0$：样本正确分类且在间隔外。
  -$0 < \alpha_i < C$：样本位于间隔边界上（支持向量）。
  -$\alpha_i = C$：样本违反间隔约束（位于间隔内或错分侧）。



## 三、核技巧与非线性SVM

### 1. 核函数的核心思想

- 非线性映射$\phi(x)$：将原始特征$x$映射到高维空间$\mathcal{H}$，使数据线性可分。
- 核函数定义：直接计算映射后的内积：
$$
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
$$
避免显式计算$\phi(x)$，从而解决“维数灾难”。

### 2. 常用核函数及其数学性质

- 线性核：
$$
K(x_i, x_j) = x_i^T x_j
$$
适用于线性可分数据，无额外参数。
- 多项式核：
$$
K(x_i, x_j) = (\gamma x_i^T x_j + c)^d
$$
  -$\gamma$控制项式权重，$ c$为常数项，$ d$为多项式次数。
- 高斯核（RBF）：
$$
K(x_i, x_j) = \exp\left( -\gamma \|x_i - x_j\|^2 \right)
$$
  -$\gamma$控制高斯函数的宽度，$ \gamma$越大模型越复杂。
- Sigmoid核：
$$
K(x_i, x_j) = \tanh(\gamma x_i^T x_j + c)
$$
类似于神经网络激活函数，需调整$\gamma$和$c$避免饱和。

### 3. 核函数选择与验证

- Mercer条件：核矩阵$K$必须是对称半正定的，确保优化问题凸性。
- 核函数评估：
  - 交叉验证：通过网格搜索测试不同核函数和参数。
  - 数据特性：线性核适合高维稀疏数据（如文本），RBF核适合低维密集非线性数据。



## 四、序列最小优化（SMO）算法详解

### 1. SMO的动机

- 大规模QP问题求解困难：传统二次规划算法复杂度为$O(n^3)$，无法处理大规模数据。
- 分解思想：每次仅优化两个变量，固定其他变量，将问题分解为子问题。

### 2. SMO算法步骤

1. 选择两个拉格朗日乘子$\alpha_i$和$\alpha_j$：
   - 启发式选择：优先选择违反KKT条件的样本。
     - 第一个变量$\alpha_i$：遍历所有样本，找到违反$0 < \alpha_i < C$的样本。
     - 第二个变量$\alpha_j$：选择使$|E_i - E_j|$最大的样本（$ E_i = f(x_i) - y_i$为预测误差）。
2. 解析求解子问题：
   - 固定其他变量，优化$\alpha_i$和$\alpha_j$，约束条件为：
$$
\alpha_i y_i + \alpha_j y_j = \zeta \quad (\zeta \text{ 为常数})
$$
   - 计算无约束最优解$\alpha_j^{\text{new}}$，并进行剪辑：
$$
\alpha_j^{\text{new, clipped}} = \begin{cases}
H & \text{if } \alpha_j^{\text{new}} \geq H \\
\alpha_j^{\text{new}} & \text{if } L < \alpha_j^{\text{new}} < H \\
L & \text{if } \alpha_j^{\text{new}} \leq L
\end{cases}
$$
其中$L = \max(0, \alpha_j - \alpha_i)$，$ H = \min(C, C + \alpha_j - \alpha_i)$（当$y_i \neq y_j$时）。
3. 更新阈值$b$：
   - 根据新的$\alpha_i$和$\alpha_j$计算$b$：
$$
b_1 = b - E_i - y_i (\alpha_i^{\text{new}} - \alpha_i^{\text{old}}) K(x_i, x_i) - y_j (\alpha_j^{\text{new}} - \alpha_j^{\text{old}}) K(x_j, x_i)
$$
$$
b_2 = b - E_j - y_i (\alpha_i^{\text{new}} - \alpha_i^{\text{old}}) K(x_i, x_j) - y_j (\alpha_j^{\text{new}} - \alpha_j^{\text{old}}) K(x_j, x_j)
$$
   - 最终$b$取$b_1$和$b_2$的平均值（若$\alpha_i, \alpha_j$在边界内）。
4. 更新误差缓存：
   - 对每个样本$k$，更新误差：
$$
E_k = E_k + y_i (\alpha_i^{\text{new}} - \alpha_i^{\text{old}}) K(x_i, x_k) + y_j (\alpha_j^{\text{new}} - \alpha_j^{\text{old}}) K(x_j, x_k) + b^{\text{new}} - b^{\text{old}}
$$
5. 终止条件：
   - 所有样本满足KKT条件（容忍一定误差$\epsilon$）。
   - 达到最大迭代次数。



## 五、多分类与参数调优

### 1. 多分类策略

- 一对多（OvR）：
  - 训练$K$个二分类器，第$k$个分类器将第$k$类作为正类，其余为负类。
  - 预测时选择$f_k(x) = w_k^T x + b_k$最大的类别。
- 一对一（OvO）：
  - 训练$\frac{K(K-1)}{2}$个分类器，每个区分一对类别$(i, j)$。
  - 预测时采用投票机制，得票最多的类别胜出。

### 2. 参数调优方法

- 网格搜索（Grid Search）：
  - 对$C$和核参数（如RBF的$\gamma$）进行交叉验证。
  - 示例代码（Scikit-learn）：
    ```python
    from sklearn.model_selection import GridSearchCV
    parameters = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    svc = SVC(kernel='rbf')
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)
    ```
- 贝叶斯优化：使用贝叶斯方法高效搜索超参数空间。

### 3. 数据预处理

- 特征标准化：将特征缩放到均值为0，方差为1：
$$
x' = \frac{x - \mu}{\sigma}
$$
- 处理类别不平衡：
  - 调整类别权重（如Scikit-learn中的 `class_weight='balanced'`）。
  - 使用过采样（SMOTE）或欠采样技术。



## 六、实际应用与案例

### 1. 图像分类

- 步骤：
  1. 提取图像特征（如HOG、SIFT）。
  2. 使用RBF核SVM进行分类。
- 案例：MNIST手写数字识别，准确率可达98%以上。

### 2. 文本分类

- 步骤：
  1. 将文本转换为TF-IDF向量。
  2. 使用线性核SVM进行分类（高维稀疏数据适合线性核）。
- 案例：垃圾邮件检测，新闻分类。

### 3. 生物信息学

- 基因表达数据分类：
  - 使用RBF核SVM区分癌症样本与正常样本。
  - 特征选择（如t-test）提升模型性能。



## 七、SVM的扩展与进阶

### 1. 支持向量回归（SVR）

- 目标：拟合一个间隔带（$ \epsilon$-insensitive tube），使大部分样本位于带内。
- 优化问题：
$$
\min_{w,b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
$$
$$
\text{s.t.} \quad y_i - (w^T x_i + b) \leq \epsilon + \xi_i, \quad (w^T x_i + b) - y_i \leq \epsilon + \xi_i^*, \quad \xi_i, \xi_i^* \geq 0
$$

### 2. 半监督SVM

- 核心思想：利用未标注数据提升分类器性能。
- TSVM（Transductive SVM）：将未标注数据作为测试集，调整超平面以最大化间隔。

### 3. 深度学习与SVM结合

- 深度特征提取：使用CNN提取图像特征，输入SVM进行分类。
- SVM作为分类层：替代Softmax，增强模型泛化能力。



## 八、总结与资源推荐

### 1. 核心贡献总结

- 结构风险最小化：通过间隔最大化控制模型复杂度。
- 核方法：将低维非线性问题映射到高维线性空间。
- 高效优化算法：SMO算法实现大规模数据训练。

### 2. 学习资源推荐

- 书籍：
  - 《统计学习方法》李航（第7章）
  - 《Pattern Recognition and Machine Learning》Christopher Bishop（第7章）
- 论文：
  - 《A Tutorial on Support Vector Machines for Pattern Recognition》Burges, 1998
  - 《Sequential Minimal Optimization: A Fast Algorithm for Training SVM》Platt, 1998
- 代码库：
  - LIBSVM（C++/Python）
  - Scikit-learn SVM模块
