# 多标签分类与概率评估的评价指标详解



## 一、多标签分类评价指标

### 1. 汉明损失（Hamming Loss）
**原理**：衡量错误预测的标签比例（包括误报和漏报），值越小越好。  
**公式**：
$$
\text{Hamming Loss} = \frac{FP + FN}{N \times L}
$$
- $FP$: 假阳性数  
- $FN$: 假阴性数  
- $N$: 样本数  
- $L$: 标签总数  

**Python实现**：
```python
from sklearn.metrics import hamming_loss
y_true = [[0, 1, 1], [1, 0, 0]]
y_pred = [[0, 1, 0], [1, 0, 1]]
print("Hamming Loss:", hamming_loss(y_true, y_pred))  # 输出 0.333
```



### 2. 精确率（Precision）、召回率（Recall）、F1分数（F1-Score）
**计算方式**：
- **Micro平均**：全局统计TP、FP、FN，计算整体指标。  
- **Macro平均**：对每个标签单独计算指标后取平均。  
- **Sample平均**：对每个样本单独计算指标后取平均。  

**公式**（Micro-Precision）：
$$
\text{Micro-P} = \frac{\sum TP_i}{\sum (TP_i + FP_i)}
$$

**Python实现**：
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Micro平均
print("Micro Precision:", precision_score(y_true, y_pred, average='micro'))  # 输出 0.6
# Macro平均
print("Macro Precision:", precision_score(y_true, y_pred, average='macro'))  # 输出 0.5
```



### 3. 杰卡德相似系数（Jaccard Similarity Score）
**原理**：预测标签与真实标签的交集与并集的比值。  
**公式**：
$$
\text{Jaccard} = \frac{|Y_{\text{true}} \cap Y_{\text{pred}}|}{|Y_{\text{true}} \cup Y_{\text{pred}}|}
$$

**Python实现**：
```python
from sklearn.metrics import jaccard_score
print("Jaccard Score:", jaccard_score(y_true, y_pred, average='samples'))  # 样本平均
```



### 4. 子集准确率（Subset Accuracy）
**原理**：仅当所有标签均正确预测时计为正确，严格但罕见。  
**公式**：
$$
\text{Subset Accuracy} = \frac{1}{N} \sum_{i=1}^N I(Y_{\text{true}}^{(i)} = Y_{\text{pred}}^{(i)})
$$

**Python实现**：
```python
from sklearn.metrics import accuracy_score
print("Subset Accuracy:", accuracy_score(y_true, y_pred))  # 输出 0.0（未完全匹配）
```



### 5. 排序损失（Ranking Loss）
**原理**：衡量标签排序错误的比例（相关标签排在无关标签之后）。  
**公式**：
$$
\text{Ranking Loss} = \frac{1}{N} \sum_{i=1}^N \frac{|\{(a,b) | f(x_i,a) \leq f(x_i,b), a \in Y_i, b \notin Y_i\}|}{|Y_i| \times |\bar{Y}_i|}
$$

**Python实现**：
```python
from sklearn.metrics import label_ranking_loss
y_score = np.array([[0.1, 0.9, 0.8], [0.8, 0.2, 0.3]])  # 预测概率
y_true = np.array([[0, 1, 1], [1, 0, 0]])
print("Ranking Loss:", label_ranking_loss(y_true, y_score))  # 输出 0.25
```



## 二、概率预测评估指标

### 1. 对数损失（Log Loss）
**原理**：衡量预测概率与真实标签的交叉熵，值越小越好。  
**公式**：
$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \sum_{l=1}^L \left( y_{il} \log p_{il} + (1-y_{il}) \log(1-p_{il}) \right)
$$

**Python实现**：
```python
from sklearn.metrics import log_loss
y_true = [[0, 1], [1, 0]]
y_prob = [[0.1, 0.9], [0.9, 0.1]]
print("Log Loss:", log_loss(y_true, y_prob))  # 输出 0.105
```



### 2. 布里尔分数（Brier Score）
**原理**：概率预测的均方误差，适用于校准评估。  
**公式**：
$$
\text{Brier Score} = \frac{1}{N \times L} \sum_{i=1}^N \sum_{l=1}^L (p_{il} - y_{il})^2
$$

**Python实现**：
```python
from sklearn.metrics import brier_score_loss

# 二分类示例
y_true_binary = [0, 1, 1, 0]
y_prob_binary = [0.1, 0.9, 0.8, 0.2]
print("Brier Score:", brier_score_loss(y_true_binary, y_prob_binary))  # 输出 0.025
```



### 3. ROC AUC（多标签扩展）
**原理**：对每个标签单独计算AUC后取平均（Macro/Micro）。  
**Python实现**：
```python
from sklearn.metrics import roc_auc_score
y_true = np.array([[0, 1], [1, 0]])
y_score = np.array([[0.1, 0.9], [0.8, 0.2]])
print("Macro AUC:", roc_auc_score(y_true, y_score, average='macro'))  # 输出 1.0
```



### 4. 校准曲线（Calibration Curve）
**原理**：检验预测概率是否与真实频率一致。  
**Python实现**：
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

y_true_binary = [0, 0, 1, 1]
y_prob_binary = [0.1, 0.4, 0.6, 0.9]

prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_binary, n_bins=2)
plt.plot(prob_pred, prob_true, marker='o')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.show()
```



## 三、指标对比与选择建议

| 指标                | 适用场景                  | 优点                          | 缺点                      |
|---------------------|-------------------------|-----------------------------|--------------------------|
| **Hamming Loss**    | 多标签分类整体错误评估      | 直观反映标签错误比例            | 忽略标签间相关性            |
| **Micro-F1**        | 关注样本级别的全局性能      | 适用于类别不平衡                | 可能掩盖小类问题            |
| **Log Loss**        | 概率预测的严格评估          | 惩罚过度自信的错误预测           | 对噪声敏感                |
| **Brier Score**     | 概率校准质量评估            | 直接衡量概率准确性               | 仅适用于概率输出            |
| **Subset Accuracy** | 严格匹配场景               | 完全正确时高得分                | 过于严格，实际应用少        |



## 四、总结
- **多标签分类**：优先选择Hamming Loss、Micro-F1和Jaccard系数，结合Ranking Loss分析排序质量。  
- **概率评估**：使用Log Loss和Brier Score评估概率准确性，结合校准曲线优化模型输出。  
- **业务需求**：根据场景选择严格指标（如Subset Accuracy）或宽松指标（如Hamming Loss）。  
- **综合评估**：联合多个指标全面验证模型性能，避免单一指标误导。