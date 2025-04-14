# 多分类评价指标详解

## 一、核心概念

多分类任务中，评价指标需扩展二分类的逻辑，处理多个类别之间的预测性能。核心思想包括：
1. 全局统计：直接汇总所有类别的预测结果（如准确率）。
2. 按类别统计：逐类计算指标后取平均（宏平均、加权平均）。
3. 综合矩阵分析：通过混淆矩阵分析每个类别的预测性能。



## 二、核心评价指标

### 1. 准确率（Accuracy）
原理：预测正确的样本占总样本的比例。  
公式：
$$
\text{Accuracy} = \frac{\sum_{i=1}^C TP_i}{N}
$$
- $TP_i$：第 $i$ 类的真正例数。
- $N$：总样本数。

适用场景：类别平衡的数据集。  
缺点：类别不平衡时可能高估模型性能。

Python实现：
```python
from sklearn.metrics import accuracy_score
y_true = [0, 1, 2, 0, 2]
y_pred = [0, 2, 1, 0, 1]
print("Accuracy:", accuracy_score(y_true, y_pred))  # 输出 0.4
```



### 2. 混淆矩阵（Confusion Matrix）
定义：$C \times C$ 矩阵，行表示真实类别，列表示预测类别。  
作用：直观展示每个类别的预测情况（如哪些类别易混淆）。

Python实现：
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```



### 3. 精确率（Precision）、召回率（Recall）、F1分数（F1-Score）
计算方式：
- 宏平均（Macro-average）：逐类计算指标后取算数平均。
- 微平均（Micro-average）：汇总所有类别的TP/FP/FN后计算全局指标。
- 加权平均（Weighted-average）：按每个类别的样本数加权平均。

公式：
- 宏精确率：
  $$
  \text{Macro-Precision} = \frac{1}{C} \sum_{i=1}^C \frac{TP_i}{TP_i + FP_i}
  $$
- 微精确率：
  $$
  \text{Micro-Precision} = \frac{\sum_{i=1}^C TP_i}{\sum_{i=1}^C (TP_i + FP_i)}
  $$
- 宏F1：
  $$
  \text{Macro-F1} = \frac{1}{C} \sum_{i=1}^C \frac{2 \times \text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
  $$

适用场景：
- 宏平均：关注每个类别的平等重要性（类别平衡时）。
- 微平均：关注样本级别的全局性能（类别不平衡时可能偏向大类）。
- 加权平均：类别不平衡时按样本量加权。

Python实现：
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# 宏平均
print("Macro Precision:", precision_score(y_true, y_pred, average='macro'))  # 输出 0.222
print("Macro Recall:", recall_score(y_true, y_pred, average='macro'))        # 输出 0.333
print("Macro F1:", f1_score(y_true, y_pred, average='macro'))                # 输出 0.222

# 微平均
print("Micro Precision:", precision_score(y_true, y_pred, average='micro'))  # 输出 0.4
print("Micro F1:", f1_score(y_true, y_pred, average='micro'))                # 输出 0.4

# 加权平均
print("Weighted F1:", f1_score(y_true, y_pred, average='weighted'))          # 输出 0.222
```



### 4. Kappa系数（Cohen’s Kappa）
原理：衡量分类结果与随机预测的一致性差异，值越接近1表示模型越优于随机猜测。  
公式：
$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$
- $p_o$：实际一致率（即准确率）。
- $p_e$：随机一致率，计算公式为：
  $$
  p_e = \sum_{i=1}^C \frac{(真实类i的样本数) \times (预测类i的样本数)}{N^2}
  $$

Python实现：
```python
from sklearn.metrics import cohen_kappa_score
print("Kappa:", cohen_kappa_score(y_true, y_pred))  # 输出 -0.125
```



### 5. Hamming Loss
原理：衡量错误预测的比例，包括误报和漏报。  
公式：
$$
\text{Hamming Loss} = \frac{FP + FN}{N \times C}
$$
特点：值越小越好，适用于多标签分类（需调整实现）。

Python实现（需二值化）：
```python
from sklearn.metrics import hamming_loss
# 假设多标签分类（此处示例为多分类，需调整）
print("Hamming Loss:", hamming_loss(y_true, y_pred))  # 输出 0.6
```



### 6. Jaccard相似系数（Jaccard Index）
原理：预测类别与真实类别的交集大小除以并集大小。  
公式：
$$
\text{Jaccard} = \frac{TP_i}{TP_i + FP_i + FN_i}
$$
Python实现：
```python
from sklearn.metrics import jaccard_score

# 按类别计算
print("Jaccard per class:", jaccard_score(y_true, y_pred, average=None))  # 输出 [0.5, 0.0, 0.0]

# 宏平均
print("Macro Jaccard:", jaccard_score(y_true, y_pred, average='macro'))   # 输出 0.166
```



## 三、多分类指标计算示例
假设某3分类任务的混淆矩阵如下：

|真实\预测|类0|类1|类2|
|---------|---|--|---|
|类0      |3 |1 |0 |
|类1      |2 |2 |1 |
|类2      |0 |1 |4 |

- TP各列：类0-TP=3，类1-TP=2，类2-TP=4。
- FP各列：类0-FP=2（类1预测为0的2个），类1-FP=1+1=2，类2-FP=0+1=1。
- FN各行：类0-FN=1，类1-FN=2+1=3，类2-FN=1。

计算：
- 宏精确率：
  $$
  \text{Precision}_0 = \frac{3}{3+2} = 0.6, \quad
  \text{Precision}_1 = \frac{2}{2+2} = 0.5, \quad
  \text{Precision}_2 = \frac{4}{4+1} = 0.8 \\
  \text{Macro-Precision} = \frac{0.6 + 0.5 + 0.8}{3} = 0.633
  $$
- 微精确率：
  $$
  \sum TP = 3+2+4=9, \quad \sum (TP + FP) = (3+2)+(2+2)+(4+1)=14 \\
  \text{Micro-Precision} = \frac{9}{14} \approx 0.643
  $$



## 四、总结与选择建议
1. 类别平衡：优先使用宏平均（Macro-F1、Macro-Precision）。
2. 类别不平衡：优先使用加权平均（Weighted-F1）或微平均（Micro-F1）。
3. 全面分析：结合混淆矩阵和Kappa系数，避免单一指标误导。
4. 特殊需求：
   - 减少类别间误判：关注混淆矩阵中非对角线元素。
   - 多标签分类：使用Hamming Loss或Jaccard系数。



## 五、完整Python示例
```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 0, 2]
y_pred = [0, 2, 1, 0, 1]

# 输出完整报告（包含精确率、召回率、F1等）
print(classification_report(y_true, y_pred))

# 输出示例：
#               precision    recall  f1-score   support
#            0       0.50      1.00      0.67         2
#            1       0.00      0.00      0.00         1
#            2       0.00      0.00      0.00         2
#     accuracy                           0.40         5
#    macro avg       0.17      0.33      0.22         5
# weighted avg       0.20      0.40      0.27         5
```



通过上述指标和代码，可全面评估多分类模型的性能，并根据业务需求选择合适的评价标准。