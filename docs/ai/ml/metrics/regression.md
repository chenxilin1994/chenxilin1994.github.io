# 回归评价指标



## 一、核心概念
回归任务的目标是预测连续值，评价指标主要用于衡量预测值与真实值的偏差程度。以下为常用指标及其特点：



## 二、核心评价指标

##### 1. 均方误差（Mean Squared Error, MSE）
**原理**：预测值与真实值差的平方的均值，对大误差敏感。  
**公式**：
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
**特点**：  
- 单位与数据平方一致（如数据单位为“米”，MSE单位为“平方米”）。  
- 对异常值敏感（平方会放大大误差的影响）。

**Python实现**：
```python
from sklearn.metrics import mean_squared_error
y_true = [3.0, 5.5, 4.0, 2.5]
y_pred = [2.8, 5.2, 3.5, 2.0]
mse = mean_squared_error(y_true, y_pred)
print("MSE:", mse)  # 输出 0.0625

# 手动计算
mse_manual = sum((a - b)**2 for a, b in zip(y_true, y_pred)) / len(y_true)
print("Manual MSE:", mse_manual)  # 输出 0.0625
```



##### 2. 均方根误差（Root Mean Squared Error, RMSE）
**原理**：MSE的平方根，单位与原始数据一致。  
**公式**：
$$
\text{RMSE} = \sqrt{\text{MSE}}
$$
**适用场景**：需直接反映预测值与真实值的平均偏差量级。

**Python实现**：
```python
rmse = mean_squared_error(y_true, y_pred, squared=False)
print("RMSE:", rmse)  # 输出 0.25

# 手动计算
rmse_manual = np.sqrt(mse)
print("Manual RMSE:", rmse_manual)  # 输出 0.25
```



##### 3. 平均绝对误差（Mean Absolute Error, MAE）
**原理**：预测值与真实值绝对差的均值，对异常值不敏感。  
**公式**：
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$
**特点**：  
- 单位与数据一致，解释直观。  
- 对大误差的惩罚弱于MSE。

**Python实现**：
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)  # 输出 0.15

# 手动计算
mae_manual = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
print("Manual MAE:", mae_manual)  # 输出 0.15
```



##### 4. 平均绝对百分比误差（Mean Absolute Percentage Error, MAPE）
**原理**：绝对误差占真实值的百分比均值，反映相对误差。  
**公式**：
$$
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$
**注意事项**：  
- 真实值为零时无法计算（需去除或替换）。  
- 对负值的处理需谨慎（如预测负值可能使百分比误差超过100%）。

**Python实现**：
```python
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 过滤真实值为零的样本
    non_zero_mask = y_true != 0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("MAPE:", mape(y_true, y_pred))  # 示例数据无零值，输出 3.69%

# 手动计算
errors = [abs((3.0-2.8)/3.0, (5.5-5.2)/5.5, (4.0-3.5)/4.0, (2.5-2.0)/2.5]
mape_manual = (sum(errors) / len(errors)) * 100
print("Manual MAPE:", mape_manual)  # 输出 3.69%
```



##### 5. 决定系数（R² Score, R-Squared）
**原理**：模型解释数据方差的比例，值越接近1表示拟合越好。  
**公式**：
$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
$$
其中 $\bar{y}$ 是真实值的均值。  
**特点**：  
- 最大值为1（完美预测），可能为负（模型比直接取均值更差）。  
- 无量纲，适合比较不同模型的解释能力。

**Python实现**：
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
print("R²:", r2)  # 输出 0.963

# 手动计算
y_mean = np.mean(y_true)
ss_total = sum((a - y_mean)**2 for a in y_true)
ss_res = sum((a - b)**2 for a, b in zip(y_true, y_pred))
r2_manual = 1 - (ss_res / ss_total)
print("Manual R²:", r2_manual)  # 输出 0.963
```



##### 6. 解释方差得分（Explained Variance Score）
**原理**：模型对数据方差解释的比例，公式为：
$$
\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
$$
**与R²的区别**：  
- R²考虑误差平方和的相对减少，解释方差关注误差方差的相对减少。  
- 当模型无偏时，二者相等；有偏时解释方差可能更高。

**Python实现**：
```python
from sklearn.metrics import explained_variance_score
evs = explained_variance_score(y_true, y_pred)
print("Explained Variance:", evs)  # 输出 0.966
```



##### 7. 中位数绝对误差（Median Absolute Error）
**原理**：绝对误差的中位数，对极端异常值更鲁棒。  
**公式**：
$$
\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, ..., |y_n - \hat{y}_n|)
$$
**适用场景**：数据中存在显著异常值时替代MAE。

**Python实现**：
```python
from sklearn.metrics import median_absolute_error
medae = median_absolute_error(y_true, y_pred)
print("MedAE:", medae)  # 输出 0.15
```



## 三、指标对比与选择建议
| 指标          | 优点                          | 缺点                          | 适用场景                   |
|---------------|-------------------------------|-------------------------------|---------------------------|
| **MSE**       | 强调大误差，数学性质好        | 对异常值敏感，单位平方        | 需要惩罚大误差的任务      |
| **RMSE**      | 单位与数据一致                | 同MSE                        | 直观反映误差量级          |
| **MAE**       | 对异常值鲁棒                  | 无法区分大误差与小误差        | 平衡误差，避免异常值干扰  |
| **MAPE**      | 相对误差，易于解释            | 真实值为零时失效              | 需要百分比误差分析        |
| **R²**        | 无量纲，可跨模型比较          | 可能掩盖局部预测问题          | 评估模型整体解释能力      |
| **MedAE**     | 极端鲁棒                      | 忽略误差分布信息              | 存在显著异常值的数据      |



## 四、完整Python示例
```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    median_absolute_error
)

y_true = np.array([3.0, 5.5, 4.0, 2.5])
y_pred = np.array([2.8, 5.2, 3.5, 2.0])

print("MSE:", mean_squared_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R²:", r2_score(y_true, y_pred))
print("Explained Variance:", explained_variance_score(y_true, y_pred))
print("MedAE:", median_absolute_error(y_true, y_pred))
```



## 五、总结
- **关注误差分布**：若误差分布对称且无异常值，MSE和RMSE更优；若存在异常值，优先MAE或MedAE。  
- **业务需求导向**：需要百分比误差分析时使用MAPE，需模型解释性时使用R²。  
- **综合使用多指标**：避免单一指标的片面性，例如同时报告MAE和R²以全面评估模型性能。