# region p1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载糖尿病数据集
data = load_diabetes()
X = data.data  # 特征矩阵 (442x10)
y = data.target  # 目标值（疾病进展指标）
feature_names = data.feature_names

# 数据探索
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
print(df.describe())  # 统计描述
sns.pairplot(df[["age", "bmi", "bp", "target"]])  # 特征与目标关系可视化
plt.show()
# endregion p1

# region p2
# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化（Z-score标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# endregion p2

# region p3
# 使用Scikit-Learn训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 获取系数与截距
print("截距项 (β0):", model.intercept_)
print("特征系数:")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")
# endregion p3

# region p4
# 预测与评估
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# 残差分析
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title("残差分布")
plt.xlabel("残差值")
plt.show()

# Q-Q图检验正态性
import statsmodels.api as sm

sm.qqplot(residuals, line="45")
plt.show()
# endregion p4

# region p5
# 真实值 vs 预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="w", s=80)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("真实值", fontsize=12)
plt.ylabel("预测值", fontsize=12)
plt.title("线性回归预测结果", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# 特征重要性排序
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
coef_df = coef_df.sort_values(by="Coefficient", key=abs, ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="viridis")
plt.title("特征系数绝对值排序", fontsize=14)
plt.xlabel("系数值", fontsize=12)
plt.ylabel("特征", fontsize=12)
plt.show()
# endregion p5
