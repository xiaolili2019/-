# 导入相关库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris   # 导入鸢尾花数据集
from sklearn.metrics import mean_squared_error, r2_score 
# mean_squared_error, r2_score  是两个常用的评估回归模型性能的指标
# mean_squared_error 是均方误差，它衡量的是预测值与真实值之间的平均平方差,MSE 越小，模型的预测结果越接近真实值，模型性能越好。
#  r2_score 是决定系数，它表示预测值与真实值之间的拟合程度,值为 1 表示完美拟合，为 0 表示模型只达到简单平均值的水平，
#  负值表示模型比简单平均值更差。


# 设置中文字体 （设置无法显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载鸢尾花数据集（花萼长度，花瓣长度 两个因子）
# Iris 数据集包含 150 个样本，分为 3 个类别，每个类别有 50 个样本。每个样本有 4 个特征，分别是：
# 1.	Sepal length（萼片长度）
# 2.	Sepal width（萼片宽度）
# 3.	Petal length（花瓣长度）
# 4.	Petal width（花瓣宽度）
# 这三个类别分别代表三种鸢尾花：
# 1.	Setosa
# 2.	Versicolor
# 3.	Virginica

# 可以修改其他特征进行线性回归拟合

iris = load_iris()
# 转换为 Pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = iris.data[:, 0].reshape(-1, 1)  # 选择花萼长度作为自变量
y = iris.data[:, 2]  # 选择花瓣长度作为因变量
# 输出数据集的形状
print(iris.data.shape)
# 输出数据集的描述信息

# Summary Statistics:
# ============== ==== ==== ======= ===== ====================
#                 Min  Max   Mean    SD   Class Correlation
# ============== ==== ==== ======= ===== ====================
# sepal length:   4.3  7.9   5.84   0.83    0.7826
# sepal width:    2.0  4.4   3.05   0.43   -0.4194
# petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
# petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
# ============== ==== ==== ======= ===== ====================

# :Missing Attribute Values: None
# :Class Distribution: 33.3% for each of 3 classes.
# :Creator: R.A. Fisher
# :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)

print(iris.DESCR)

# 查看数据集基本信息
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 4 columns):
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   sepal length (cm)  150 non-null    float64
#  1   sepal width (cm)   150 non-null    float64
#  2   petal length (cm)  150 non-null    float64
#  3   petal width (cm)   150 non-null    float64
# dtypes: float64(4)

#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# count         150.000000        150.000000         150.000000        150.000000
# mean            5.843333          3.057333           3.758000          1.199333
# std             0.828066          0.435866           1.765298          0.762238
# min             4.300000          2.000000           1.000000          0.100000
# 25%             5.100000          2.800000           1.600000          0.300000
# 50%             5.800000          3.000000           4.350000          1.300000
# 75%             6.400000          3.300000           5.100000          1.800000
# max             7.900000          4.400000           6.900000          2.500000

print(iris_df.info())

print(iris_df.describe())



# 创建线性回归模型
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='真实值') # 绘制散点图,颜色是蓝色。
plt.plot(X, y_pred, color='red', label='预测值')

# 打印预测函数
intercept = model.intercept_ # 获取线性回归模型的截距
coef = model.coef_[0] # 获取线性回归模型的系数,线性预测函数: y = 系数 * x + 截距。
prediction_function = f'预测函数: y = {coef:.2f} * x + {intercept:.2f}'
print(prediction_function)

# 在图中显示预测函数
plt.text(4.5, 6.5, prediction_function, fontsize=12, color='black')

plt.xlabel('花萼长度 (cm)')
plt.ylabel('花瓣长度 (cm)')
plt.legend()
plt.title('花萼长度与花瓣长度的线性拟合')
plt.show()

# 计算和打印模型的性能指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'均方误差 (MSE): {mse}')
print(f'R^2得分: {r2}')

# 均方误差 (MSE): 0.7430610341321241
# R^2得分: 0.759954645772515
# MSE 越小越接近真实性 0.74 预测一般
# 拟合程度 较好
