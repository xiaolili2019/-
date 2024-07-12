import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载波士顿房价数据集
boston_df = pd.read_csv('boston_housing_data.csv')

# 输出相关信息
# 索引列表是波士顿房价数据集:
# Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
#        'PTRATIO', 'B', 'LSTAT', 'target'],
#       dtype='object')

# CRIM: 各城镇的人均犯罪率
# ZN: 占地面积超过 25,000 平方英尺的住宅用地比例
# INDUS: 城镇中非零售营业比例的土地面积
# CHAS: 查尔斯河虚拟变量（如果某条记录在河边则为 1，否则为 0）
# NOX: 一氧化氮浓度（每千万份）
# RM: 每个住宅的平均房间数
# AGE: 1940 年之前建造的自住房屋的比例
# DIS: 到波士顿五个就业中心的加权距离
# RAD: 径向公路的可达性指数
# TAX: 每 10,000 美元的全额物业税率
# PTRATIO: 城镇的师生比例
# B: 1000(Bk - 0.63)^2 其中 Bk 是城镇中黑人的比例
# LSTAT: 人口中地位较低者的百分比
# target: 自住房屋的中位数价格（单位：千美元）
print(boston_df.columns)
# data columns (total 14 columns):     
#   Column   Non-Null Count  Dtype  
# ---  ------   --------------  -----  
#  0   CRIM     506 non-null    float64
#  1   ZN       506 non-null    float64
#  2   INDUS    506 non-null    float64
#  3   CHAS     506 non-null    float64
#  4   NOX      506 non-null    float64
#  5   RM       506 non-null    float64
#  6   AGE      506 non-null    float64
#  7   DIS      506 non-null    float64
#  8   RAD      506 non-null    float64
#  9   TAX      506 non-null    float64
#  10  PTRATIO  506 non-null    float64
#  11  B        506 non-null    float64
#  12  LSTAT    506 non-null    float64
#  13  target   506 non-null    float64
# dtypes: float64(14)
# 数据类型都是float64 总共有13个特征
# 描述了波士顿房价数据集中的各个特征列和目标列的数据情况
# CRIM: 各城镇的人均犯罪率（非空值计数为 506）。
# ZN: 占地面积超过 25,000 平方英尺的住宅用地比例（非空值计数为 506）。
# INDUS: 城镇中非零售营业比例的土地面积（非空值计数为 506）。
# CHAS: 查尔斯河虚拟变量（如果某条记录在河边则为 1，否则为 0，非空值计数为 506）。
# NOX: 一氧化氮浓度（每千万份，非空值计数为 506）。
# RM: 每个住宅的平均房间数（非空值计数为 506）。
# AGE: 1940 年之前建造的自住房屋的比例（非空值计数为 506）。
# DIS: 到波士顿五个就业中心的加权距离（非空值计数为 506）。
# RAD: 径向公路的可达性指数（非空值计数为 506）。
# TAX: 每 10,000 美元的全额物业税率（非空值计数为 506）。
# PTRATIO: 城镇的师生比例（非空值计数为 506）。
# B: 1000(Bk - 0.63)^2 其中 Bk 是城镇中黑人的比例（非空值计数为 506）。
# LSTAT: 人口中地位较低者的百分比（非空值计数为 506）。
# target: 自住房屋的中位数价格（单位：千美元，非空值计数为 506）。
# 这些信息表明，该数据集中的所有列均包含 506 个非空值，数据类型为浮点数，适合用于机器学习算法的训练和分析。
# 都是非空506 说明数据没有空值,不需要进行数据处理
print(boston_df.info())

#              CRIM          ZN       INDUS        CHAS         NOX          RM  ...         RAD         TAX     PTRATIO           B       LSTAT      target
# count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  ...  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000
# mean     3.613524   11.363636   11.136779    0.069170    0.554695    6.284634  ...    9.549407  408.237154   18.455534  356.674032   12.653063   22.532806
# std      8.601545   23.322453    6.860353    0.253994    0.115878    0.702617  ...    8.707259  168.537116    2.164946   91.294864    7.141062    9.197104
# min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000  ...    1.000000  187.000000   12.600000    0.320000    1.730000    5.000000
# 25%      0.082045    0.000000    5.190000    0.000000    0.449000    5.885500  ...    4.000000  279.000000   17.400000  375.377500    6.950000   17.025000
# 50%      0.256510    0.000000    9.690000    0.000000    0.538000    6.208500  ...    5.000000  330.000000   19.050000  391.440000   11.360000   21.200000
# 75%      3.677083   12.500000   18.100000    0.000000    0.624000    6.623500  ...   24.000000  666.000000   20.200000  396.225000   16.955000   25.000000
# max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000  ...   24.000000  711.000000   22.000000  396.900000   37.970000   50.000000

# [8 rows x 14 columns]
print(boston_df.describe())


# 选择特征和目标
X = boston_df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']].values  # 选择多个特征作为自变量
y = boston_df['target'].values  # 房价作为目标变量

# 划分训练集和测试集(测试集为20%,训练集为80%,随机生成)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# 创建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')
plt.xlabel('实际房价')
plt.ylabel('预测房价')
plt.title('实际房价&预测房价')
plt.legend()
plt.show()

# 计算性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差 (MSE): {mse:.2f}')
print(f'R²得分: {r2:.2f}')

# 均方误差 (MSE): 24.29 意味着模型对于某些样本的预测偏离实际值较远，或者模型整体的预测精度不高。
# R²得分: 67%的因变量的方差是一个合理的拟合度量
#
