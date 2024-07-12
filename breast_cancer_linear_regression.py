# 导入相关库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载乳腺癌数据集
cancer = load_breast_cancer()
# 数据转换成DataFrame
cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
# 创建一个target列
cancer_df['target'] = cancer.target  

# 输出相关信息
#"数据集列名
print(cancer_df.columns)
# 数据集形状
print( cancer_df.shape)
# 数据集信息摘要:
print( cancer_df.info())
# 数据集描述性统计信息:
print(cancer_df.describe())

# 选择特征和目标
X = cancer_df.drop('target', axis=1)  # 选择所有特征作为自变量
y = cancer_df['target']  # 选择目标变量（0：恶性，1：良性）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算和打印模型的性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差 (MSE): {mse}')
print(f'R^2得分: {r2}')

# 可视化真实值和预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('乳腺癌数据的线性回归拟合')
plt.show()

# 均方误差 (MSE): 0.06410886247029474
# R^2得分: 0.727101612622355
# 分析:
# 从mse 和R^2得分上看分为0.7271表明模型可以解释目标变量（乳腺癌类型）方差的约72.71%，这个拟合效果可以说是中等偏上的,
# MSE值较小，说明模型的预测值与真实值的差异相对较小，预测精度较高。
# 从plt图上看,数据集不是太适合使用线性进行拟合

