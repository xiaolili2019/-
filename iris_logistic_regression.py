# 导入相关库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score,mean_squared_error, r2_score 


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# 加载数据集
iris = load_iris()
# 转换为 Pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = iris.data[:, [0, 2]]  # 花萼长度和花瓣长度作为特征
y = iris.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 创建逻辑回归模型
lr = LogisticRegression()

# 训练模型
lr.fit(X_train, y_train)

# 预测测试集结果
y_pred = lr.predict(X_test)

# 输出准确率和分类报告
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率: {accuracy:.2f}')
print('分类报告:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差 (MSE): {mse:.2f}')

# 计算决定系数（R^2）
r2 = r2_score(y_test, y_pred)
print(f'R^2得分: {r2:.2f}')

# 可视化分类结果及决策边界
plt.figure(figsize=(10, 6))

# 绘制训练集中的数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolor='k', label='训练集')

# 绘制测试集中的数据点
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set2, marker='x', s=100, edgecolor='k', label='测试集')

# 绘制决策边界
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

# 添加图例和标签
plt.xlabel('花萼长度 (cm)')
plt.ylabel('花瓣长度 (cm)')
plt.title('逻辑回归分类结果及决策边界')
plt.legend()
plt.show()




# 准确率 (Accuracy): 0.90
# 分类报告:
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        11
#   versicolor       0.86      0.92      0.89        13
#    virginica       0.80      0.67      0.73         6

#     accuracy                           0.90        30
#    macro avg       0.89      0.86      0.87        30
# weighted avg       0.90      0.90      0.90        30


# 简单分析

# 模型在测试集上的准确率为 0.90，表示模型预测正确的比例为90%。
# 分类报告（Classification Report）:

# 对于类别 setosa，模型表现完美，精确率（precision）、召回率（recall）和 F1-score 都达到了1.00，支持数为11。
# 对于类别 versicolor，精确率为0.86，召回率为0.92，F1-score为0.89，支持数为13。
# 对于类别 virginica，精确率为0.80，召回率为0.67，F1-score为0.73，支持数为6。
# 加权平均的准确率（weighted avg）为0.90，加权平均的召回率（weighted avg）也为0.90，加权平均的 F1-score 为0.90。
# 模型的均方误差为0.10，说明模型预测的值与真实值的平均平方差较小。
# 模型的决定系数为0.81，这个值表示模型可以解释目标变量方差的81%，是一个比较良好的拟合度。