# 导入相关库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载数据集
cancer = load_breast_cancer()
# 数据转换成DataFrame格式
cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target 

# 打印相关信息
print(cancer_df.shape)
print(cancer_df.info())
print(cancer_df.describe())

X = cancer_df.drop('target', axis=1) # 所有特性作为自变量
y = cancer.target  # 作为目标变量

# 分割数据集为训练集和测试集(训练集为80% 测试集20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化并训练逻辑回归模型
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率: {accuracy:.2f}')

# 分类报告
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 计算均方误差 (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差: {mse:.2f}')

# 计算R^2分数 (R^2 Score)
r2 = r2_score(y_test, y_pred)
print(f'R^2分数: {r2:.2f}')

# 可视化分析
# 1. 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = range(len(cancer.target_names))
plt.xticks(tick_marks, cancer.target_names)
plt.yticks(tick_marks, cancer.target_names)
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.grid(False)
plt.show()


#            precision    recall  f1-score   support

#    malignant       1.00      1.00      1.00        39
#       benign       1.00      1.00      1.00        75

#     accuracy                           1.00       114
#    macro avg       1.00      1.00      1.00       114
# weighted avg       1.00      1.00      1.00       114


# 从分类报告来看，模型在这个测试集每个指标都是 1.00，表示所有预测都正确,说明模型比较适合这类分类