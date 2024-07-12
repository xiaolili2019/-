# 导入相关的库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 逻辑回归预测思路:把房价转化成二元分类任务(也可以转换成多元分类任务),设置阈值为房价中位数,将房价作为目标变量（标签），将特征作为输入变量（特征矩阵）。
# 将房价与某个阈值比较，大于阈值则为一类（高价房），小于或等于阈值则为另一类（低价房）

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载波士顿房价数据集(从函数中自动下载,手动下载并加载)
boston_df = pd.read_csv('boston_housing_data.csv')
# 输出相关信息
# 形状 (506, 14) 506行14列
#     CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  target
# 0    0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98    24.0
# 1    0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14    21.6
# 2    0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03    34.7
# 3    0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94    33.4
# 4    0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33    36.2
# ..       ...   ...    ...   ...    ...    ...   ...     ...  ...    ...      ...     ...    ...     ...
# 501  0.06263   0.0  11.93   0.0  0.573  6.593  69.1  2.4786  1.0  273.0     21.0  391.99   9.67    22.4
# 502  0.04527   0.0  11.93   0.0  0.573  6.120  76.7  2.2875  1.0  273.0     21.0  396.90   9.08    20.6
# 503  0.06076   0.0  11.93   0.0  0.573  6.976  91.0  2.1675  1.0  273.0     21.0  396.90   5.64    23.9
# 504  0.10959   0.0  11.93   0.0  0.573  6.794  89.3  2.3889  1.0  273.0     21.0  393.45   6.48    22.0
# 505  0.04741   0.0  11.93   0.0  0.573  6.030  80.8  2.5050  1.0  273.0     21.0  396.90   7.88    11.9

# [506 rows x 14 columns]
print(boston_df)
print(boston_df.shape)
print(boston_df.describe)

# 选择特征和目标
X = boston_df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']].values  # 所有特征作为自变量
y = boston_df['target'].values  # 房价作为目标变量

# 将房价转换为二元分类任务（设置阈值为房价中位数）
median_price = boston_df['target'].median()
y_binary = (y > median_price).astype(int)  # 高于中位数的房价为类别 1，低于等于中位数的房价为类别 0

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率：{accuracy:.2f}')


# 输出分类报告
class_report = classification_report(y_test, y_pred)
print('分类报告:')
print(class_report)


# 可视化
# 1. 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = range(len(set(y_binary)))
plt.xticks(tick_marks, set(y_binary))
plt.yticks(tick_marks, set(y_binary))
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.grid(False)
plt.show()

# 简要分析
# 准确率：0.84 逻辑回归模型在波士顿房价数据集上的分类效果良好，准确率较高.
# 分类报告:
#               precision    recall  f1-score   support

#            0       0.88      0.85      0.86        60
#            1       0.80      0.83      0.81        42

#     accuracy                           0.84       102
#    macro avg       0.84      0.84      0.84       102
# weighted avg       0.84      0.84      0.84       102

# 准确率达到了0.84，这意味着模型正确预测了84%的样本。
# 分类报告显示了每个类别的精确度（precision）、召回率（recall）和 F1-score。在二元分类任务中，类别 0 的精确度为0.88，召回率为0.85；类别 1 的精确度为0.80，召回率为0.83。
# Macro avg 表示所有类别的平均值，显示了模型在类别间的平均表现，而 weighted avg 则考虑了样本不均衡的影响。
# 综上所述，逻辑回归模型在这个任务上能够较为准确地区分高价房和低价房，整体表现良好。
