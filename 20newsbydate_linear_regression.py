# 导入相关的库
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载训练集
train_data = load_files('20newsbydate/20news-bydate-train', encoding='latin1')

# 加载测试集
test_data = load_files('20newsbydate/20news-bydate-test', encoding='latin1')

# 提取文本特征
tfvectorizer = TfidfVectorizer(max_features=1000)  # 使用1000个词
X_train = tfvectorizer.fit_transform(train_data.data)
y_train = train_data.target

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上预测（示例中仅用于演示，实际上不应使用线性回归进行文本分类）
X_test = tfvectorizer.transform(test_data.data)
y_test = test_data.target
y_pred = model.predict(X_test)

# 计算和打印模型的性能指标（示例中仅用于演示，实际上不适用于文本分类问题）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差 (MSE): {mse}')
print(f'R^2得分: {r2}')

# 可视化观察拟合效果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='-')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('新闻组数据集线性回归拟合')
plt.show()

# 简要分析
# 均方误差 (MSE): 16.639911293409188
# R^2得分: 0.4622926644373311
# 线性回归在文本分类问题上的应用并不合适。
# 预测结果与真实值的散点分布比较分散，MSE较大，R^2得分较低，说明模型无法很好地捕捉文本数据的复杂关系。
