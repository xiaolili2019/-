# 导入相关库
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer将文本数据转换为 TF-IDF 特征表示的工具类
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 加载训练集(无法自动下载,手动下载导入数据)
train_data = load_files('20newsbydate/20news-bydate-train', encoding='latin1') # encoding='latin1' 编码模式

# 加载测试集
test_data = load_files('20newsbydate/20news-bydate-test', encoding='latin1') # encoding='latin1' 编码模式

# 提取文本特征
# 在使用 TfidfVectorizer 进行文本特征提取时，设置 max_features=2000 的意思是限制向量化后的特征词汇表中最多包含 2000 个最重要的
# TfidfVectorizer 将文本数据转换为 TF-IDF 特征矩阵，其中每个文档表示为一个向量，每个词汇表示为矩阵的一列
tfvectorizer = TfidfVectorizer(max_features=2000)  # 最多取2000个
X_train = tfvectorizer.fit_transform(train_data.data)
y_train = train_data.target

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上预测
X_test = tfvectorizer.transform(test_data.data)
y_test = test_data.target
y_pred = model.predict(X_test)

# 计算准确率
accuracy_scores = accuracy_score(y_test, y_pred)
print(f'准确率: {accuracy_scores:.2f}')

# 输出分类报告
print(classification_report(y_test, y_pred, target_names=train_data.target_names))

# 分析:
#  准确率: 0.73
#                           precision    recall  f1-score   support

#              alt.atheism       0.63      0.63      0.63       319
#            comp.graphics       0.62      0.69      0.65       389
#  comp.os.ms-windows.misc       0.69      0.70      0.69       394
# comp.sys.ibm.pc.hardware       0.62      0.61      0.61       392
#    comp.sys.mac.hardware       0.68      0.68      0.68       385
#           comp.windows.x       0.76      0.67      0.71       395
#             misc.forsale       0.77      0.85      0.81       390
#                rec.autos       0.79      0.77      0.78       396
#          rec.motorcycles       0.83      0.86      0.85       398
#       rec.sport.baseball       0.74      0.85      0.79       397
#         rec.sport.hockey       0.87      0.85      0.86       399
#                sci.crypt       0.94      0.82      0.88       396
#          sci.electronics       0.60      0.63      0.62       393
#                  sci.med       0.74      0.69      0.71       396
#                sci.space       0.84      0.85      0.84       394
#   soc.religion.christian       0.77      0.87      0.82       398
#       talk.politics.guns       0.63      0.83      0.71       364
#    talk.politics.mideast       0.93      0.74      0.82       376
#       talk.politics.misc       0.63      0.51      0.56       310
#       talk.religion.misc       0.57      0.43      0.49       251

#                 accuracy                           0.73      7532
#                macro avg       0.73      0.73      0.73      7532
#             weighted avg       0.74      0.73      0.73      7532
# 模型在测试集上的整体准确率为 73%模型对于多类别分类效果不是太好,对不同类别的表现有所不同比如 rec.sport.hockey ,sci.crypt 表现较好.
# 有些类别的精确率高但召回率较低，或者相反，这可能反映了模型在不同类别上的预测策略不同，需要根据具体应用场景进行进一步调整和优化。

# 思路:
# 将文本数据转换为数值特征表示（TF-IDF 特征向量）。
# TfidfVectorizer 使用这个函数 将文本数据转换为 TF-IDF 特征矩阵，其中每个文档表示为一个向量，每个词汇表示为矩阵的一列
# 使用逻辑回归模型进行多类别分类。
# LogisticRegression()
# model.fit(X_train, y_train)

# 通过训练集和测试集的对比，评估模型在未知数据上的泛化能力和分类准确性。
# X_test = vectorizer.transform(test_data.data)
# y_test = test_data.target
# y_pred = model.predict(X_test)